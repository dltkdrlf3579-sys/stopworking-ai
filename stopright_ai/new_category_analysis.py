from __future__ import annotations

import json
import re
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from .case_prepare import row_to_case
from .config import load_config
from .daily import configure_llm_runtime_from_config, format_elapsed
from .db_adapter import load_execute_sql
from .llm_factory import create_llm
from .llm_json import invoke_json
from .logging import install_timestamped_print


FALL_SUBCATEGORIES = {
    "사다리",
    "작업방법",
    "밟음",
    "시설/작업환경",
    "안전고리",
    "작업공간",
    "배관간섭",
    "그레이팅",
}

CONTACT_SUBCATEGORIES = {
    "시설/작업환경",
    "중성화",
    "Leak",
    "냄새",
    "미상의 액체",
    "Powder",
    "응축수",
    "Gas",
    "Fume",
}

OUTPUT_CONTRACT = """

[필수 출력 규칙]
반드시 아래 JSON 객체 하나만 출력한다.
{
  "NEW_대분류": "추락 또는 접액 또는 빈 문자열",
  "NEW_소분류": "선택한 대분류의 허용 소분류 또는 빈 문자열"
}

- NEW_대분류는 "추락", "접액", "" 중 하나만 허용한다.
- NEW_대분류가 "추락"이면 NEW_소분류는 다음 중 하나만 허용한다:
  사다리, 작업방법, 밟음, 시설/작업환경, 안전고리, 작업공간, 배관간섭, 그레이팅
- NEW_대분류가 "접액"이면 NEW_소분류는 다음 중 하나만 허용한다:
  시설/작업환경, 중성화, Leak, 냄새, 미상의 액체, Powder, 응축수, Gas, Fume
- 두 대분류 모두 해당하지 않으면 두 값을 모두 빈 문자열로 출력한다.
- 설명, 근거, 마크다운, 추가 필드는 출력하지 않는다.
"""

INVALID_OUTPUT_RETRY = """

이전 응답은 허용된 분류값을 지키지 않았다.
사건을 다시 분류하고 필수 출력 규칙에 맞는 JSON 객체 하나만 출력하라.
"""

EXCEL_ILLEGAL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
EXCEL_MAX_CELL_CHARS = 32767


def run_new_category_analysis(
    config_path: str = "config.ini",
    query_file: str = "queries/new_category_analysis.sql",
    prompt_file: str = "policies/new_category_prompt.md",
    output_path: str | None = None,
    max_rows: int = 0,
    max_workers: int | None = None,
    llm: Any | None = None,
) -> dict[str, Any]:
    install_timestamped_print()
    config_file = Path(config_path).resolve()
    project_dir = config_file.parent
    config = load_config(str(config_file))
    configure_llm_runtime_from_config(config)

    query_path = resolve_project_path(project_dir, query_file)
    prompt_path = resolve_project_path(project_dir, prompt_file)
    query = read_required_text(query_path, "분석용 SQL")
    prompt = read_required_text(prompt_path, "분류 프롬프트")

    print(f"[new-category] query={query_path}")
    execute_sql = load_execute_sql(config)
    source_df = execute_sql(query)
    if not isinstance(source_df, pd.DataFrame):
        raise TypeError("execute_SQL(query)의 반환값이 pandas DataFrame이 아닙니다.")
    if source_df.empty:
        print("[new-category] 조회 결과가 0건이므로 종료합니다.")
        return {"rows": 0, "errors": 0, "output": ""}

    validate_source_columns(source_df, config)
    if max_rows > 0:
        source_df = source_df.head(max_rows).copy()
    else:
        source_df = source_df.copy()

    if llm is None:
        llm = create_llm(config)

    workers = max_workers
    if workers is None:
        workers = config.getint(
            "new_category_analysis",
            "max_workers",
            fallback=config.getint(
                "daily_prediction",
                "max_workers",
                fallback=config.getint("runtime", "max_workers", fallback=8),
            ),
        )
    workers = max(1, int(workers))

    print(f"[new-category] rows={len(source_df)}, max_workers={workers}")
    classifications, errors = classify_dataframe(source_df, config, llm, prompt, workers)

    result_df = source_df.copy()
    result_df["NEW_대분류"] = [item["major"] for item in classifications]
    result_df["NEW_소분류"] = [item["minor"] for item in classifications]

    output = resolve_output_path(project_dir, output_path)
    write_result_excel(result_df, errors, output, config)

    major_counts = result_df["NEW_대분류"].replace("", "해당없음").value_counts().to_dict()
    print(f"[new-category] completed: output={output}")
    print(f"[new-category] counts={major_counts}, errors={len(errors)}")
    return {
        "rows": int(len(result_df)),
        "errors": int(len(errors)),
        "output": str(output),
        "major_counts": major_counts,
    }


def classify_dataframe(
    df: pd.DataFrame,
    config: Any,
    llm: Any,
    prompt: str,
    max_workers: int,
) -> tuple[list[dict[str, str]], list[dict[str, Any]]]:
    records = list(df.to_dict("records"))
    total = len(records)
    results = [{"major": "", "minor": ""} for _ in records]
    errors: list[dict[str, Any]] = []
    progress_every = config.getint("runtime", "progress_every", fallback=10)
    heartbeat_seconds = config.getint("runtime", "heartbeat_seconds", fallback=30)
    id_column = config.get("columns", "id", fallback="출원번호")
    started = time.monotonic()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_positions = {}
        for position, row in enumerate(records):
            case = row_to_case(row, config)
            future = executor.submit(classify_case_safely, case, llm, prompt)
            future_positions[future] = (position, row.get(id_column, ""))

        pending = set(future_positions)
        done_count = 0
        while pending:
            completed, pending = wait(
                pending,
                timeout=heartbeat_seconds,
                return_when=FIRST_COMPLETED,
            )
            if not completed:
                print(f"[new-category] heartbeat: completed={done_count}/{total}, pending={len(pending)}")
                continue

            for future in completed:
                position, row_id = future_positions[future]
                item = future.result()
                results[position] = {"major": item["major"], "minor": item["minor"]}
                if item["error"]:
                    errors.append(
                        {
                            "원본_행번호": position + 2,
                            "출원번호": row_id,
                            "오류내용": item["error"],
                        }
                    )

                done_count += 1
                if done_count == total or (progress_every > 0 and done_count % progress_every == 0):
                    print(f"[new-category] progress: {done_count}/{total}")

    print(f"[new-category] classification done: elapsed={format_elapsed(time.monotonic() - started)}")
    return results, errors


def classify_case_safely(case: dict[str, Any], llm: Any, prompt: str) -> dict[str, str]:
    try:
        major, minor = classify_case(case, llm, prompt)
        return {"major": major, "minor": minor, "error": ""}
    except Exception as exc:
        return {"major": "", "minor": "", "error": str(exc)}


def classify_case(case: dict[str, Any], llm: Any, prompt: str) -> tuple[str, str]:
    system = prompt.rstrip() + OUTPUT_CONTRACT
    user = build_case_prompt(case)
    last_error: Exception | None = None

    for attempt in range(2):
        response = invoke_json(
            llm,
            system,
            user if attempt == 0 else user + INVALID_OUTPUT_RETRY,
            images=case.get("image_data_urls", []),
        )
        try:
            return normalize_category_response(response)
        except ValueError as exc:
            last_error = exc

    raise ValueError(f"분류값 검증 실패: {last_error}")


def build_case_prompt(case: dict[str, Any]) -> str:
    image_note = "첨부 이미지를 함께 확인한다." if case.get("image_data_urls") else "첨부 이미지가 없다."
    return f"""
[분류 대상]
출원번호: {case.get('id', '')}
제목: {case.get('title', '')}
기존 대분류: {case.get('major', '')}
기존 중분류: {case.get('middle', '')}
현상: {case.get('phenomenon_text', '')}
조치: {case.get('action', '')}
이미지: {image_note}

상세 현상·조치와 이미지에서 직접 확인되는 내용을 기준으로 신규 대분류와 소분류를 선택하라.
""".strip()


def normalize_category_response(response: dict[str, Any]) -> tuple[str, str]:
    major = normalize_major(response.get("NEW_대분류", ""))
    minor = normalize_minor(response.get("NEW_소분류", ""))

    if major == "":
        return "", ""

    allowed = FALL_SUBCATEGORIES if major == "추락" else CONTACT_SUBCATEGORIES
    if minor not in allowed:
        raise ValueError(f"NEW_대분류={major!r}에 허용되지 않은 NEW_소분류={minor!r}")
    return major, minor


def normalize_major(value: Any) -> str:
    text = str(value or "").strip()
    if text in {"", "해당없음", "해당 없음", "없음", "null", "None"}:
        return ""
    if text not in {"추락", "접액"}:
        raise ValueError(f"허용되지 않은 NEW_대분류={text!r}")
    return text


def normalize_minor(value: Any) -> str:
    text = str(value or "").strip()
    compact = re.sub(r"\s+", "", text).lower()
    aliases = {
        "시설/작업환경": "시설/작업환경",
        "leak": "Leak",
        "powder": "Powder",
        "gas": "Gas",
        "fume": "Fume",
    }
    return aliases.get(compact, text)


def validate_source_columns(df: pd.DataFrame, config: Any) -> None:
    columns = config["columns"]
    configured = [
        columns.get("id", "출원번호"),
        columns.get("title", "제목"),
        columns.get("major", "대분류"),
        columns.get("middle", "중분류"),
        columns.get("phenomenon", "현상"),
        columns.get("action", "조치"),
    ]
    missing = [column for column in configured if column not in df.columns]
    if missing:
        raise ValueError(f"분석용 SQL 결과에 필요한 컬럼이 없습니다: {missing}")


def read_required_text(path: Path, label: str) -> str:
    if not path.exists():
        raise FileNotFoundError(f"{label} 파일을 찾지 못했습니다: {path}")
    text = path.read_text(encoding="utf-8").strip()
    meaningful_lines = [line for line in text.splitlines() if line.strip() and not line.lstrip().startswith("--")]
    meaningful_text = "\n".join(meaningful_lines).strip().strip(";").strip()
    if not meaningful_text:
        raise ValueError(f"{label} 파일에 실제 내용이 없습니다: {path}")
    return text


def resolve_project_path(project_dir: Path, value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = project_dir / path
    return path.resolve()


def resolve_output_path(project_dir: Path, output_path: str | None) -> Path:
    if output_path:
        output = resolve_project_path(project_dir, output_path)
    else:
        output = project_dir / "outputs" / f"new_category_analysis_{datetime.now():%Y%m%d_%H%M%S}.xlsx"
    output.parent.mkdir(parents=True, exist_ok=True)
    return output


def write_result_excel(
    result_df: pd.DataFrame,
    errors: list[dict[str, Any]],
    output_path: Path,
    config: Any,
) -> None:
    export_df = make_excel_safe(result_df)
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        export_df.to_excel(writer, sheet_name="분류결과", index=False)
        if errors:
            make_excel_safe(pd.DataFrame(errors)).to_excel(writer, sheet_name="오류목록", index=False)

        worksheet = writer.sheets["분류결과"]
        worksheet.freeze_panes = "A2"
        worksheet.auto_filter.ref = worksheet.dimensions

        id_column = config.get("columns", "id", fallback="출원번호")
        width_targets = {
            id_column: 22,
            "NEW_대분류": 14,
            "NEW_소분류": 20,
        }
        for column_index, column_name in enumerate(export_df.columns, start=1):
            if column_name in width_targets:
                worksheet.column_dimensions[worksheet.cell(1, column_index).column_letter].width = width_targets[column_name]


def make_excel_safe(df: pd.DataFrame) -> pd.DataFrame:
    safe_df = df.copy()
    for column in safe_df.columns:
        if safe_df[column].dtype != object:
            continue
        safe_df[column] = safe_df[column].map(excel_safe_value)
    return safe_df


def excel_safe_value(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    cleaned = EXCEL_ILLEGAL_CHARS.sub("", value)
    if len(cleaned) <= EXCEL_MAX_CELL_CHARS:
        return cleaned
    marker = " ...[Excel 셀 길이 제한으로 이하 생략]"
    return cleaned[: EXCEL_MAX_CELL_CHARS - len(marker)] + marker
