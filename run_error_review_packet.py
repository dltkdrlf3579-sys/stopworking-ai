from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd


JIN = "진성"
GA = "가성"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a compact human/LLM review packet from FN and FPerr prediction rows."
    )
    parser.add_argument("--run-dir", default="", help="Output run directory. Defaults to latest directory under outputs/.")
    parser.add_argument("--prefix", default="train_baseline", help="File prefix, e.g. train_baseline or validation_baseline.")
    parser.add_argument("--max-samples", type=int, default=12, help="Representative samples per error type.")
    parser.add_argument("--sample-chars", type=int, default=900, help="Max characters per long sample field.")
    args = parser.parse_args()

    run_dir = resolve_run_dir(args.run_dir)
    print(f"[review-packet] run_dir={run_dir}", flush=True)

    fperr = load_error_slice(run_dir, args.prefix, "fperr", label=GA, pred=JIN)
    fn = load_error_slice(run_dir, args.prefix, "fn", label=JIN, pred=GA)

    out_md = run_dir / f"{args.prefix}_error_review_packet.md"
    out_xlsx = run_dir / f"{args.prefix}_error_review_samples.xlsx"

    report = build_report(run_dir, args.prefix, fn, fperr, args.max_samples, args.sample_chars)
    out_md.write_text(report, encoding="utf-8")

    with pd.ExcelWriter(out_xlsx) as writer:
        representative_rows(fn, args.max_samples).to_excel(writer, sheet_name="FN_true_as_false", index=False)
        representative_rows(fperr, args.max_samples).to_excel(writer, sheet_name="FP_false_as_true", index=False)
        cluster_table(fn).to_excel(writer, sheet_name="FN_clusters", index=False)
        cluster_table(fperr).to_excel(writer, sheet_name="FPerr_clusters", index=False)

    print(f"[review-packet] markdown={out_md}", flush=True)
    print(f"[review-packet] excel={out_xlsx}", flush=True)
    print(f"[review-packet] FN={len(fn)}, FPerr={len(fperr)}", flush=True)


def resolve_run_dir(value: str) -> Path:
    if value:
        path = Path(value)
        if not path.exists():
            raise FileNotFoundError(f"run-dir not found: {path}")
        return path

    outputs = Path("outputs")
    if not outputs.exists():
        raise FileNotFoundError("outputs/ not found. Pass --run-dir with the actual result folder.")

    candidates = [p for p in outputs.iterdir() if p.is_dir()]
    if not candidates:
        raise FileNotFoundError("No run directories under outputs/. Pass --run-dir.")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def load_error_slice(run_dir: Path, prefix: str, kind: str, label: str, pred: str) -> pd.DataFrame:
    direct = run_dir / f"{prefix}_{kind}.csv"
    if direct.exists():
        return enrich(pd.read_csv(direct))

    pred_path = run_dir / f"{prefix}_predictions.csv"
    if not pred_path.exists():
        raise FileNotFoundError(f"Neither {direct.name} nor {pred_path.name} exists in {run_dir}")

    df = enrich(pd.read_csv(pred_path))
    return df[(df["label_norm"] == label) & (df["pred_norm"] == pred)].copy()


def enrich(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    work = df.copy()
    work["label_norm"] = work.get("label", "").map(normalize_label) if "label" in work else ""
    work["pred_norm"] = work.get("pred", "").map(normalize_label) if "pred" in work else ""
    work["evidence_obj"] = work.get("evidence", "").map(parse_jsonish) if "evidence" in work else [{}] * len(work)
    work["primary_route_evidence"] = work["evidence_obj"].map(lambda x: get_nested(x, "primary_route"))
    if "route_primary" not in work:
        work["route_primary"] = work["primary_route_evidence"]
    work["work_phase"] = work["evidence_obj"].map(lambda x: get_nested(x, "work_phase"))
    work["actual_worker_exposure"] = work["evidence_obj"].map(lambda x: get_nested(x, "actual_worker_exposure"))
    work["simple_correction_possible"] = work["evidence_obj"].map(lambda x: get_nested(x, "simple_correction_possible"))
    work["special_response_or_control"] = work["evidence_obj"].map(lambda x: get_nested(x, "special_response_or_control"))
    work["pipe_approval_status"] = work["evidence_obj"].map(lambda x: get_nested(x, "pipe_support_evidence.approval_status"))
    work["pipe_stepping_context"] = work["evidence_obj"].map(lambda x: get_nested(x, "pipe_support_evidence.stepping_context"))
    work["leak_substance_status"] = work["evidence_obj"].map(lambda x: get_nested(x, "leak_contact_evidence.substance_status_at_stop"))
    work["leak_injury_risk"] = work["evidence_obj"].map(lambda x: get_nested(x, "leak_contact_evidence.injury_risk"))
    work["leak_response_level"] = work["evidence_obj"].map(lambda x: get_nested(x, "leak_contact_evidence.response_level"))
    work["review_bucket"] = work.apply(assign_bucket, axis=1)
    return work


def build_report(
    run_dir: Path,
    prefix: str,
    fn: pd.DataFrame,
    fperr: pd.DataFrame,
    max_samples: int,
    sample_chars: int,
) -> str:
    lines = [
        f"# Error Review Packet",
        "",
        f"- run_dir: `{run_dir}`",
        f"- prefix: `{prefix}`",
        f"- FN true_as_false: `{len(fn)}`",
        f"- FPerr false_as_true: `{len(fperr)}`",
        "",
        "## FPerr Clusters: 정답 가성 / AI 진성",
        "",
    ]
    lines.extend(markdown_table(cluster_table(fperr), ["review_bucket", "major", "middle", "count", "avg_confidence", "top_terms"]))
    lines.extend(["", "## FN Clusters: 정답 진성 / AI 가성", ""])
    lines.extend(markdown_table(cluster_table(fn), ["review_bucket", "major", "middle", "count", "avg_confidence", "top_terms"]))

    lines.extend(["", "## Representative FPerr Samples", ""])
    lines.extend(sample_blocks(representative_rows(fperr, max_samples), sample_chars))
    lines.extend(["", "## Representative FN Samples", ""])
    lines.extend(sample_blocks(representative_rows(fn, max_samples), sample_chars))

    lines.extend(
        [
            "",
            "## How To Use",
            "",
            "나에게 전부 주지 말고 이 파일에서 아래만 복붙하면 된다.",
            "",
            "1. FPerr Clusters 표 상위 10줄",
            "2. FN Clusters 표 상위 10줄",
            "3. Representative FPerr Samples 중 헷갈리는 5개",
            "4. Representative FN Samples 중 헷갈리는 5개",
        ]
    )
    return "\n".join(lines) + "\n"


def cluster_table(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["review_bucket", "major", "middle", "count", "avg_confidence", "top_terms"])

    group_cols = [col for col in ["review_bucket", "major", "middle"] if col in df.columns]
    rows = []
    for key, group in df.groupby(group_cols, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        row = dict(zip(group_cols, key))
        row["count"] = int(len(group))
        row["avg_confidence"] = round(to_numeric(group.get("confidence", pd.Series(dtype=float))).mean(), 1)
        row["top_terms"] = ", ".join(top_terms(group, limit=12))
        rows.append(row)
    result = pd.DataFrame(rows)
    return result.sort_values(["count", "avg_confidence"], ascending=[False, False]).reset_index(drop=True)


def representative_rows(df: pd.DataFrame, max_samples: int) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    sort_cols = []
    ascending = []
    if "confidence" in df:
        df = df.copy()
        df["_confidence_num"] = to_numeric(df["confidence"])
        sort_cols.append("_confidence_num")
        ascending.append(False)
    for col in ["review_bucket", "major", "middle"]:
        if col in df:
            sort_cols.append(col)
            ascending.append(True)

    work = df.sort_values(sort_cols, ascending=ascending) if sort_cols else df.copy()
    selected = []
    seen_groups: dict[tuple, int] = {}
    for _, row in work.iterrows():
        group = tuple(row.get(col, "") for col in ["review_bucket", "major", "middle"])
        if seen_groups.get(group, 0) >= 2:
            continue
        selected.append(row)
        seen_groups[group] = seen_groups.get(group, 0) + 1
        if len(selected) >= max_samples:
            break

    if len(selected) < max_samples:
        selected_ids = {str(row.get("id", "")) for row in selected}
        for _, row in work.iterrows():
            if str(row.get("id", "")) in selected_ids:
                continue
            selected.append(row)
            if len(selected) >= max_samples:
                break

    return pd.DataFrame(selected).drop(columns=["_confidence_num"], errors="ignore")


def sample_blocks(df: pd.DataFrame, sample_chars: int) -> list[str]:
    if df.empty:
        return ["없음"]

    lines = []
    for idx, row in enumerate(df.to_dict("records"), start=1):
        evidence = row.get("evidence_obj", {})
        lines.extend(
            [
                f"### Sample {idx}",
                "",
                f"- id: `{row.get('id', '')}`",
                f"- label/pred: `{row.get('label', '')}` / `{row.get('pred', '')}`",
                f"- major/middle: `{row.get('major', '')}` / `{row.get('middle', '')}`",
                f"- bucket: `{row.get('review_bucket', '')}`",
                f"- confidence: `{row.get('confidence', '')}`",
                f"- title: {compact(row.get('title', ''), 220)}",
                f"- reason: {compact(row.get('reason', ''), sample_chars)}",
                f"- decisive_evidence: {compact(row.get('decisive_evidence', ''), sample_chars)}",
                f"- key_evidence: {compact(get_nested(evidence, 'key_evidence'), sample_chars)}",
                f"- visual_evidence: {compact(get_nested(evidence, 'visual_evidence'), sample_chars)}",
                f"- pipe_support_evidence: {compact(get_nested(evidence, 'pipe_support_evidence'), sample_chars)}",
                f"- leak_contact_evidence: {compact(get_nested(evidence, 'leak_contact_evidence'), sample_chars)}",
                "",
            ]
        )
    return lines


def assign_bucket(row: pd.Series) -> str:
    text = " ".join(
        str(row.get(col, ""))
        for col in [
            "route_primary",
            "primary_route_evidence",
            "major",
            "middle",
            "title",
            "reason",
            "decisive_evidence",
        ]
    )
    evidence = row.get("evidence_obj", {})
    text += " " + compact(get_nested(evidence, "key_evidence"), 2000)
    text += " " + compact(get_nested(evidence, "pipe_support_evidence"), 2000)
    text += " " + compact(get_nested(evidence, "leak_contact_evidence"), 2000)

    lowered = text.lower()
    if any(token in text for token in ["지정 도구", "특정 도구", "대체 도구", "다른 도구", "대체 작업방법", "규정위반", "안전기준", "손들기"]):
        return "standard_rule_deviation"
    if any(token in text for token in ["배관", "서포트", "밟", "발판", "Toxic Duct", "덕트"]):
        return "pipe_support_or_access"
    if any(token in text for token in ["누출", "접액", "DIW", "응축수", "미상", "냄새", "가스", "방제", "ERT"]):
        return "leak_contact"
    if any(token in text for token in ["추락", "낙하", "고소", "사다리", "개구부", "난간", "그레이팅"]):
        return "height_fall"
    if any(token in lowered for token in ["ppe", "보호구", "안전고리", "안전대"]):
        return "ppe"
    if any(token in text for token in ["허가", "SOP", "서류", "협의", "일정", "작업예정", "사전"]):
        return "admin_prework"
    return "general"


def top_terms(df: pd.DataFrame, limit: int = 12) -> list[str]:
    text_parts = []
    for col in ["title", "major", "middle", "reason", "decisive_evidence", "evidence"]:
        if col in df:
            text_parts.extend(df[col].fillna("").astype(str).tolist())
    text = " ".join(text_parts)
    tokens = re.findall(r"[가-힣A-Za-z0-9_/]{2,}", text)
    stop = {
        "작업",
        "확인",
        "진성",
        "가성",
        "판단",
        "경우",
        "위험",
        "상세",
        "내용",
        "근거",
        "이미지",
        "불명확",
        "해당없음",
    }
    counts: dict[str, int] = {}
    for token in tokens:
        if token in stop:
            continue
        counts[token] = counts.get(token, 0) + 1
    return [term for term, _ in sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:limit]]


def markdown_table(df: pd.DataFrame, cols: list[str], max_rows: int = 20) -> list[str]:
    if df.empty:
        return ["없음"]
    view = df.head(max_rows).copy()
    cols = [col for col in cols if col in view.columns]
    lines = ["|" + "|".join(cols) + "|", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, row in view.iterrows():
        values = [compact(row.get(col, ""), 180).replace("|", "/") for col in cols]
        lines.append("|" + "|".join(values) + "|")
    return lines


def parse_jsonish(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return value
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return {}
    text = str(value).strip()
    if not text:
        return {}
    try:
        return json.loads(text)
    except Exception:
        return {}


def get_nested(value: Any, path: str) -> Any:
    current = value
    for part in path.split("."):
        if isinstance(current, dict):
            current = current.get(part, "")
        else:
            return ""
    return current


def normalize_label(value: Any) -> str:
    text = "" if value is None else str(value).strip()
    if "진" in text or "\uf9de" in text:
        return JIN
    if "가" in text:
        return GA
    return text


def to_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(0)


def compact(value: Any, limit: int) -> str:
    if isinstance(value, (dict, list)):
        text = json.dumps(value, ensure_ascii=False)
    else:
        text = "" if value is None else str(value)
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > limit:
        return text[:limit].rstrip() + "..."
    return text


if __name__ == "__main__":
    main()
