from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta, timezone
from html import escape
from pathlib import Path
from typing import Any

import pandas as pd

from .config import load_config
from .daily import configure_llm_runtime_from_config
from .db_adapter import load_execute_sql
from .html_extract import html_to_text, remove_embedded_assets
from .llm_factory import create_llm
from .llm_json import invoke_json
from .logging import install_timestamped_print


KST = timezone(timedelta(hours=9))
FIELDS = {
    "id": "출원번호", "company": "업체명", "date": "등록일", "prediction": "AI판정",
    "title": "제목", "major": "대분류", "phenomenon": "현상", "action": "조치",
    "reason": "판단근거", "error": "에러여부", "order": "판정일시",
}
PART_SCHEMA = '''
[분할 분석 출력]
{"observations": [{"title": "주요 등록 사유", "detail": "원문에서 확인한 사실과 판정 사유 2문장 이내", "ids": ["출원번호"]}]}
이 호출에는 한 업체의 사례만 제공된다. 핵심 관찰 1~4개만 출력한다.
가성 사례가 있다면 등록 계기와 기존 AI의 가성 판정 근거를 분리해 설명한다.
"협의가 많음" 같은 단어 요약에서 끝내지 말고 무엇 때문에 어떤 조치·협의를 했는지 설명한다.
빈도 수치를 만들지 말고, 각 관찰에 실제 입력 출원번호를 1~5개 연결한다.
'''
FINAL_SCHEMA = '''
[최종 보고서 출력]
{
  "overview": "보고 기간 전체 현황을 2~3문장으로 요약",
  "issues": [{"title": "주요 이슈", "detail": "확인된 현황 2문장 이내", "ids": ["출원번호"]}],
  "companies": [{"company": "업체명", "detail": "가성 등록의 주요 사유와 확인할 사항 2문장 이내", "ids": ["출원번호"]}],
  "actions": ["후속 확인 권고"]
}
issues는 최대 4개, companies는 최대 6개, actions는 최대 3개로 작성한다.
업체별 표는 코드가 별도로 표시하므로 표를 출력하지 않는다.
companies에는 가성 건수가 많은 업체를 우선 설명하되, 제공 근거가 있는 업체만 쓴다.
통계에는 전수 집계만 사용한다. 분할 관찰은 대표 근거이지 주제별 전체 빈도가 아니다.
'''


def text(value: Any) -> str:
    return "" if value is None or pd.isna(value) else str(value).strip()


def local_day(value: Any) -> date:
    stamp = pd.Timestamp(value)
    if pd.isna(stamp):
        raise ValueError("등록일 또는 판정일시가 비어 있습니다.")
    if stamp.tzinfo is not None:
        stamp = stamp.tz_convert(KST)
    return stamp.date()


def resolve_period(target_date: str | None = None, start_date: str | None = None,
                   end_date: str | None = None) -> tuple[date, date]:
    if target_date is not None and (start_date is not None or end_date is not None):
        raise ValueError("--target-date와 --start-date/--end-date는 함께 사용할 수 없습니다.")
    if (start_date is None) != (end_date is None):
        raise ValueError("기간 보고는 --start-date와 --end-date를 모두 지정하세요.")
    if start_date is not None:
        start, end = date.fromisoformat(start_date), date.fromisoformat(end_date)
    else:
        start = end = date.fromisoformat(target_date) if target_date is not None else datetime.now(KST).date() - timedelta(days=1)
    if start > end:
        raise ValueError("시작일은 종료일보다 늦을 수 없습니다.")
    return start, end


def prepare_rows(df: pd.DataFrame, config: Any, target: date,
                 end_date: date | None = None) -> tuple[pd.DataFrame, dict]:
    end = end_date or target
    if target > end:
        raise ValueError("시작일은 종료일보다 늦을 수 없습니다.")
    if not isinstance(df, pd.DataFrame) or not df.columns.is_unique:
        raise ValueError("쿼리 결과는 중복 컬럼명이 없는 DataFrame이어야 합니다.")
    columns = {key: config.get("columns", key, fallback=name) for key, name in FIELDS.items()}
    missing = [columns[k] for k in ("id", "company", "date", "prediction") if columns[k] not in df.columns]
    if missing:
        raise ValueError(f"보고서 쿼리 필수 컬럼 누락: {missing}. SQL 오류 여부와 컬럼 매핑을 확인하세요.")
    data = pd.DataFrame({key: df[col] if col in df.columns else "" for key, col in columns.items()}, index=df.index)
    queried = len(data)
    days = data["date"].map(local_day)
    mask = days.map(lambda day: target <= day <= end).astype(bool)
    data = data.loc[mask].copy()
    data["date"] = days.loc[mask].map(lambda day: day.isoformat())
    outside = queried - len(data)
    data["id"] = data["id"].map(text)
    if data["id"].eq("").any():
        raise ValueError("출원번호가 빈 행이 있습니다. 중복 집계 방지를 위해 ID를 확인하세요.")
    before = len(data)
    if data["id"].duplicated().any():
        # 이력 조인 결과라면 최신 시각이 명확해야 하나의 판정만 집계할 수 있다.
        stamps = pd.to_datetime(data["order"], errors="coerce", utc=True)
        duplicate = data["id"].duplicated(keep=False)
        if stamps[duplicate].isna().any():
            raise ValueError("중복 출원번호가 있지만 판정일시가 없습니다. SQL에서 최신 한 건으로 조회하세요.")
        data["_order"] = stamps
        if data.loc[duplicate].duplicated(["id", "_order"], keep=False).any():
            raise ValueError("동일 출원번호·판정일시가 중복됩니다. SQL에서 최신 한 건으로 조회하세요.")
        data = data.sort_values("_order").drop_duplicates("id", keep="last").drop(columns="_order")
    for key in data.columns:
        # Empty filtered columns can retain numeric/boolean DB dtypes.
        data[key] = data[key].map(text).astype(str)
    data["company"] = data["company"].replace("", "업체명 미확인")
    data["major"] = data["major"].replace("", "미분류")
    data["prediction"] = data["prediction"].where(data["prediction"].isin(["진성", "가성"]), "보류/미판정")
    error = data["error"].str.lower().isin(["true", "1", "1.0", "y", "yes", "오류", "에러"])
    data.loc[error, "prediction"] = "보류/미판정"
    return data.sort_values(["company", "id"]).reset_index(drop=True), {
        "queried_rows": queried, "outside_date_rows": outside, "duplicate_rows_removed": before - len(data),
    }


def counts(group: pd.DataFrame) -> dict:
    true = int(group["prediction"].eq("진성").sum())
    false = int(group["prediction"].eq("가성").sum())
    valid = true + false
    return {"total": len(group), "true": true, "false": false, "pending": len(group) - valid,
            "false_pct": round(false / valid * 100, 1) if valid else None}


def aggregate(data: pd.DataFrame, target: date, metadata: dict, end_date: date | None = None) -> dict:
    end = end_date or target
    companies = [{"company": name, **counts(group)} for name, group in data.groupby("company", sort=True)]
    companies.sort(key=lambda c: (-c["false"], -c["total"], c["company"]))
    issues = [{"major": name, **counts(group)} for name, group in data.groupby("major", sort=True)]
    issues.sort(key=lambda i: (-i["total"], i["major"]))
    return {"date": target.isoformat(), "start_date": target.isoformat(), "end_date": end.isoformat(),
            "period_label": target.isoformat() if target == end else f"{target} ~ {end}",
            "report_type": "일일" if target == end else "기간",
            "overall": counts(data), "companies": companies,
            "issues": issues, "data_checks": metadata}


def case_batches(data: pd.DataFrame, max_chars: int, field_chars: int) -> list[list[dict]]:
    batches, batch, size = [], [], 2
    for row in data.to_dict("records"):
        case = {}
        for key in ("id", "company", "date", "prediction", "major", "title", "phenomenon", "action", "reason"):
            value = row[key]
            if key in {"title", "phenomenon", "action", "reason"}:
                value = html_to_text(remove_embedded_assets(value))
                if len(value) > field_chars:
                    value = value[:field_chars] + " [이하 생략]"
            case[key] = value
        length = len(json.dumps(case, ensure_ascii=False)) + 2
        if length > max_chars:
            raise ValueError("단일 사례가 분할 입력 한도를 초과했습니다. 필드 길이 설정을 확인하세요.")
        if batch and size + length > max_chars:
            batches.append(batch)
            batch, size = [], 2
        batch.append(case)
        size += length
    if batch:
        batches.append(batch)
    return batches


def validate_notes(notes: Any, allowed_ids: set[str], *, company_ids: dict | None = None) -> None:
    if not isinstance(notes, list):
        raise ValueError("AI 보고서의 관찰 목록은 배열이어야 합니다.")
    for note in notes:
        if not isinstance(note, dict) or not isinstance(note.get("detail"), str) or not note["detail"].strip():
            raise ValueError("AI 보고서 관찰 문장이 비어 있습니다.")
        label = "company" if company_ids is not None else "title"
        if not isinstance(note.get(label), str) or not note[label].strip():
            raise ValueError(f"AI 보고서의 {label}이 없습니다.")
        ids = note.get("ids")
        if not isinstance(ids, list) or not ids or any(not isinstance(i, str) for i in ids):
            raise ValueError("AI 관찰에 근거 출원번호가 없습니다.")
        if not set(ids) <= allowed_ids:
            raise ValueError("AI가 입력에 없는 출원번호를 근거로 사용했습니다.")
        if company_ids is not None and not set(ids) <= company_ids.get(note["company"], set()):
            raise ValueError("AI가 다른 업체의 사례를 업체 분석 근거로 사용했습니다.")


def analyse(data: pd.DataFrame, stats: dict, prompt: str, llm: Any, config: Any,
            partial_file: Path | None = None) -> tuple[dict, list]:
    max_chars = config.getint("report", "max_batch_chars", fallback=24000)
    field_chars = config.getint("report", "max_case_chars", fallback=800)
    if max_chars < 1000 or field_chars < 50:
        raise ValueError("max_batch_chars는 1000 이상, max_case_chars는 50 이상이어야 합니다.")
    batches = [batch for _, group in data.groupby("company", sort=True)
               for batch in case_batches(group, max_chars, field_chars)]
    partials = [None] * len(batches)

    def analyse_batch(batch: list[dict]) -> dict:
        payload = {"start_date": stats["start_date"], "end_date": stats["end_date"], "cases": batch}
        part = invoke_json(llm, prompt + PART_SCHEMA, json.dumps(payload, ensure_ascii=False))
        validate_notes(part.get("observations"), {c["id"] for c in batch})
        if not 1 <= len(part["observations"]) <= 4:
            raise ValueError("분할 분석 관찰은 1~4개여야 합니다.")
        return {"company": batch[0]["company"], "analysed_rows": len(batch), "observations": part["observations"]}

    workers = max(1, config.getint("report", "max_workers", fallback=4))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(analyse_batch, batch): i for i, batch in enumerate(batches)}
        for done, future in enumerate(as_completed(futures), 1):
            partials[futures[future]] = future.result()
            if partial_file:
                save_json(partial_file, [p for p in partials if p is not None])
            print(f"[daily-report] AI analysis {done}/{len(batches)} completed", flush=True)
    print("[daily-report] composing report", flush=True)
    ownership = dict(zip(data["id"], data["company"]))
    payload = json.dumps({"statistics": stats, "observations": partials, "case_companies": ownership}, ensure_ascii=False)
    if len(payload) > config.getint("report", "max_final_chars", fallback=100000):
        raise ValueError("최종 종합 입력 한도를 초과했습니다. summary.json과 partial_analysis.json을 확인하세요.")
    result = invoke_json(llm, prompt + FINAL_SCHEMA, payload)
    if not isinstance(result.get("overview"), str) or not result["overview"].strip():
        raise ValueError("AI 보고서 요약이 비어 있습니다.")
    company_ids = {name: set(g["id"]) for name, g in data.groupby("company")}
    validate_notes(result.get("issues"), set(data["id"]))
    validate_notes(result.get("companies"), set(data["id"]), company_ids=company_ids)
    actions = result.get("actions")
    if not isinstance(actions, list) or any(not isinstance(x, str) for x in actions):
        raise ValueError("후속 확인사항은 문자열 배열이어야 합니다.")
    if len(result["issues"]) > 4 or len(result["companies"]) > 6 or len(actions) > 3:
        raise ValueError("AI 보고서 항목 수가 출력 규칙을 초과했습니다.")
    return result, partials


def render_html(stats: dict, narrative: dict, demo: bool = False) -> str:
    esc = lambda value: escape(str(value), quote=True)
    period = stats.get("period_label", stats["date"])
    report_type = stats.get("report_type", "일일")
    summary_title = "오늘의 요약" if report_type == "일일" else "기간 요약"
    def table(headers: list, rows: list) -> str:
        head = "".join(f'<th style="padding:11px 10px;text-align:left;border-bottom:2px solid #cad5d7;background:#edf3f3">{esc(h)}</th>' for h in headers)
        body = "".join("<tr>" + "".join(f'<td style="padding:10px;border-bottom:1px solid #e1e5e7">{esc(c)}</td>' for c in row) + "</tr>" for row in rows)
        return f'<table style="width:100%;border-collapse:collapse;font-size:14px;margin:16px 0"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>'
    def notes(items: list, key: str) -> str:
        return "".join(f'<h3 style="font-size:16px;margin:20px 0 6px">{esc(n[key])}</h3><p style="margin:0">{esc(n["detail"])}</p><p style="font-size:12px;color:#64716e;margin:5px 0">근거: {esc(", ".join(n["ids"]))}</p>' for n in items)
    overall = stats["overall"]
    metric_rows = [[overall["total"], overall["true"], overall["false"], overall["pending"]]]
    company_rows = [[c["company"], c["total"], c["true"], c["false"], c["pending"],
                     "-" if c["false_pct"] is None else f'{c["false_pct"]:.1f}%'] for c in stats["companies"]]
    issue_rows = [[c["major"], c["total"], c["true"], c["false"]] for c in stats["issues"]]
    demo_banner = '<p style="background:#fff1d6;padding:12px;font-weight:bold">시연용 가상 데이터 · 예시 문장 (DB/API 미호출)</p>' if demo else ""
    action_html = "".join(f'<li style="margin-bottom:8px">{esc(a)}</li>' for a in narrative["actions"])
    return f'''<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>{esc(period)} 작업중지권 {esc(report_type)} 보고서</title></head>
<body style="margin:0;background:#f2f4f5;color:#243331;font-family:'Malgun Gothic',Arial,sans-serif;line-height:1.7">
<main style="max-width:980px;margin:0 auto;background:#fff;padding:28px 24px">
{demo_banner}<p style="color:#087e79;font-weight:bold;margin:0">안전관리 {esc(report_type)} 현황</p>
<h1 style="font-size:26px;margin:6px 0">{esc(period)} 작업중지권 보고서</h1>
<p style="color:#65736e;margin:0">대상 기간 등록 사례 · 업체별 AI 판정 및 주요 이슈</p>
{table(['전체 등록', 'AI 진성', 'AI 가성', '보류·미판정'], metric_rows)}
<h2 style="font-size:19px;border-top:2px solid #087e79;padding-top:20px">{summary_title}</h2><p>{esc(narrative['overview'])}</p>
<h2 style="font-size:19px;margin-top:30px">등록 유형별 현황</h2>{table(['등록 분류', '전체', 'AI 진성', 'AI 가성'], issue_rows)}
{notes(narrative['issues'], 'title')}
<h2 style="font-size:19px;margin-top:30px">업체별 현황</h2><p style="font-size:12px;color:#65736e">AI 가성 건수 내림차순 · 가성 비율 = 가성 / (진성 + 가성), 보류·미판정 제외</p>
{table(['업체명', '전체', 'AI 진성', 'AI 가성', '보류', '가성 비율'], company_rows)}
{notes(narrative['companies'], 'company')}
<h2 style="font-size:19px;margin-top:30px">후속 확인사항</h2><ul>{action_html}</ul>
<p style="font-size:12px;color:#65736e;border-top:1px solid #dce2e0;padding-top:18px">기존 AI 판정 결과를 요약한 자료입니다. 집계는 전체 대상 건 기준이며 서술은 현상·조치·판단 근거의 요약입니다. AI 가성 건수만으로 업체의 안전수준을 평가하지 않습니다.</p>
</main></body></html>'''


def demo_data(target: date, end_date: date | None = None) -> pd.DataFrame:
    days = ((end_date or target) - target).days + 1
    specs = [
        ("시연 A설비", "배관/서포트", 8, 5, "협소 구간 배관 밟음 협의", "작업방법 협의 후 작업 진행"),
        ("시연 B시공", "서류/행정", 12, 2, "작업허가서 서명 누락", "서류 보완 후 작업 진행"),
        ("시연 C정비", "누출/접촉", 6, 4, "작업 구간 미상 액체 발견", "담당부서 확인 및 구역 통제"),
    ]
    rows = []
    for company, major, total, true_count, phenomenon, action in specs:
        for i in range(total):
            rows.append({"출원번호": f"DEMO-{len(rows)+1:04d}", "업체명": company,
                         "등록일": (target + timedelta(days=len(rows) % days)).isoformat(), "AI판정": "진성" if i < true_count else "가성",
                         "제목": phenomenon, "대분류": major, "현상": phenomenon,
                         "조치": action, "판단근거": "시연용 가상 판정", "에러여부": False})
    return pd.DataFrame(rows)


def demo_narrative(data: pd.DataFrame, stats: dict) -> dict:
    top = stats["companies"][0]
    return {
        "overview": f'시연용 가상 사례 {len(data)}건을 집계했습니다. 서류 보완 및 배관 밟음 관련 등록을 확인할 수 있습니다.',
        "issues": [{"title": "작업 준비 과정의 서류 확인", "detail": "작업허가서 서명 누락과 보완 사례를 예시로 구성했습니다.", "ids": ["DEMO-0009"]}],
        "companies": [{"company": top["company"], "detail": f'전체 {top["total"]}건 중 AI 가성 {top["false"]}건입니다. 예시 기록에는 허가서 서명 보완이 기재되어 있습니다.', "ids": ["DEMO-0011"]}],
        "actions": ["작업 전 서류 확인 절차 안내 검토", "실제 데이터 연결 후 업체별 사유와 집계 확인"],
    }


def run_report(config_path: str = "daily_report.ini", target_date: str | None = None,
               *, start_date: str | None = None, end_date: str | None = None,
               demo: bool = False, llm: Any = None) -> Path:
    install_timestamped_print()
    start, end = resolve_period(target_date, start_date, end_date)
    config_file = Path(config_path).resolve()
    config = load_config(str(config_file))
    root = config_file.parent
    path = lambda key, fallback: (root / config.get("report", key, fallback=fallback)).resolve()
    period_key = start.isoformat() if start == end else f"{start}_{end}"
    output = path("output_dir", "outputs/daily_reports") / f'{period_key}_{datetime.now(KST):%H%M%S_%f}'
    output.mkdir(parents=True, exist_ok=False)
    api = None
    if not demo:
        api = load_config(str(path("api_config", "config.ini")))
    print(f"[daily-report] period={start}~{end}, output={output}", flush=True)
    if demo:
        df = demo_data(start, end).rename(columns={name: config.get("columns", key, fallback=name)
                                             for key, name in FIELDS.items()})
    else:
        query = path("query_file", "queries/daily_company_report.sql").read_text(encoding="utf-8-sig")
        if not any(line.strip().strip(";") for line in query.splitlines() if not line.lstrip().startswith("--")):
            raise ValueError("보고서 전용 SQL에 실제 조회 쿼리를 입력하세요.")
        dates = {"target_date": start, "start_date": start, "end_date": end, "next_date": end + timedelta(days=1)}
        for key, value in dates.items():
            query = query.replace("{" + key + "}", value.isoformat())
        df = load_execute_sql(api)(query)
    data, metadata = prepare_rows(df, config, start, end)
    stats = aggregate(data, start, metadata, end)
    save_json(output / "summary.json", stats)
    print(f"[daily-report] queried_rows={metadata['queried_rows']}, outside_date_rows={metadata['outside_date_rows']}, "
          f"rows={len(data)}, companies={len(stats['companies'])}", flush=True)
    partials = []
    if demo:
        narrative = demo_narrative(data, stats)
    elif data.empty:
        narrative = {"overview": "대상 기간에 조회된 작업중지권 사례가 없습니다.", "issues": [], "companies": [], "actions": []}
    else:
        prompt = path("prompt_file", "policies/daily_company_report.md").read_text(encoding="utf-8-sig")
        if not prompt.strip():
            raise ValueError("보고서 프롬프트가 비어 있습니다.")
        configure_llm_runtime_from_config(api)
        try:
            narrative, partials = analyse(data, stats, prompt, llm or create_llm(api), config,
                                          output / "partial_analysis.json")
        except Exception as exc:
            save_json(output / "error.json", {"error": str(exc), "date": start.isoformat(),
                                               "start_date": start.isoformat(), "end_date": end.isoformat()})
            print(f"[daily-report] report failed; summary retained: {output}", flush=True)
            raise
    save_json(output / "partial_analysis.json", partials)
    save_json(output / "analysis.json", narrative)
    html = render_html(stats, narrative, demo)
    report_file = output / "report.html"
    report_file.write_text(html, encoding="utf-8")
    subject = f'{"[시연] " if demo else ""}[작업중지권] {stats["period_label"]} 업체별 {stats["report_type"]} 보고서'
    save_json(output / "mail_content.json", {"subject": subject, "html_body": html,
                                             "text_summary": narrative["overview"], "demo": demo})
    print(f"[daily-report] done: {report_file}", flush=True)
    return report_file


def save_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
