from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


JIN = "\uc9c4\uc131"
GA = "\uac00\uc131"
BORYU = "\ubcf4\ub958"

GENERATED_NAMES = {
    "combined_predictions.csv",
    "error_cases.csv",
    "error_clusters.csv",
    "representative_errors.csv",
    "recurring_errors.csv",
}

KEYWORD_GROUPS = [
    ("pipe_step", ["\ubc30\uad00", "\uc11c\ud3ec\ud2b8", "\ubc1f", "\uc9c8\uc18c", "\uc815\uc81c\uc9c8\uc18c", "\ucf00\ubbf8\uceec"]),
    ("physical_interference", ["\uac04\uc12d", "\ub3d9\uc120", "\uacf5\uac04", "\ud611\uc18c", "\ucda9\ub3cc", "\ub07c\uc784", "\uc811\ucd09"]),
    ("contractor_complete", ["\ud611\ub825\ud68c\uc0ac", "\uc790\uccb4", "\uac1c\uc120\uc644\ub8cc", "\uc870\uce58\uc644\ub8cc"]),
    ("ppe", ["PPE", "\ubcf4\ud638\uad6c", "\ubbf8\ucc29\uc6a9", "\ucc29\uc6a9", "\uad50\uccb4"]),
    ("precheck", ["\uc791\uc5c5\uc804", "\uc0ac\uc804", "\uc810\uac80", "\ud655\uc778", "\uc785\uace0"]),
    ("admin", ["\ud5c8\uac00", "\uc11c\ub958", "SOP", "\ud589\uc815", "\ud611\uc758", "\uc77c\uc815", "\ub300\uae30"]),
    ("height_fall", ["\uace0\uc18c", "\ucd94\ub77d", "\ub09c\uac04", "\ubc1c\ud310", "\uc0ac\ub2e4\ub9ac", "\uc544\uc6c3\ud2b8\ub9ac\uac70"]),
    ("chemical_process", ["\ub204\ucd9c", "\ud654\ud559", "\uc720\ud574", "\uac00\uc2a4", "\uc9c8\uc2dd", "\uc0b0\uc18c", "LEL", "\uc911\uc131\ud654", "\ud37c\uc9c0", "\uce58\ud658"]),
    ("electricity", ["\uac10\uc804", "\ud65c\uc120", "\ub204\uc804", "\uc2a4\ud30c\ud06c", "\uc804\uc6d0", "\ucc28\ub2e8"]),
    ("fire_explosion", ["\ud654\uc7ac", "\ud3ed\ubc1c", "\ubd88\uaf43", "\ubc1c\ud654", "\uc778\ud654"]),
    ("static_defect", ["\ud53c\ubcf5", "\ud0c4\ud654", "\ubcfc\ud2b8", "\ud30c\uc190", "\uade0\uc5f4", "\ud6fc\uc190", "\ubd88\ub7c9"]),
    ("process_parameter", ["\uc555\ub825", "\uc628\ub3c4", "\uc54c\ub78c", "\uc778\ud130\ub85d", "\uac10\uc555", "\ub0c9\uac01"]),
    ("housekeeping", ["\uc815\ub9ac\uc815\ub3c8", "\ud1b5\ub85c", "\uccad\uc18c", "\uc801\uce58"]),
    ("weather", ["\uc6b0\ucc9c", "\uac15\ud48d", "\ud3ed\uc5fc", "\uacb0\ube59", "\uc801\uc124", "\uae30\uc0c1"]),
]

STOPWORDS = {
    "true",
    "false",
    "none",
    "null",
    "step",
    "json",
    "work",
    "risk",
    "\uc5c6\uc74c",
    "\uc788\uc74c",
    "\ubd88\uba85",
    "\uc791\uc5c5",
    "\uc870\uce58",
    "\ud310\ub2e8",
    "\uacbd\uc6b0",
    "\uc0c1\uc138",
    "\ub0b4\uc6a9",
    "\uadfc\uac70",
    "\uc0ac\ub840",
    "\uc704\ud5d8",
}


def analyze_prediction_outputs(
    inputs: list[str | Path],
    output_root: str | Path = "artifacts/prediction_analysis",
    max_clusters: int = 20,
    samples_per_cluster: int = 5,
    policy_path: str | Path | None = "policies/current_policy.md",
    max_policy_chars: int = 30000,
) -> dict[str, Path]:
    files = discover_prediction_files(inputs)
    if not files:
        raise FileNotFoundError("No prediction CSV/XLSX files found.")

    out_dir = Path(output_root) / datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)

    combined = load_prediction_files(files)
    normalized = normalize_predictions(combined)
    eligible = normalized[~normalized["excluded_norm"]].copy()
    errors = eligible[eligible["correct_norm"] == False].copy()

    clusters_df, cluster_payloads = build_error_cluster_outputs(
        eligible,
        errors,
        max_clusters=max_clusters,
        samples_per_cluster=samples_per_cluster,
    )
    recurring_df = build_recurring_errors(errors)
    representatives_df = build_representative_errors(cluster_payloads)
    metrics = compute_overall_metrics(normalized, eligible, errors, files)
    policy_text = load_policy_text(policy_path, max_policy_chars=max_policy_chars)
    llm_brief = build_llm_brief(metrics, cluster_payloads, policy_text, policy_path)

    paths = {
        "combined_predictions": out_dir / "combined_predictions.csv",
        "error_cases": out_dir / "error_cases.csv",
        "error_clusters": out_dir / "error_clusters.csv",
        "representative_errors": out_dir / "representative_errors.csv",
        "recurring_errors": out_dir / "recurring_errors.csv",
        "llm_cluster_brief": out_dir / "llm_cluster_brief.json",
        "llm_analysis_prompt": out_dir / "llm_analysis_prompt.md",
        "policy_snapshot": out_dir / "policy_snapshot.md",
        "report": out_dir / "error_analysis_report.md",
    }

    normalized.to_csv(paths["combined_predictions"], index=False, encoding="utf-8-sig")
    errors.to_csv(paths["error_cases"], index=False, encoding="utf-8-sig")
    clusters_df.to_csv(paths["error_clusters"], index=False, encoding="utf-8-sig")
    representatives_df.to_csv(paths["representative_errors"], index=False, encoding="utf-8-sig")
    recurring_df.to_csv(paths["recurring_errors"], index=False, encoding="utf-8-sig")
    paths["llm_cluster_brief"].write_text(json.dumps(llm_brief, ensure_ascii=False, indent=2), encoding="utf-8")
    paths["llm_analysis_prompt"].write_text(build_llm_prompt(llm_brief), encoding="utf-8")
    paths["policy_snapshot"].write_text(policy_text or "Policy file not found or empty.", encoding="utf-8")
    paths["report"].write_text(build_markdown_report(metrics, clusters_df, recurring_df, paths), encoding="utf-8")

    return paths


def discover_prediction_files(inputs: list[str | Path]) -> list[Path]:
    found: list[Path] = []
    for item in inputs:
        path = Path(item)
        if path.is_file() and is_prediction_file(path):
            found.append(path)
        elif path.is_dir():
            for child in path.rglob("*"):
                if child.is_file() and is_prediction_file(child):
                    found.append(child)

    return sorted(set(found), key=lambda p: str(p).lower())


def is_prediction_file(path: Path) -> bool:
    name = path.name.lower()
    if name in GENERATED_NAMES:
        return False
    if path.suffix.lower() not in {".csv", ".xlsx", ".xls"}:
        return False
    return name == "predictions.csv" or "prediction" in name


def load_prediction_files(files: list[Path]) -> pd.DataFrame:
    frames = []
    for order, path in enumerate(files, start=1):
        df = read_table(path)
        df["source_order"] = order
        df["source_path"] = str(path)
        df["source_file"] = path.name
        df["source_stage"] = infer_source_stage(path)
        frames.append(df)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False)


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)

    last_exc: Exception | None = None
    for encoding in ["utf-8-sig", "utf-8", "cp949"]:
        try:
            return pd.read_csv(path, encoding=encoding)
        except Exception as exc:
            last_exc = exc
    raise RuntimeError(f"Failed to read {path}") from last_exc


def infer_source_stage(path: Path) -> str:
    lower = "/".join(part.lower() for part in path.parts)
    name = path.name.lower()
    if "train" in name and "baseline" in name:
        return "train_baseline"
    if "validation" in name and "baseline" in name:
        return "validation_baseline"
    if name == "predictions.csv":
        return f"candidate:{path.parent.name}"
    if "daily" in lower:
        return "daily_prediction"
    return path.stem


def normalize_predictions(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    ensure_columns(work, ["id", "label", "pred", "correct", "confidence", "major", "middle", "title", "reason"])

    work["label_norm"] = work["label"].map(normalize_label)
    work["pred_norm"] = work["pred"].map(normalize_label)
    work["confidence_num"] = work["confidence"].map(parse_number)
    work["exclude_from_metrics_norm"] = series_or_default(work, "exclude_from_metrics", False)
    work["exclude_from_metrics_norm"] = work["exclude_from_metrics_norm"].map(normalize_bool)
    work["evidence_obj"] = series_or_default(work, "evidence", "").map(parse_structured)
    work["decisive_evidence_obj"] = series_or_default(work, "decisive_evidence", "").map(parse_structured)
    work["visual_evidence_text"] = work["evidence_obj"].map(lambda value: compact_text(extract_evidence_field(value, "visual_evidence"), 1200))
    work["key_evidence_text"] = work["evidence_obj"].map(lambda value: compact_text(extract_evidence_field(value, "key_evidence"), 1200))
    work["missing_evidence_text"] = work["evidence_obj"].map(lambda value: compact_text(extract_evidence_field(value, "missing_evidence"), 800))
    work["decisive_evidence_text"] = work["decisive_evidence_obj"].map(lambda value: compact_text(value, 1200))
    work["has_visual_evidence"] = work["visual_evidence_text"].map(lambda text: bool(str(text).strip()))
    work["analysis_text"] = work.apply(row_analysis_text, axis=1)
    work["keyword_bucket"] = work["analysis_text"].map(assign_keyword_bucket)
    work["keyword_bucket_label"] = work["keyword_bucket"].map(keyword_bucket_label)
    work["top_terms_row"] = work["analysis_text"].map(lambda text: ", ".join(top_terms([text], 8)))

    valid_label = work["label_norm"].isin([JIN, GA])
    valid_pred = work["pred_norm"].isin([JIN, GA])
    work["excluded_norm"] = work["exclude_from_metrics_norm"] | ~valid_label | ~valid_pred | (work["pred_norm"] == BORYU)
    work["correct_norm"] = valid_label & valid_pred & (work["label_norm"] == work["pred_norm"])
    work["error_type"] = work.apply(classify_error_type, axis=1)
    return work


def ensure_columns(df: pd.DataFrame, columns: Iterable[str]) -> None:
    for col in columns:
        if col not in df.columns:
            df[col] = ""


def series_or_default(df: pd.DataFrame, column: str, default: Any) -> pd.Series:
    if column in df.columns:
        return df[column]
    return pd.Series([default] * len(df), index=df.index)


def normalize_label(value: Any) -> str:
    text = safe_str(value).strip()
    if not text:
        return ""
    if BORYU in text or "\ubcf4" in text or "\u8e42" in text:
        return BORYU
    if JIN in text or "\uc9c4" in text or "\uf9de" in text:
        return JIN
    if GA in text or "\uac00" in text or "\u5a9b" in text:
        return GA
    return text


def parse_number(value: Any) -> int:
    text = safe_str(value)
    match = re.search(r"-?\d+", text)
    if not match:
        return 0
    return int(match.group(0))


def normalize_bool(value: Any) -> bool:
    if value is None:
        return False
    try:
        if bool(pd.isna(value)):
            return False
    except Exception:
        pass
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = safe_str(value).strip().lower()
    if text in {"true", "1", "yes", "y", "t"}:
        return True
    if text in {"false", "0", "no", "n", "f", ""}:
        return False
    return False


def parse_structured(value: Any) -> Any:
    if value is None:
        return {}
    try:
        if bool(pd.isna(value)):
            return {}
    except Exception:
        pass
    if isinstance(value, (dict, list)):
        return value
    text = safe_str(value).strip()
    if not text:
        return {}
    try:
        return json.loads(text)
    except Exception:
        return text


def extract_evidence_field(value: Any, key: str) -> Any:
    if isinstance(value, dict):
        return value.get(key, [])
    return []


def row_analysis_text(row: pd.Series) -> str:
    parts = [
        row.get("title", ""),
        row.get("major", ""),
        row.get("middle", ""),
        row.get("reason", ""),
        row.get("applied_step", ""),
        row.get("decisive_evidence_text", ""),
        row.get("key_evidence_text", ""),
        row.get("visual_evidence_text", ""),
        row.get("missing_evidence_text", ""),
        row.get("error", ""),
    ]
    return " ".join(safe_str(part) for part in parts if safe_str(part).strip())


def classify_error_type(row: pd.Series) -> str:
    if bool(row.get("excluded_norm", False)):
        return "EXCLUDED"
    label = row.get("label_norm", "")
    pred = row.get("pred_norm", "")
    if label == JIN and pred == GA:
        return "FN_true_as_false"
    if label == GA and pred == JIN:
        return "FP_false_as_true"
    if label == pred and label in {JIN, GA}:
        return "CORRECT"
    return "OTHER"


def assign_keyword_bucket(text: str) -> str:
    upper_text = safe_str(text).upper()
    for name, keywords in KEYWORD_GROUPS:
        for keyword in keywords:
            if keyword.upper() in upper_text:
                return name
    return "unknown"


def keyword_bucket_label(name: str) -> str:
    labels = {
        "pipe_step": "pipe/support stepping",
        "physical_interference": "space/path/equipment interference",
        "contractor_complete": "contractor self-correction",
        "ppe": "PPE",
        "precheck": "pre-work check",
        "admin": "admin/procedure",
        "height_fall": "height/fall",
        "chemical_process": "chemical/process safety",
        "electricity": "electricity",
        "fire_explosion": "fire/explosion",
        "static_defect": "static defect",
        "process_parameter": "process parameter",
        "housekeeping": "housekeeping/path clearing",
        "weather": "weather",
        "unknown": "unknown",
    }
    return labels.get(name, name)


def build_error_cluster_outputs(
    eligible: pd.DataFrame,
    errors: pd.DataFrame,
    max_clusters: int,
    samples_per_cluster: int,
) -> tuple[pd.DataFrame, list[dict]]:
    if errors.empty:
        return pd.DataFrame(), []

    group_cols = ["error_type", "keyword_bucket", "major"]
    clusters = []
    payloads = []
    grouped = errors.groupby(group_cols, dropna=False)
    for key, group in sorted(grouped, key=lambda item: len(item[1]), reverse=True)[:max_clusters]:
        error_type, bucket, major = key
        texts = list(group["analysis_text"].map(safe_str))
        terms = top_terms(texts, limit=12)
        representatives = select_representative_rows(group, samples_per_cluster)
        correct_same_label = select_reference_rows(eligible, group, same_label=True, limit=samples_per_cluster)
        regression_guards = select_reference_rows(eligible, group, same_label=False, limit=samples_per_cluster)
        payload = {
            "error_type": safe_str(error_type),
            "keyword_bucket": safe_str(bucket),
            "keyword_bucket_label": keyword_bucket_label(safe_str(bucket)),
            "major": safe_str(major),
            "count": int(len(group)),
            "unique_ids": int(group["id"].astype(str).nunique()) if "id" in group else int(len(group)),
            "avg_confidence": round(float(group["confidence_num"].mean()), 2) if len(group) else 0,
            "visual_evidence_rows": int(group["has_visual_evidence"].sum()) if "has_visual_evidence" in group else 0,
            "top_terms": terms,
            "representative_errors": [compact_sample(row) for row in representatives.to_dict("records")],
            "correct_same_label_samples": [compact_sample(row) for row in correct_same_label.to_dict("records")],
            "regression_guard_samples": [compact_sample(row) for row in regression_guards.to_dict("records")],
        }
        payloads.append(payload)
        clusters.append(
            {
                "error_type": payload["error_type"],
                "keyword_bucket": payload["keyword_bucket"],
                "keyword_bucket_label": payload["keyword_bucket_label"],
                "major": payload["major"],
                "count": payload["count"],
                "unique_ids": payload["unique_ids"],
                "avg_confidence": payload["avg_confidence"],
                "visual_evidence_rows": payload["visual_evidence_rows"],
                "top_terms": ", ".join(terms),
                "sample_ids": ", ".join(safe_str(row.get("id", "")) for row in representatives.to_dict("records")),
            }
        )
    return pd.DataFrame(clusters), payloads


def select_representative_rows(group: pd.DataFrame, limit: int) -> pd.DataFrame:
    if group.empty:
        return group
    work = group.copy()
    work["_visual_rank"] = work["has_visual_evidence"].map(lambda value: 1 if value else 0)
    work = work.sort_values(["confidence_num", "_visual_rank", "source_order"], ascending=[False, False, False])
    selected = []
    seen_ids = set()
    for _, row in work.iterrows():
        row_id = safe_str(row.get("id", ""))
        if row_id and row_id in seen_ids:
            continue
        seen_ids.add(row_id)
        selected.append(row)
        if len(selected) >= limit:
            break
    if not selected:
        return work.head(limit).drop(columns=["_visual_rank"], errors="ignore")
    return pd.DataFrame(selected).drop(columns=["_visual_rank"], errors="ignore")


def select_reference_rows(eligible: pd.DataFrame, error_group: pd.DataFrame, same_label: bool, limit: int) -> pd.DataFrame:
    if eligible.empty or error_group.empty:
        return pd.DataFrame()
    first = error_group.iloc[0]
    target_label = first.get("label_norm")
    if not same_label:
        target_label = GA if target_label == JIN else JIN

    refs = eligible[
        (eligible["correct_norm"] == True)
        & (eligible["label_norm"] == target_label)
        & (eligible["keyword_bucket"] == first.get("keyword_bucket"))
    ].copy()
    if refs.empty:
        refs = eligible[
            (eligible["correct_norm"] == True)
            & (eligible["label_norm"] == target_label)
            & (eligible["major"].astype(str) == safe_str(first.get("major", "")))
        ].copy()
    if refs.empty:
        refs = eligible[(eligible["correct_norm"] == True) & (eligible["label_norm"] == target_label)].copy()
    return select_representative_rows(refs, limit) if not refs.empty else pd.DataFrame()


def build_representative_errors(cluster_payloads: list[dict]) -> pd.DataFrame:
    rows = []
    for idx, cluster in enumerate(cluster_payloads, start=1):
        for sample in cluster["representative_errors"]:
            row = dict(sample)
            row["cluster_rank"] = idx
            row["cluster_error_type"] = cluster["error_type"]
            row["cluster_keyword_bucket"] = cluster["keyword_bucket"]
            row["cluster_major"] = cluster["major"]
            row["cluster_top_terms"] = ", ".join(cluster["top_terms"])
            rows.append(row)
    return pd.DataFrame(rows)


def build_recurring_errors(errors: pd.DataFrame) -> pd.DataFrame:
    if errors.empty or "id" not in errors.columns:
        return pd.DataFrame()
    rows = []
    for row_id, group in errors.groupby(errors["id"].astype(str), dropna=False):
        rows.append(
            {
                "id": row_id,
                "error_count": int(len(group)),
                "source_count": int(group["source_path"].astype(str).nunique()),
                "labels": ", ".join(sorted(set(group["label_norm"].map(safe_str)))),
                "preds": ", ".join(sorted(set(group["pred_norm"].map(safe_str)))),
                "error_types": ", ".join(sorted(set(group["error_type"].map(safe_str)))),
                "majors": ", ".join(sorted(set(group["major"].map(safe_str)))),
                "keyword_buckets": ", ".join(sorted(set(group["keyword_bucket"].map(safe_str)))),
                "avg_confidence": round(float(group["confidence_num"].mean()), 2),
                "max_confidence": int(group["confidence_num"].max()),
                "title": compact_text(group.iloc[0].get("title", ""), 200),
                "last_reason": compact_text(group.sort_values("source_order").iloc[-1].get("reason", ""), 500),
            }
        )
    return pd.DataFrame(rows).sort_values(["error_count", "max_confidence"], ascending=[False, False])


def compute_overall_metrics(normalized: pd.DataFrame, eligible: pd.DataFrame, errors: pd.DataFrame, files: list[Path]) -> dict:
    labels = eligible["label_norm"] if not eligible.empty else pd.Series(dtype=str)
    preds = eligible["pred_norm"] if not eligible.empty else pd.Series(dtype=str)
    tp = int(((labels == JIN) & (preds == JIN)).sum())
    fn = int(((labels == JIN) & (preds == GA)).sum())
    fp = int(((labels == GA) & (preds == JIN)).sum())
    tn = int(((labels == GA) & (preds == GA)).sum())
    eligible_n = int(len(eligible))
    accuracy = float((eligible["correct_norm"] == True).mean()) if eligible_n else 0.0

    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "file_count": len(files),
        "files": [str(path) for path in files],
        "total_rows": int(len(normalized)),
        "eligible_rows": eligible_n,
        "excluded_rows": int(len(normalized) - eligible_n),
        "unique_ids": int(normalized["id"].astype(str).nunique()) if "id" in normalized else int(len(normalized)),
        "error_rows": int(len(errors)),
        "accuracy": round(accuracy, 6),
        "tp_true": tp,
        "fn_true_as_false": fn,
        "fp_false_as_true": fp,
        "tn_false": tn,
        "fn_rate_among_true": round(safe_div(fn, tp + fn), 6),
        "fp_rate_among_false": round(safe_div(fp, tn + fp), 6),
    }


def load_policy_text(policy_path: str | Path | None, max_policy_chars: int = 30000) -> str:
    if not policy_path:
        return ""
    path = Path(policy_path)
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8")
    if max_policy_chars > 0 and len(text) > max_policy_chars:
        return text[:max_policy_chars].rstrip() + "\n\n...[policy truncated for analysis prompt]"
    return text


def build_llm_brief(metrics: dict, cluster_payloads: list[dict], policy_text: str = "", policy_path: str | Path | None = None) -> dict:
    return {
        "purpose": "Summarized prediction error clusters. Use this instead of loading every raw prediction row.",
        "current_policy_path": str(policy_path or ""),
        "current_policy": policy_text,
        "metrics": metrics,
        "clusters": cluster_payloads,
    }


def build_llm_prompt(llm_brief: dict) -> str:
    brief = json.dumps(llm_brief, ensure_ascii=False, indent=2)
    return f"""You are reviewing a stop-work-right 진성/가성 classifier.

The raw prediction files were already aggregated by Python. Do not ask for all raw rows.
Analyze the summarized clusters below and produce a concise diagnosis.

Focus on:
1. repeated FN patterns where true 진성 was predicted 가성,
2. repeated FP patterns where true 가성 was predicted 진성,
3. which exact parts of the current policy caused, failed to prevent, or contradicted the repeated errors,
4. clusters that should NOT be fixed by prompt rules because labels look inconsistent or evidence is insufficient,
5. the smallest safe policy changes to test next.

Return JSON with:
{{
  "root_causes": [],
  "current_policy_problem_points": [],
  "high_value_policy_changes": [],
  "risky_policy_changes_to_avoid": [],
  "label_quality_issues": [],
  "next_validation_plan": []
}}

[Aggregated Prediction Error Brief]
{brief}
"""


def build_markdown_report(metrics: dict, clusters_df: pd.DataFrame, recurring_df: pd.DataFrame, paths: dict[str, Path]) -> str:
    lines = [
        "# Prediction Error Analysis",
        "",
        f"- Generated at: {metrics['generated_at']}",
        f"- Files scanned: {metrics['file_count']}",
        f"- Total rows: {metrics['total_rows']}",
        f"- Eligible rows: {metrics['eligible_rows']}",
        f"- Excluded rows: {metrics['excluded_rows']}",
        f"- Unique IDs: {metrics['unique_ids']}",
        f"- Error rows: {metrics['error_rows']}",
        f"- Accuracy: {metrics['accuracy']:.4f}",
        f"- FN true-as-false: {metrics['fn_true_as_false']} ({metrics['fn_rate_among_true']:.4f})",
        f"- FP false-as-true: {metrics['fp_false_as_true']} ({metrics['fp_rate_among_false']:.4f})",
        "",
        "## Output Files",
    ]
    for name, path in paths.items():
        lines.append(f"- {name}: `{path}`")

    lines.extend(["", "## Top Error Clusters", ""])
    lines.extend(markdown_table(clusters_df.head(20), ["error_type", "keyword_bucket_label", "major", "count", "avg_confidence", "visual_evidence_rows", "top_terms"]))

    lines.extend(["", "## Recurring Error IDs", ""])
    lines.extend(markdown_table(recurring_df.head(20), ["id", "error_count", "source_count", "error_types", "majors", "keyword_buckets", "max_confidence", "title"]))

    lines.extend(
        [
            "",
            "## How To Use",
            "",
            "1. Open `error_clusters.csv` to find the largest repeated failure groups.",
            "2. Open `representative_errors.csv` to inspect a few high-confidence wrong examples per cluster.",
            "3. Paste `llm_analysis_prompt.md` into the internal LLM when you want a policy-change diagnosis without sending every raw row.",
        ]
    )
    return "\n".join(lines) + "\n"


def markdown_table(df: pd.DataFrame, columns: list[str]) -> list[str]:
    if df.empty:
        return ["No rows."]
    cols = [col for col in columns if col in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for _, row in df[cols].iterrows():
        values = [compact_text(row.get(col, ""), 120).replace("|", "/") for col in cols]
        lines.append("| " + " | ".join(values) + " |")
    return lines


def compact_sample(row: dict) -> dict:
    return {
        "source_stage": safe_str(row.get("source_stage", "")),
        "source_file": safe_str(row.get("source_file", "")),
        "id": safe_str(row.get("id", "")),
        "title": compact_text(row.get("title", ""), 200),
        "major": safe_str(row.get("major", "")),
        "middle": safe_str(row.get("middle", "")),
        "label": safe_str(row.get("label_norm", row.get("label", ""))),
        "pred": safe_str(row.get("pred_norm", row.get("pred", ""))),
        "confidence": int(row.get("confidence_num", 0) or 0),
        "reason": compact_text(row.get("reason", ""), 700),
        "applied_step": safe_str(row.get("applied_step", "")),
        "decisive_evidence": compact_text(row.get("decisive_evidence_text", ""), 700),
        "key_evidence": compact_text(row.get("key_evidence_text", ""), 700),
        "visual_evidence": compact_text(row.get("visual_evidence_text", ""), 700),
        "top_terms": safe_str(row.get("top_terms_row", "")),
    }


def top_terms(texts: list[str], limit: int = 10) -> list[str]:
    counter: Counter[str] = Counter()
    joined = " ".join(safe_str(text) for text in texts)
    upper_joined = joined.upper()
    for _, keywords in KEYWORD_GROUPS:
        for keyword in keywords:
            if keyword.upper() in upper_joined:
                counter[keyword] += upper_joined.count(keyword.upper())

    for token in re.findall(r"[A-Za-z0-9_]{2,}|[\uac00-\ud7a3]{2,}", joined):
        lowered = token.lower()
        if lowered in STOPWORDS:
            continue
        if len(token) > 30:
            continue
        counter[token] += 1

    return [term for term, _ in counter.most_common(limit)]


def compact_text(value: Any, limit: int = 500) -> str:
    text = structured_to_text(value)
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > limit:
        return text[:limit].rstrip() + "..."
    return text


def structured_to_text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if isinstance(value, float) and math.isnan(value):
            return ""
    except Exception:
        pass
    if isinstance(value, dict):
        return " ".join(f"{key}: {structured_to_text(val)}" for key, val in value.items())
    if isinstance(value, list):
        return " ".join(structured_to_text(item) for item in value)
    return safe_str(value)


def safe_str(value: Any) -> str:
    if value is None:
        return ""
    try:
        if isinstance(value, float) and math.isnan(value):
            return ""
    except Exception:
        pass
    return str(value)


def safe_div(num: int | float, den: int | float) -> float:
    if den == 0:
        return 0.0
    return float(num / den)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Analyze accumulated prediction CSV/XLSX files.")
    parser.add_argument("inputs", nargs="*", help="Prediction files or folders. Defaults to outputs.")
    parser.add_argument("--output-dir", default="artifacts/prediction_analysis", help="Analysis output root folder.")
    parser.add_argument("--max-clusters", type=int, default=20)
    parser.add_argument("--samples-per-cluster", type=int, default=5)
    parser.add_argument("--policy-path", default="policies/current_policy.md", help="Current policy/prompt file to include in the LLM analysis prompt.")
    parser.add_argument("--max-policy-chars", type=int, default=30000, help="Maximum policy characters included in the LLM brief.")
    args = parser.parse_args(argv)

    inputs = args.inputs or ["outputs"]
    paths = analyze_prediction_outputs(
        inputs=inputs,
        output_root=args.output_dir,
        max_clusters=args.max_clusters,
        samples_per_cluster=args.samples_per_cluster,
        policy_path=args.policy_path,
        max_policy_chars=args.max_policy_chars,
    )
    print("[prediction-analysis] done")
    for name, path in paths.items():
        print(f"[prediction-analysis] {name}: {path}")


if __name__ == "__main__":
    main()
