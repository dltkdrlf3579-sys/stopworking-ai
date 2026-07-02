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
    "pipe_support_correct_false.csv",
    "pipe_support_correct_true.csv",
    "pipe_support_contrast_clusters.csv",
    "pipe_support_errors.csv",
    "pipe_support_focus.csv",
    "pipe_support_fp.csv",
    "pipe_support_fn.csv",
    "pipe_support_representatives.csv",
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

PIPE_SUPPORT_CORE_KEYWORDS = [
    "\ubc30\uad00",
    "\uc11c\ud3ec\ud2b8",
    "\uc815\uc81c\uc9c8\uc18c",
    "\ucf00\ubbf8\uceec",
    "\ub355\ud2b8",
    "\ud2b8\ub808\uc774",
    "\uc124\ube44 \ud504\ub808\uc784",
    "\ubc1f",
    "\ubc1f\uc74c",
]

PIPE_SUPPORT_CONTEXT_KEYWORDS = [
    "\ubc1c\ud310",
    "\uc0ac\ub2e4\ub9ac",
    "\ube44\uacc4",
    "\ub09c\uac04",
    "\uac1c\uad6c\ubd80",
    "\uace0\uc18c",
    "\ucd94\ub77d",
    "\ub099\ud558",
    "\uacf5\uac04",
    "\ud611\uc18c",
    "\uac04\uc12d",
    "\ub3d9\uc120",
    "\ud1b5\ub85c",
    "\uc811\ucd09",
    "\ud30c\uc190",
    "\ub204\ucd9c",
    "\ud611\uc758",
    "\uc124\uce58",
]

PIPE_SUPPORT_TRUE_CLUES = [
    "\ubc1f\uc544\uc57c",
    "\ubc1f\uc74c \ud544\uc694",
    "\ubc1f\uace0",
    "\ubc1f\uc73c\uba70",
    "\ubc1c\ud310 \ubd80\uc871",
    "\uc774\ub3d9\uacbd\ub85c \ubd80\uc871",
    "\uacf5\uac04 \ud611\uc18c",
    "\ud611\uc18c",
    "\uac04\uc12d",
    "\ubd80\uc11c \ud611\uc758",
    "\uc2dc\uacf5\uadf8\ub8f9",
    "\uc791\uc5c5\ubc29\ubc95 \ubcc0\uacbd",
    "\uc784\uc2dc\ubc1c\ud310",
    "\ube44\uacc4",
    "\ub09c\uac04",
    "\ub36e\uac1c",
    "\uc124\ube44 \ubcf4\ud638",
    "\ucd94\ub77d",
    "\ub099\ud558",
    "\ub204\ucd9c",
    "\ud30c\uc190",
    "\uc811\ucd09",
]

PIPE_SUPPORT_FALSE_CLUES = [
    "\uc791\uc5c5\uc608\uc815",
    "\uc791\uc5c5 \uc608\uc815",
    "\ud604\uc7a5\ud655\uc778 \uc2dc",
    "\uc0ac\uc804",
    "\uc0ac\uc804\uc810\uac80",
    "\uc791\uc5c5\uc804",
    "DRI",
    "\uacc4\ud68d",
    "\uc77c\uc815 \uc870\uc728",
    "\ud5c8\uac00",
    "\uc11c\ub958",
    "\uad50\uc721",
    "\uc7ac\uccb4\uacb0",
    "\uad50\uccb4",
    "\uc815\ub9ac\uc815\ub3c8",
    "\ud1b5\ub85c\ud655\ubcf4",
    "\uc790\uc7ac \uc774\ub3d9",
    "\uc704\uce58 \ubcc0\uacbd",
    "\ub2e8\uc21c",
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
    include_candidates: bool = False,
) -> dict[str, Path]:
    files = discover_prediction_files(inputs, include_candidates=include_candidates)
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
    fn_clusters_df = filter_clusters(clusters_df, "FN_true_as_false")
    fp_clusters_df = filter_clusters(clusters_df, "FP_false_as_true")
    recurring_df = build_recurring_errors(errors)
    representatives_df = build_representative_errors(cluster_payloads)
    metrics = compute_overall_metrics(normalized, eligible, errors, files)
    metrics["include_candidates"] = bool(include_candidates)
    policy_text = load_policy_text(policy_path, max_policy_chars=max_policy_chars)
    llm_brief = build_llm_brief(metrics, cluster_payloads, policy_text, policy_path)
    pipe_support_bundle = build_pipe_support_focus_bundle(
        eligible,
        samples_per_group=samples_per_cluster,
        policy_text=policy_text,
        policy_path=policy_path,
    )

    paths = {
        "combined_predictions": out_dir / "combined_predictions.csv",
        "error_cases": out_dir / "error_cases.csv",
        "error_clusters": out_dir / "error_clusters.csv",
        "fn_clusters": out_dir / "fn_clusters.csv",
        "fp_clusters": out_dir / "fp_clusters.csv",
        "representative_errors": out_dir / "representative_errors.csv",
        "recurring_errors": out_dir / "recurring_errors.csv",
        "llm_cluster_brief": out_dir / "llm_cluster_brief.json",
        "llm_analysis_prompt": out_dir / "llm_analysis_prompt.md",
        "policy_snapshot": out_dir / "policy_snapshot.md",
        "pipe_support_focus": out_dir / "pipe_support_focus.csv",
        "pipe_support_errors": out_dir / "pipe_support_errors.csv",
        "pipe_support_fn": out_dir / "pipe_support_fn.csv",
        "pipe_support_fp": out_dir / "pipe_support_fp.csv",
        "pipe_support_correct_true": out_dir / "pipe_support_correct_true.csv",
        "pipe_support_correct_false": out_dir / "pipe_support_correct_false.csv",
        "pipe_support_contrast_clusters": out_dir / "pipe_support_contrast_clusters.csv",
        "pipe_support_representatives": out_dir / "pipe_support_representatives.csv",
        "pipe_support_llm_prompt": out_dir / "pipe_support_llm_prompt.md",
        "pipe_support_report": out_dir / "pipe_support_report.md",
        "report": out_dir / "error_analysis_report.md",
    }

    normalized.to_csv(paths["combined_predictions"], index=False, encoding="utf-8-sig")
    errors.to_csv(paths["error_cases"], index=False, encoding="utf-8-sig")
    clusters_df.to_csv(paths["error_clusters"], index=False, encoding="utf-8-sig")
    fn_clusters_df.to_csv(paths["fn_clusters"], index=False, encoding="utf-8-sig")
    fp_clusters_df.to_csv(paths["fp_clusters"], index=False, encoding="utf-8-sig")
    representatives_df.to_csv(paths["representative_errors"], index=False, encoding="utf-8-sig")
    recurring_df.to_csv(paths["recurring_errors"], index=False, encoding="utf-8-sig")
    paths["llm_cluster_brief"].write_text(json.dumps(llm_brief, ensure_ascii=False, indent=2), encoding="utf-8")
    paths["llm_analysis_prompt"].write_text(build_llm_prompt(llm_brief), encoding="utf-8")
    paths["policy_snapshot"].write_text(policy_text or "Policy file not found or empty.", encoding="utf-8")
    pipe_support_bundle["focus"].to_csv(paths["pipe_support_focus"], index=False, encoding="utf-8-sig")
    pipe_support_bundle["errors"].to_csv(paths["pipe_support_errors"], index=False, encoding="utf-8-sig")
    pipe_support_bundle["fn"].to_csv(paths["pipe_support_fn"], index=False, encoding="utf-8-sig")
    pipe_support_bundle["fp"].to_csv(paths["pipe_support_fp"], index=False, encoding="utf-8-sig")
    pipe_support_bundle["correct_true"].to_csv(paths["pipe_support_correct_true"], index=False, encoding="utf-8-sig")
    pipe_support_bundle["correct_false"].to_csv(paths["pipe_support_correct_false"], index=False, encoding="utf-8-sig")
    pipe_support_bundle["clusters"].to_csv(paths["pipe_support_contrast_clusters"], index=False, encoding="utf-8-sig")
    pipe_support_bundle["representatives"].to_csv(paths["pipe_support_representatives"], index=False, encoding="utf-8-sig")
    paths["pipe_support_llm_prompt"].write_text(pipe_support_bundle["llm_prompt"], encoding="utf-8")
    paths["pipe_support_report"].write_text(build_pipe_support_report(pipe_support_bundle, paths), encoding="utf-8")
    paths["report"].write_text(build_markdown_report(metrics, clusters_df, recurring_df, paths), encoding="utf-8")

    return paths


def discover_prediction_files(inputs: list[str | Path], include_candidates: bool = False) -> list[Path]:
    found: list[Path] = []
    for item in inputs:
        path = Path(item)
        if path.is_file() and is_prediction_file(path, include_candidates=include_candidates):
            found.append(path)
        elif path.is_dir():
            for child in path.rglob("*"):
                if child.is_file() and is_prediction_file(child, include_candidates=include_candidates):
                    found.append(child)

    return sorted(set(found), key=lambda p: str(p).lower())


def is_prediction_file(path: Path, include_candidates: bool = False) -> bool:
    name = path.name.lower()
    if name in GENERATED_NAMES:
        return False
    if path.suffix.lower() not in {".csv", ".xlsx", ".xls"}:
        return False
    if name == "predictions.csv":
        return include_candidates
    return "prediction" in name


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
    ensure_columns(
        work,
        [
            "id",
            "label",
            "pred",
            "correct",
            "confidence",
            "major",
            "middle",
            "title",
            "reason",
            "source_order",
            "source_path",
            "source_file",
            "source_stage",
        ],
    )
    work["source_order"] = work["source_order"].map(parse_number)

    work["label_norm"] = work["label"].map(normalize_label)
    work["pred_norm"] = work["pred"].map(normalize_label)
    work["confidence_num"] = work["confidence"].map(parse_number)
    work["exclude_from_metrics_norm"] = series_or_default(work, "exclude_from_metrics", False)
    work["exclude_from_metrics_norm"] = work["exclude_from_metrics_norm"].map(normalize_bool)
    work["evidence_obj"] = series_or_default(work, "evidence", "").map(parse_structured)
    work["decisive_evidence_obj"] = series_or_default(work, "decisive_evidence", "").map(parse_structured)
    work["visual_evidence_text"] = work["evidence_obj"].map(lambda value: compact_text(extract_evidence_field(value, "visual_evidence"), 1200))
    work["pipe_support_evidence_text"] = work["evidence_obj"].map(lambda value: compact_text(extract_evidence_field(value, "pipe_support_evidence"), 1200))
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
        row.get("pipe_support_evidence_text", ""),
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


def filter_clusters(clusters_df: pd.DataFrame, error_type: str) -> pd.DataFrame:
    if clusters_df.empty or "error_type" not in clusters_df.columns:
        return pd.DataFrame()
    return clusters_df[clusters_df["error_type"] == error_type].copy()


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
    true_total = tp + fn
    false_total = tn + fp
    target_true_recall = 0.80
    true_tp_needed_for_80 = int(math.ceil(true_total * target_true_recall)) if true_total else 0
    additional_true_tp_needed_for_80 = max(0, true_tp_needed_for_80 - tp)

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
        "true_total": true_total,
        "false_total": false_total,
        "tp_true": tp,
        "fn_true_as_false": fn,
        "fp_false_as_true": fp,
        "tn_false": tn,
        "true_recall": round(safe_div(tp, true_total), 6),
        "true_precision": round(safe_div(tp, tp + fp), 6),
        "false_recall": round(safe_div(tn, false_total), 6),
        "false_precision": round(safe_div(tn, tn + fn), 6),
        "fn_rate_among_true": round(safe_div(fn, tp + fn), 6),
        "fp_rate_among_false": round(safe_div(fp, tn + fp), 6),
        "target_true_recall": target_true_recall,
        "true_tp_needed_for_80_recall": true_tp_needed_for_80,
        "additional_true_tp_needed_for_80_recall": additional_true_tp_needed_for_80,
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


def build_pipe_support_focus_bundle(
    eligible: pd.DataFrame,
    samples_per_group: int,
    policy_text: str = "",
    policy_path: str | Path | None = None,
) -> dict[str, Any]:
    if eligible.empty:
        focus = eligible.copy()
    else:
        focus = eligible[eligible.apply(is_pipe_support_focus_row, axis=1)].copy()

    if not focus.empty:
        focus["pipe_support_subtype"] = focus["analysis_text"].map(assign_pipe_support_subtype)
        focus["pipe_support_true_clues"] = focus["analysis_text"].map(lambda text: ", ".join(matched_keywords(text, PIPE_SUPPORT_TRUE_CLUES, 12)))
        focus["pipe_support_false_clues"] = focus["analysis_text"].map(lambda text: ", ".join(matched_keywords(text, PIPE_SUPPORT_FALSE_CLUES, 12)))
    else:
        focus["pipe_support_subtype"] = ""
        focus["pipe_support_true_clues"] = ""
        focus["pipe_support_false_clues"] = ""

    errors = focus[focus["correct_norm"] == False].copy() if "correct_norm" in focus else pd.DataFrame()
    fn = focus[focus["error_type"] == "FN_true_as_false"].copy() if "error_type" in focus else pd.DataFrame()
    fp = focus[focus["error_type"] == "FP_false_as_true"].copy() if "error_type" in focus else pd.DataFrame()
    correct_true = focus[(focus["correct_norm"] == True) & (focus["label_norm"] == JIN)].copy() if "label_norm" in focus else pd.DataFrame()
    correct_false = focus[(focus["correct_norm"] == True) & (focus["label_norm"] == GA)].copy() if "label_norm" in focus else pd.DataFrame()
    clusters = build_pipe_support_contrast_clusters(focus)
    representatives = build_pipe_support_representatives(focus, samples_per_group)
    metrics = compute_pipe_support_metrics(focus)
    brief = {
        "purpose": "Focused pipe/support stepping and path-interference error brief. Use this to improve true recall without broad FP increase.",
        "current_policy_path": str(policy_path or ""),
        "metrics": metrics,
        "clusters": clusters.to_dict("records") if not clusters.empty else [],
        "representative_samples": representatives.to_dict("records") if not representatives.empty else [],
        "current_policy": policy_text,
    }

    return {
        "metrics": metrics,
        "focus": select_pipe_support_columns(focus),
        "errors": select_pipe_support_columns(errors),
        "fn": select_pipe_support_columns(fn),
        "fp": select_pipe_support_columns(fp),
        "correct_true": select_pipe_support_columns(correct_true),
        "correct_false": select_pipe_support_columns(correct_false),
        "clusters": clusters,
        "representatives": representatives,
        "llm_prompt": build_pipe_support_llm_prompt(brief),
    }


def is_pipe_support_focus_row(row: pd.Series) -> bool:
    if safe_str(row.get("keyword_bucket", "")) == "pipe_step":
        return True
    text = safe_str(row.get("analysis_text", ""))
    if not text.strip():
        return False
    core_hits = matched_keywords(text, PIPE_SUPPORT_CORE_KEYWORDS, 4)
    context_hits = matched_keywords(text, PIPE_SUPPORT_CONTEXT_KEYWORDS, 4)
    if core_hits and context_hits:
        return True

    height_access = matched_keywords(text, ["발판", "사다리", "비계", "난간", "개구부", "고소", "추락"], 3)
    path_problem = matched_keywords(text, ["공간", "협소", "간섭", "동선", "통로", "밟"], 3)
    return bool(height_access and path_problem)


def assign_pipe_support_subtype(text: str) -> str:
    if matched_keywords(text, ["밟", "밟음", "밟아야", "밟고"], 1) and matched_keywords(text, ["배관", "서포트", "정제질소", "케미컬", "덕트", "트레이"], 1):
        return "forced_pipe_or_support_stepping"
    if matched_keywords(text, ["누출", "접액", "냄새", "응축수", "DIW", "화학", "가스", "접촉"], 1):
        return "pipe_leak_or_contact_uncertainty"
    if matched_keywords(text, ["발판", "사다리", "비계", "난간", "개구부", "고소", "추락", "낙하"], 1):
        return "height_or_access_fall"
    if matched_keywords(text, ["공간", "협소", "간섭", "동선", "통로", "충돌", "끼임"], 1):
        return "space_path_interference"
    return "pipe_support_general"


def matched_keywords(text: str, keywords: list[str], limit: int = 20) -> list[str]:
    upper_text = safe_str(text).upper()
    matches = []
    for keyword in keywords:
        if keyword.upper() in upper_text and keyword not in matches:
            matches.append(keyword)
        if len(matches) >= limit:
            break
    return matches


def build_pipe_support_contrast_clusters(focus: pd.DataFrame) -> pd.DataFrame:
    if focus.empty:
        return pd.DataFrame()

    rows = []
    group_cols = ["error_type", "pipe_support_subtype", "major"]
    grouped = focus.groupby(group_cols, dropna=False)
    for key, group in sorted(grouped, key=lambda item: (error_sort_rank(item[0][0]), -len(item[1]))):
        error_type, subtype, major = key
        texts = list(group["analysis_text"].map(safe_str))
        representatives = select_representative_rows(group, 4)
        rows.append(
            {
                "error_type": safe_str(error_type),
                "pipe_support_subtype": safe_str(subtype),
                "major": safe_str(major),
                "count": int(len(group)),
                "unique_ids": int(group["id"].astype(str).nunique()) if "id" in group else int(len(group)),
                "label_counts": value_counts_text(group.get("label_norm", pd.Series(dtype=str))),
                "pred_counts": value_counts_text(group.get("pred_norm", pd.Series(dtype=str))),
                "avg_confidence": round(float(group["confidence_num"].mean()), 2) if len(group) else 0,
                "visual_evidence_rows": int(group["has_visual_evidence"].sum()) if "has_visual_evidence" in group else 0,
                "true_clues": value_counts_from_csv(group.get("pipe_support_true_clues", pd.Series(dtype=str)), 10),
                "false_clues": value_counts_from_csv(group.get("pipe_support_false_clues", pd.Series(dtype=str)), 10),
                "top_terms": ", ".join(top_terms(texts, limit=12)),
                "sample_ids": ", ".join(safe_str(row.get("id", "")) for row in representatives.to_dict("records")),
            }
        )
    return pd.DataFrame(rows)


def build_pipe_support_representatives(focus: pd.DataFrame, samples_per_group: int) -> pd.DataFrame:
    if focus.empty:
        return pd.DataFrame()

    rows = []
    group_cols = ["error_type", "pipe_support_subtype", "major"]
    grouped = focus.groupby(group_cols, dropna=False)
    for key, group in sorted(grouped, key=lambda item: (error_sort_rank(item[0][0]), -len(item[1])))[:40]:
        error_type, subtype, major = key
        representatives = select_representative_rows(group, samples_per_group)
        for sample in representatives.to_dict("records"):
            row = compact_pipe_support_sample(sample)
            row["cluster_error_type"] = safe_str(error_type)
            row["cluster_pipe_support_subtype"] = safe_str(subtype)
            row["cluster_major"] = safe_str(major)
            row["cluster_count"] = int(len(group))
            rows.append(row)
    return pd.DataFrame(rows)


def compact_pipe_support_sample(row: dict) -> dict:
    sample = compact_sample(row)
    sample["pipe_support_subtype"] = safe_str(row.get("pipe_support_subtype", ""))
    sample["pipe_support_true_clues"] = safe_str(row.get("pipe_support_true_clues", ""))
    sample["pipe_support_false_clues"] = safe_str(row.get("pipe_support_false_clues", ""))
    sample["pipe_support_evidence"] = compact_text(row.get("pipe_support_evidence_text", ""), 800)
    return sample


def select_pipe_support_columns(df: pd.DataFrame) -> pd.DataFrame:
    preferred = [
        "source_stage",
        "source_file",
        "id",
        "title",
        "major",
        "middle",
        "label_norm",
        "pred_norm",
        "correct_norm",
        "error_type",
        "confidence_num",
        "pipe_support_subtype",
        "pipe_support_true_clues",
        "pipe_support_false_clues",
        "reason",
        "applied_step",
        "decisive_evidence_text",
        "key_evidence_text",
        "visual_evidence_text",
        "pipe_support_evidence_text",
        "missing_evidence_text",
        "top_terms_row",
        "source_path",
    ]
    cols = [col for col in preferred if col in df.columns]
    if not cols:
        return pd.DataFrame(columns=preferred)
    return df[cols].copy()


def compute_pipe_support_metrics(focus: pd.DataFrame) -> dict:
    if focus.empty:
        return {
            "focus_rows": 0,
            "focus_errors": 0,
            "focus_accuracy": 0.0,
            "true_total": 0,
            "false_total": 0,
            "tp_true": 0,
            "fn_true_as_false": 0,
            "fp_false_as_true": 0,
            "tn_false": 0,
            "true_recall": 0.0,
            "true_precision": 0.0,
            "false_recall": 0.0,
            "false_precision": 0.0,
        }

    labels = focus["label_norm"]
    preds = focus["pred_norm"]
    tp = int(((labels == JIN) & (preds == JIN)).sum())
    fn = int(((labels == JIN) & (preds == GA)).sum())
    fp = int(((labels == GA) & (preds == JIN)).sum())
    tn = int(((labels == GA) & (preds == GA)).sum())
    true_total = tp + fn
    false_total = tn + fp
    return {
        "focus_rows": int(len(focus)),
        "focus_errors": int((focus["correct_norm"] == False).sum()),
        "focus_accuracy": round(float((focus["correct_norm"] == True).mean()), 6),
        "true_total": true_total,
        "false_total": false_total,
        "tp_true": tp,
        "fn_true_as_false": fn,
        "fp_false_as_true": fp,
        "tn_false": tn,
        "true_recall": round(safe_div(tp, true_total), 6),
        "true_precision": round(safe_div(tp, tp + fp), 6),
        "false_recall": round(safe_div(tn, false_total), 6),
        "false_precision": round(safe_div(tn, tn + fn), 6),
    }


def build_pipe_support_llm_prompt(brief: dict) -> str:
    brief_text = json.dumps(brief, ensure_ascii=False, indent=2)
    return f"""너는 작업중지권 진성/가성 분류 정책을 검토하는 안전관리 데이터 분석가다.

아래 자료는 전체 오답이 아니라 `pipe/support stepping`, 배관/서포트 밟음, 발판·이동경로 부족, 공간 협소, 설비 간섭 관련 사례만 따로 모은 것이다.
목표는 진성 recall을 올리는 것이지만, 같은 군집에서 FP도 같이 발생하므로 넓은 진성 규칙을 만들면 안 된다.

반드시 다음 관점으로 분석하라.
1. FN_true_as_false와 FP_false_as_true를 같은 subtype/대분류 안에서 비교한다.
2. "배관", "발판", "협의", "공간", "추락" 같은 단어 자체가 아니라, 실제로 판정을 가르는 discriminator를 찾는다.
3. discriminator는 작업중/작업개시직전, 안전한 발판 부재, 밟으면 안 되는 설비를 밟아야 하는 강제성, 물리적 보강 필요, 행정 협의인지 기술 협의인지, 사전점검인지 실제 노출 상태인지로 나눈다.
4. `pipe/support` 전체를 진성으로 보내는 제안은 금지한다.
5. 정책 변경이 필요하다면 1~2문장짜리 좁은 보강 규칙과, FP 방지 반례 조건을 함께 제안한다.
6. 라벨 기준 자체가 흔들리는 영역이면 정책 변경 대신 "라벨 기준 확인 필요"로 분리한다.

JSON만 출력하라.
{{
  "fn_fp_discriminators": [],
  "high_value_narrow_rule": "",
  "fp_guardrail": "",
  "image_evidence_to_check": [],
  "policy_change_risk": "",
  "do_not_change": [],
  "next_manual_review_targets": []
}}

[Pipe/Support Focus Brief]
{brief_text}
"""


def build_pipe_support_report(bundle: dict[str, Any], paths: dict[str, Path]) -> str:
    metrics = bundle["metrics"]
    clusters = bundle["clusters"]
    representatives = bundle["representatives"]
    lines = [
        "# Pipe/Support Focus Analysis",
        "",
        "This report isolates pipe/support stepping, foothold/path shortage, and equipment-interference cases.",
        "",
        "## Metrics",
        "",
        f"- Focus rows: {metrics.get('focus_rows', 0)}",
        f"- Focus errors: {metrics.get('focus_errors', 0)}",
        f"- Focus accuracy: {metrics.get('focus_accuracy', 0):.4f}",
        f"- True recall in focus: {metrics.get('true_recall', 0):.4f}",
        f"- True precision in focus: {metrics.get('true_precision', 0):.4f}",
        f"- False recall in focus: {metrics.get('false_recall', 0):.4f}",
        f"- FN true-as-false in focus: {metrics.get('fn_true_as_false', 0)}",
        f"- FP false-as-true in focus: {metrics.get('fp_false_as_true', 0)}",
        "",
        "## Output Files",
        "",
    ]
    for name in [
        "pipe_support_focus",
        "pipe_support_errors",
        "pipe_support_fn",
        "pipe_support_fp",
        "pipe_support_correct_true",
        "pipe_support_correct_false",
        "pipe_support_contrast_clusters",
        "pipe_support_representatives",
        "pipe_support_llm_prompt",
    ]:
        if name in paths:
            lines.append(f"- {name}: `{paths[name]}`")

    lines.extend(["", "## FN To Improve True Recall", ""])
    fn_clusters = clusters[clusters["error_type"] == "FN_true_as_false"] if not clusters.empty and "error_type" in clusters else pd.DataFrame()
    lines.extend(markdown_table(fn_clusters.head(20), ["pipe_support_subtype", "major", "count", "avg_confidence", "visual_evidence_rows", "true_clues", "false_clues", "top_terms"]))

    lines.extend(["", "## FP To Guard Accuracy", ""])
    fp_clusters = clusters[clusters["error_type"] == "FP_false_as_true"] if not clusters.empty and "error_type" in clusters else pd.DataFrame()
    lines.extend(markdown_table(fp_clusters.head(20), ["pipe_support_subtype", "major", "count", "avg_confidence", "visual_evidence_rows", "true_clues", "false_clues", "top_terms"]))

    lines.extend(["", "## Correct True References", ""])
    correct_true = clusters[(clusters["error_type"] == "CORRECT") & (clusters["label_counts"].str.contains(JIN, na=False))] if not clusters.empty and "label_counts" in clusters else pd.DataFrame()
    lines.extend(markdown_table(correct_true.head(10), ["pipe_support_subtype", "major", "count", "avg_confidence", "true_clues", "false_clues", "top_terms"]))

    lines.extend(["", "## Representative Samples", ""])
    lines.extend(markdown_table(representatives.head(20), ["cluster_error_type", "cluster_pipe_support_subtype", "cluster_major", "id", "label", "pred", "confidence", "pipe_support_true_clues", "pipe_support_false_clues", "title"]))

    lines.extend(
        [
            "",
            "## How To Use",
            "",
            "1. Open `pipe_support_fn.csv` and `pipe_support_fp.csv` side by side.",
            "2. Paste `pipe_support_llm_prompt.md` into the internal LLM for a focused diagnosis.",
            "3. Only accept a new policy sentence if it catches repeated FN while clearly excluding the listed FP clusters.",
        ]
    )
    return "\n".join(lines) + "\n"


def error_sort_rank(error_type: Any) -> int:
    order = {
        "FN_true_as_false": 0,
        "FP_false_as_true": 1,
        "CORRECT": 2,
        "OTHER": 3,
        "EXCLUDED": 4,
    }
    return order.get(safe_str(error_type), 9)


def value_counts_text(series: pd.Series) -> str:
    if series.empty:
        return ""
    counts = series.map(safe_str).value_counts(dropna=False)
    return ", ".join(f"{idx}:{count}" for idx, count in counts.items() if safe_str(idx))


def value_counts_from_csv(series: pd.Series, limit: int = 10) -> str:
    counter: Counter[str] = Counter()
    for value in series.map(safe_str):
        for item in [part.strip() for part in value.split(",") if part.strip()]:
            counter[item] += 1
    return ", ".join(term for term, _ in counter.most_common(limit))


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
        f"- True recall: {metrics.get('true_recall', 0):.4f}",
        f"- True precision: {metrics.get('true_precision', 0):.4f}",
        f"- False recall: {metrics.get('false_recall', 0):.4f}",
        f"- False precision: {metrics.get('false_precision', 0):.4f}",
        f"- FN true-as-false: {metrics['fn_true_as_false']} ({metrics['fn_rate_among_true']:.4f})",
        f"- FP false-as-true: {metrics['fp_false_as_true']} ({metrics['fp_rate_among_false']:.4f})",
        f"- Additional TP needed for true recall 80%: {metrics.get('additional_true_tp_needed_for_80_recall', 0)}",
        f"- Candidate predictions included: {metrics.get('include_candidates', False)}",
        "",
        "## Output Files",
    ]
    for name, path in paths.items():
        lines.append(f"- {name}: `{path}`")

    lines.extend(["", "## Top Error Clusters", ""])
    lines.extend(markdown_table(clusters_df.head(20), ["error_type", "keyword_bucket_label", "major", "count", "avg_confidence", "visual_evidence_rows", "top_terms"]))

    fn_clusters = filter_clusters(clusters_df, "FN_true_as_false")
    fp_clusters = filter_clusters(clusters_df, "FP_false_as_true")

    lines.extend(["", "## FN Clusters To Improve True Recall", ""])
    lines.extend(markdown_table(fn_clusters.head(15), ["keyword_bucket_label", "major", "count", "avg_confidence", "visual_evidence_rows", "top_terms"]))

    lines.extend(["", "## FP Clusters To Guard Accuracy", ""])
    lines.extend(markdown_table(fp_clusters.head(15), ["keyword_bucket_label", "major", "count", "avg_confidence", "visual_evidence_rows", "top_terms"]))

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
        "pipe_support_evidence": compact_text(row.get("pipe_support_evidence_text", ""), 700),
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
    parser.add_argument(
        "--include-candidates",
        action="store_true",
        help="Also include candidate-folder predictions.csv files. Default analyzes baseline/current-policy prediction files only.",
    )
    args = parser.parse_args(argv)

    inputs = args.inputs or ["outputs"]
    paths = analyze_prediction_outputs(
        inputs=inputs,
        output_root=args.output_dir,
        max_clusters=args.max_clusters,
        samples_per_cluster=args.samples_per_cluster,
        policy_path=args.policy_path,
        max_policy_chars=args.max_policy_chars,
        include_candidates=args.include_candidates,
    )
    print("[prediction-analysis] done")
    for name, path in paths.items():
        print(f"[prediction-analysis] {name}: {path}")


if __name__ == "__main__":
    main()
