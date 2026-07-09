from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

import pandas as pd


JIN = "진성"
GA = "가성"

ACTIVE_PHASES = {"작업중", "작업개시직전"}
ACTIVE_WORK = {"실제작업중", "작업개시직전위험구역진입"}
PREWORK_PHASES = {"작업전계획", "사전협의", "작업예정", "기간연장", "불명확"}
PREWORK_WORK = {"작업전현장확인", "작업예정문서협의", "기간연장요청", "불명확"}
POSITIVE_FORCE = {"실제밟음", "밟을수밖에없음", "안전발판부족"}
WEAK_FORCE = {"작업방법상가능성", "단순검토", "없음", "불명확", "해당없음"}
POSITIVE_PURPOSES = {"안전기준이탈승인", "위험통제협의", "물리적보강협의", "대체작업방법승인"}
ADMIN_PURPOSES = {"단순이메일협의", "일정조율", "일반허가", "단순공유", "해당없음", "불명확", ""}
POSITIVE_COUNTERMEASURES = {"보강판", "비계", "임시발판", "난간", "덮개", "작업방법변경", "설비보호조치"}
EMPTY_COUNTERMEASURES = {"없음", "불명확", "해당없음", ""}


def simulate_pipe_support_gate(pred_df: pd.DataFrame, profile: str = "balanced") -> pd.DataFrame:
    """Return a copy with a deterministic pipe/support false gate applied.

    The gate only flips ``진성`` to ``가성``. It never upgrades a case to ``진성``.
    It is intentionally narrow and based on structured evidence fields rather
    than route_score text keyword totals.
    """
    work = pred_df.copy()
    if work.empty:
        return work

    if "pred_before_pipe_gate" not in work:
        work["pred_before_pipe_gate"] = work.get("pred", "")
    work["pipe_gate_profile"] = profile
    work["pipe_gate_applied"] = False
    work["pipe_gate_reason"] = ""

    for idx, row in work.iterrows():
        if normalize_label(row.get("pred", "")) != JIN:
            continue
        hit, reason = pipe_support_false_gate(row, profile=profile)
        if not hit:
            continue

        work.at[idx, "pred"] = GA
        work.at[idx, "pipe_gate_applied"] = True
        work.at[idx, "pipe_gate_reason"] = reason
        work.at[idx, "applied_step"] = "pipe_support_false_gate"
        if "reason" in work.columns:
            old_reason = safe_str(row.get("reason", "")).strip()
            gate_note = f"pipe_support_false_gate: {reason}"
            work.at[idx, "reason"] = f"{old_reason}\n{gate_note}" if old_reason else gate_note
        if "label" in work.columns:
            work.at[idx, "correct"] = normalize_label(row.get("label", "")) == GA
    return work


def pipe_support_false_gate(row: pd.Series | dict[str, Any], profile: str = "balanced") -> tuple[bool, str]:
    evidence = parse_structured(row.get("evidence", {}))
    pipe = evidence.get("pipe_support_evidence", {}) if isinstance(evidence, dict) else {}
    if not isinstance(pipe, dict):
        return False, "no_pipe_evidence"

    is_pipe_case = field(pipe, "is_pipe_support_case")
    route_primary = field(row, "route_primary")
    if is_pipe_case != "예" and route_primary != "pipe_support":
        return False, "not_pipe_support"

    phase = field(pipe, "work_phase_for_pipe")
    actual_work = field(pipe, "actual_or_planned_work")
    stepping = field(pipe, "stepping_context")
    forced = field(pipe, "forced_stepping_level")
    standard_blocked = field(pipe, "standard_method_blocked")
    permission_required = field(pipe, "permission_required_by_company_rule")
    approval_purpose = field(pipe, "approval_purpose")
    admin_only = field(pipe, "admin_only_signal")
    physical_status = field(pipe, "physical_action_status")
    reinforcement = field(pipe, "reinforcement_or_method_change")
    countermeasures = set(list_field(pipe, "physical_countermeasure"))

    phase_positive = phase in ACTIVE_PHASES or actual_work in ACTIVE_WORK
    prework_or_unclear = phase in PREWORK_PHASES or actual_work in PREWORK_WORK
    force_positive = forced in POSITIVE_FORCE or stepping in {"실제밟음", "밟을필요있음"}
    force_weak = forced in WEAK_FORCE or stepping in {"밟음가능성검토", "단순허가요청", "단순계획", "없음", "불명확"}
    approval_positive = (
        standard_blocked == "예"
        and permission_required == "예"
        and approval_purpose in POSITIVE_PURPOSES
    )
    countermeasure_positive = (
        bool(countermeasures & POSITIVE_COUNTERMEASURES)
        and (physical_status in {"실시됨", "실시필요"} or reinforcement == "있음")
    )
    countermeasure_weak = not countermeasures or countermeasures <= EMPTY_COUNTERMEASURES or physical_status in {"검토만", "없음", "불명확", "해당없음"}
    admin_purpose = approval_purpose in ADMIN_PURPOSES or admin_only == "예"
    admin_negative = (
        admin_only == "예"
        or approval_purpose in ADMIN_PURPOSES
        or prework_or_unclear
        or force_weak
        or countermeasure_weak
    )

    positive_count = sum([phase_positive, force_positive, approval_positive, countermeasure_positive])
    negative_count = sum([admin_purpose, prework_or_unclear, force_weak, countermeasure_weak])
    strong_true = phase_positive and (force_positive or approval_positive or countermeasure_positive)

    if strong_true:
        return False, "strong_pipe_true_evidence"

    if profile == "strict":
        if (
            positive_count == 0
            and admin_purpose
            and prework_or_unclear
            and force_weak
            and countermeasure_weak
        ):
            return True, "strict_admin_or_unclear_no_positive_pipe_evidence"
        return False, "strict_no_gate"

    if profile == "approval_only":
        if (
            positive_count == 0
            and admin_purpose
            and not phase_positive
            and not force_positive
            and not countermeasure_positive
        ):
            return True, "approval_only_admin_or_unclear_no_positive_pipe_evidence"
        return False, "approval_only_no_gate"

    if profile == "balanced":
        if admin_negative and positive_count == 0 and negative_count >= 2:
            return True, "balanced_admin_or_unclear_no_positive_pipe_evidence"
        if prework_or_unclear and not force_positive and not countermeasure_positive and not approval_positive:
            return True, "balanced_prework_no_force_countermeasure_or_approval"
        if approval_purpose in ADMIN_PURPOSES and not phase_positive and not force_positive and not countermeasure_positive:
            return True, "balanced_admin_approval_without_physical_evidence"
        return False, "balanced_no_gate"

    raise ValueError(f"Unknown pipe gate profile: {profile}")


def compute_metrics(pred_df: pd.DataFrame, pred_col: str = "pred", label_col: str = "label") -> dict[str, Any]:
    if pred_df.empty:
        return empty_metrics()
    if label_col not in pred_df or pred_col not in pred_df:
        return empty_metrics(total_n=len(pred_df))

    work = pred_df[build_eval_mask(pred_df, pred_col=pred_col, label_col=label_col)].copy()

    labels = work[label_col].map(normalize_label)
    preds = work[pred_col].map(normalize_label)

    n = int(len(work))
    if n == 0:
        return empty_metrics(total_n=len(pred_df))

    tp = int(((labels == JIN) & (preds == JIN)).sum())
    fn = int(((labels == JIN) & (preds == GA)).sum())
    fp = int(((labels == GA) & (preds == JIN)).sum())
    tn = int(((labels == GA) & (preds == GA)).sum())
    return {
        "total_n": int(len(pred_df)),
        "n": n,
        "accuracy": safe_div(tp + tn, n),
        "true_precision": safe_div(tp, tp + fp),
        "true_recall": safe_div(tp, tp + fn),
        "false_precision": safe_div(tn, tn + fn),
        "false_recall": safe_div(tn, tn + fp),
        "tp_true": tp,
        "fn_true_as_false": fn,
        "fp_false_as_true": fp,
        "tn_false": tn,
        "ai_true_count": int((preds == JIN).sum()),
        "ai_false_count": int((preds == GA).sum()),
        "label_true_count": int((labels == JIN).sum()),
        "label_false_count": int((labels == GA).sum()),
    }


def build_eval_mask(pred_df: pd.DataFrame, pred_col: str = "pred", label_col: str = "label") -> pd.Series:
    if pred_df.empty:
        return pd.Series([], dtype=bool, index=pred_df.index)
    if label_col not in pred_df or pred_col not in pred_df:
        return pd.Series([False] * len(pred_df), index=pred_df.index)

    excluded = pred_df.get("exclude_from_metrics", False)
    if isinstance(excluded, pd.Series):
        excluded_mask = excluded.map(normalize_bool)
    else:
        excluded_mask = pd.Series([normalize_bool(excluded)] * len(pred_df), index=pred_df.index)

    labels = pred_df[label_col].map(normalize_label)
    preds = pred_df[pred_col].map(normalize_label)
    valid = labels.isin({JIN, GA}) & preds.isin({JIN, GA})
    return (~excluded_mask) & valid


def summarize_gate(base_df: pd.DataFrame, gated_df: pd.DataFrame, name: str) -> dict[str, Any]:
    compare_df = gated_df.copy()
    if "pred_before_pipe_gate" not in compare_df:
        compare_df["pred_before_pipe_gate"] = base_df.get("pred", "")

    base = compute_metrics(compare_df, pred_col="pred_before_pipe_gate")
    gated = compute_metrics(compare_df, pred_col="pred")
    flips = gated_df[gated_df.get("pipe_gate_applied", False).map(normalize_bool)].copy()
    changed_pred_rows = int(
        (
            compare_df["pred_before_pipe_gate"].map(normalize_label)
            != compare_df["pred"].map(normalize_label)
        ).sum()
    )
    eval_mask = build_eval_mask(compare_df)
    changed_eval_rows = int(
        (
            eval_mask
            & (
                compare_df["pred_before_pipe_gate"].map(normalize_label)
                != compare_df["pred"].map(normalize_label)
            )
        ).sum()
    )
    eval_flips = flips[build_eval_mask(flips)].copy() if not flips.empty else flips.copy()
    if "label" in flips:
        flip_good = int((flips["label"].map(normalize_label) == GA).sum())
        flip_bad = int((flips["label"].map(normalize_label) == JIN).sum())
        eval_flip_good = int((eval_flips["label"].map(normalize_label) == GA).sum()) if not eval_flips.empty else 0
        eval_flip_bad = int((eval_flips["label"].map(normalize_label) == JIN).sum()) if not eval_flips.empty else 0
    else:
        flip_good = 0
        flip_bad = 0
        eval_flip_good = 0
        eval_flip_bad = 0

    return {
        "profile": name,
        "n": gated.get("n", base.get("n", 0)),
        "flips": int(len(flips)),
        "changed_pred_rows": changed_pred_rows,
        "changed_eval_rows": changed_eval_rows,
        "eval_flips": int(len(eval_flips)),
        "flip_good_false_to_false": flip_good,
        "flip_bad_true_to_false": flip_bad,
        "flip_precision": safe_div(flip_good, flip_good + flip_bad),
        "eval_flip_good_false_to_false": eval_flip_good,
        "eval_flip_bad_true_to_false": eval_flip_bad,
        "eval_flip_precision": safe_div(eval_flip_good, eval_flip_good + eval_flip_bad),
        "accuracy_before": base.get("accuracy", 0),
        "accuracy_after": gated.get("accuracy", 0),
        "accuracy_delta": gated.get("accuracy", 0) - base.get("accuracy", 0),
        "TP_before": base.get("true_precision", 0),
        "TP_after": gated.get("true_precision", 0),
        "TP_delta": gated.get("true_precision", 0) - base.get("true_precision", 0),
        "TR_before": base.get("true_recall", 0),
        "TR_after": gated.get("true_recall", 0),
        "TR_delta": gated.get("true_recall", 0) - base.get("true_recall", 0),
        "FPerr_before": base.get("fp_false_as_true", 0),
        "FPerr_after": gated.get("fp_false_as_true", 0),
        "FPerr_delta": gated.get("fp_false_as_true", 0) - base.get("fp_false_as_true", 0),
        "FN_before": base.get("fn_true_as_false", 0),
        "FN_after": gated.get("fn_true_as_false", 0),
        "FN_delta": gated.get("fn_true_as_false", 0) - base.get("fn_true_as_false", 0),
        "ai_true_before": base.get("ai_true_count", 0),
        "ai_true_after": gated.get("ai_true_count", 0),
        "ai_true_delta": gated.get("ai_true_count", 0) - base.get("ai_true_count", 0),
    }


def build_gate_diagnostics(pred_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, row in pred_df.iterrows():
        evidence = parse_structured(row.get("evidence", {}))
        pipe = evidence.get("pipe_support_evidence", {}) if isinstance(evidence, dict) else {}
        if not isinstance(pipe, dict):
            pipe = {}

        route_primary = field(row, "route_primary")
        is_pipe_case = field(pipe, "is_pipe_support_case")
        pred = normalize_label(row.get("pred", ""))
        is_candidate = pred == JIN and (route_primary == "pipe_support" or is_pipe_case == "예")

        diagnostic = {
            "id": field(row, "id"),
            "label": normalize_label(row.get("label", "")),
            "pred": pred,
            "is_gate_candidate": is_candidate,
            "route_primary": route_primary,
            "is_pipe_support_case": is_pipe_case,
            "work_phase_for_pipe": field(pipe, "work_phase_for_pipe"),
            "actual_or_planned_work": field(pipe, "actual_or_planned_work"),
            "stepping_context": field(pipe, "stepping_context"),
            "forced_stepping_level": field(pipe, "forced_stepping_level"),
            "standard_method_blocked": field(pipe, "standard_method_blocked"),
            "permission_required_by_company_rule": field(pipe, "permission_required_by_company_rule"),
            "approval_purpose": field(pipe, "approval_purpose"),
            "admin_only_signal": field(pipe, "admin_only_signal"),
            "physical_action_status": field(pipe, "physical_action_status"),
            "reinforcement_or_method_change": field(pipe, "reinforcement_or_method_change"),
            "physical_countermeasure": "|".join(list_field(pipe, "physical_countermeasure")),
        }
        for profile in ["strict", "balanced", "approval_only"]:
            hit, reason = pipe_support_false_gate(row, profile=profile)
            diagnostic[f"{profile}_hit"] = hit
            diagnostic[f"{profile}_reason"] = reason
        rows.append(diagnostic)
    return pd.DataFrame(rows)


def build_gate_diagnostics_summary(diagnostics: pd.DataFrame) -> pd.DataFrame:
    if diagnostics.empty:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []

    def add(name: str, value: Any) -> None:
        rows.append({"metric": name, "value": value})

    add("total_rows", len(diagnostics))
    add("pred_true_rows", int((diagnostics["pred"] == JIN).sum()))
    add("pipe_route_rows", int((diagnostics["route_primary"] == "pipe_support").sum()))
    add("pipe_evidence_yes_rows", int((diagnostics["is_pipe_support_case"] == "예").sum()))
    add("gate_candidate_rows", int(diagnostics["is_gate_candidate"].map(normalize_bool).sum()))
    for profile in ["strict", "balanced", "approval_only"]:
        hit_mask = diagnostics[f"{profile}_hit"].map(normalize_bool)
        candidate_mask = diagnostics["is_gate_candidate"].map(normalize_bool)
        candidate_hit = diagnostics[hit_mask & candidate_mask]
        add(f"{profile}_hit_rows_all_preds", int(hit_mask.sum()))
        add(f"{profile}_candidate_hit_rows_actual_flips", int(len(candidate_hit)))
        if "label" in candidate_hit:
            add(f"{profile}_candidate_hit_actual_false", int((candidate_hit["label"] == GA).sum()))
            add(f"{profile}_candidate_hit_actual_true", int((candidate_hit["label"] == JIN).sum()))
            add(
                f"{profile}_candidate_hit_precision",
                round(safe_div(int((candidate_hit["label"] == GA).sum()), len(candidate_hit)), 4),
            )

    for col in [
        "work_phase_for_pipe",
        "actual_or_planned_work",
        "stepping_context",
        "forced_stepping_level",
        "standard_method_blocked",
        "permission_required_by_company_rule",
        "approval_purpose",
        "admin_only_signal",
        "physical_action_status",
        "reinforcement_or_method_change",
    ]:
        counts = diagnostics.loc[diagnostics["is_gate_candidate"].map(normalize_bool), col].value_counts(dropna=False).head(12)
        for key, count in counts.items():
            add(f"candidate_{col}={key}", int(count))

    for profile in ["strict", "balanced", "approval_only"]:
        counts = diagnostics.loc[diagnostics["is_gate_candidate"].map(normalize_bool), f"{profile}_reason"].value_counts(dropna=False)
        for key, count in counts.items():
            add(f"{profile}_reason={key}", int(count))

    return pd.DataFrame(rows)


def load_prediction_files(input_path: Path, include_candidates: bool = False) -> pd.DataFrame:
    paths = find_prediction_files(input_path, include_candidates=include_candidates)
    frames = []
    for path in paths:
        frame = pd.read_csv(path)
        frame["source_file"] = str(path)
        frames.append(frame)
    if not frames:
        raise FileNotFoundError(f"No prediction CSV files found under {input_path}")
    return pd.concat(frames, ignore_index=True)


def find_prediction_files(input_path: Path, include_candidates: bool = False) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    names = {
        "train_baseline_predictions.csv",
        "validation_baseline_predictions.csv",
        "train_predictions.csv",
        "validation_predictions.csv",
    }
    paths = []
    for path in input_path.rglob("*.csv"):
        if path.name not in names and path.name != "predictions.csv":
            continue
        if path.name == "predictions.csv" and not include_candidates:
            continue
        parts = {part.lower() for part in path.parts}
        if not include_candidates and any("candidate" in part for part in parts):
            continue
        paths.append(path)
    return sorted(paths)


def parse_structured(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return value
    if value is None:
        return {}
    try:
        if pd.isna(value):
            return {}
    except Exception:
        pass
    text = str(value).strip()
    if not text:
        return {}
    try:
        return json.loads(text)
    except Exception:
        pass
    try:
        return ast.literal_eval(text)
    except Exception:
        return {}


def field(obj: Any, key: str) -> str:
    if isinstance(obj, pd.Series):
        value = obj.get(key, "")
    elif isinstance(obj, dict):
        value = obj.get(key, "")
    else:
        value = ""
    return safe_str(value).strip()


def list_field(obj: dict[str, Any], key: str) -> list[str]:
    value = obj.get(key, [])
    if isinstance(value, list):
        return [safe_str(item).strip() for item in value if safe_str(item).strip()]
    if isinstance(value, str):
        return [part.strip() for part in value.split("|") if part.strip()]
    return []


def normalize_label(value: Any) -> str:
    text = safe_str(value).strip()
    if text in {JIN, GA}:
        return text
    if "진" in text:
        return JIN
    if "가" in text:
        return GA
    return text


def normalize_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    try:
        if pd.isna(value):
            return False
    except Exception:
        pass
    text = str(value).strip().lower()
    compact = text.replace(" ", "")
    true_values = {"true", "1", "yes", "y", "t", "예", "네", "맞음", "있음", "해당"}
    false_values = {
        "false",
        "0",
        "no",
        "n",
        "f",
        "",
        "아니오",
        "아니요",
        "아님",
        "없음",
        "해당없음",
        "해당없슴",
        "없슴",
        "nan",
        "none",
        "null",
    }
    if compact in true_values:
        return True
    if compact in false_values:
        return False
    return bool(value)


def safe_str(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    return str(value)


def safe_div(numerator: int | float, denominator: int | float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def empty_metrics(total_n: int = 0) -> dict[str, Any]:
    return {
        "total_n": total_n,
        "n": 0,
        "accuracy": 0,
        "true_precision": 0,
        "true_recall": 0,
        "false_precision": 0,
        "false_recall": 0,
        "tp_true": 0,
        "fn_true_as_false": 0,
        "fp_false_as_true": 0,
        "tn_false": 0,
        "ai_true_count": 0,
        "ai_false_count": 0,
        "label_true_count": 0,
        "label_false_count": 0,
    }
