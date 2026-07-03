from __future__ import annotations

from copy import deepcopy
from typing import Any

import pandas as pd

from .evaluate import compute_metrics


def run_guardrail_autotune(
    train_pred_df: pd.DataFrame,
    validation_pred_df: pd.DataFrame,
    config: Any,
) -> dict:
    """Tune route-score guardrail thresholds on train and validate on holdout.

    This is intentionally post-hoc: it does not call the LLM again and does not
    rewrite policy text. It tests narrow 진성->가성 guardrail thresholds using
    the already-recorded route_scorecard values.
    """
    candidates = build_guardrail_candidates(config)
    train_base_metrics = compute_metrics(train_pred_df)
    validation_base_metrics = compute_metrics(validation_pred_df)

    rows = []
    best = None
    for candidate in candidates:
        tuned_train = apply_guardrail_candidate(train_pred_df, candidate)
        train_metrics = compute_metrics(tuned_train)
        train_delta = metric_delta(train_base_metrics, train_metrics)
        accepted, reason = passes_train_gate(train_base_metrics, train_metrics, config)

        tuned_validation = apply_guardrail_candidate(validation_pred_df, candidate)
        validation_metrics = compute_metrics(tuned_validation)
        validation_delta = metric_delta(validation_base_metrics, validation_metrics)

        row = {
            **candidate,
            "accepted_train_gate": accepted,
            "train_gate_reason": reason,
            "train_accuracy": train_metrics.get("accuracy", 0),
            "train_score": train_metrics.get("score", 0),
            "train_true_recall": train_metrics.get("true_recall", 0),
            "train_true_precision": train_metrics.get("true_precision", 0),
            "train_false_recall": train_metrics.get("false_recall", 0),
            "train_false_precision": train_metrics.get("false_precision", 0),
            "train_flips": int(tuned_train.get("route_autotune_applied", pd.Series(dtype=bool)).fillna(False).sum()),
            "train_accuracy_delta": train_delta.get("accuracy", 0),
            "train_score_delta": train_delta.get("score", 0),
            "train_true_recall_delta": train_delta.get("true_recall", 0),
            "train_true_precision_delta": train_delta.get("true_precision", 0),
            "validation_accuracy": validation_metrics.get("accuracy", 0),
            "validation_score": validation_metrics.get("score", 0),
            "validation_true_recall": validation_metrics.get("true_recall", 0),
            "validation_true_precision": validation_metrics.get("true_precision", 0),
            "validation_false_recall": validation_metrics.get("false_recall", 0),
            "validation_false_precision": validation_metrics.get("false_precision", 0),
            "validation_flips": int(tuned_validation.get("route_autotune_applied", pd.Series(dtype=bool)).fillna(False).sum()),
            "validation_accuracy_delta": validation_delta.get("accuracy", 0),
            "validation_score_delta": validation_delta.get("score", 0),
            "validation_true_recall_delta": validation_delta.get("true_recall", 0),
            "validation_true_precision_delta": validation_delta.get("true_precision", 0),
        }
        rows.append(row)

        if accepted and is_better_candidate(row, best):
            best = {
                "candidate": candidate,
                "train_metrics": train_metrics,
                "validation_metrics": validation_metrics,
                "train_delta": train_delta,
                "validation_delta": validation_delta,
                "train_predictions": tuned_train,
                "validation_predictions": tuned_validation,
                "summary_row": row,
            }

    if best is None:
        no_op = {
            "name": "autotuned_guardrail_noop",
            "min_false_score": 999,
            "min_margin": 999,
            "allowed_routes": ["pipe_support", "leak_contact"],
        }
        best = {
            "candidate": no_op,
            "train_metrics": train_base_metrics,
            "validation_metrics": validation_base_metrics,
            "train_delta": metric_delta(train_base_metrics, train_base_metrics),
            "validation_delta": metric_delta(validation_base_metrics, validation_base_metrics),
            "train_predictions": train_pred_df.copy(),
            "validation_predictions": validation_pred_df.copy(),
            "summary_row": {
                **no_op,
                "accepted_train_gate": False,
                "train_gate_reason": "no candidate passed train gate",
                "train_accuracy": train_base_metrics.get("accuracy", 0),
                "train_score": train_base_metrics.get("score", 0),
                "train_true_recall": train_base_metrics.get("true_recall", 0),
                "train_true_precision": train_base_metrics.get("true_precision", 0),
                "train_false_recall": train_base_metrics.get("false_recall", 0),
                "train_false_precision": train_base_metrics.get("false_precision", 0),
                "train_flips": 0,
                "validation_accuracy": validation_base_metrics.get("accuracy", 0),
                "validation_score": validation_base_metrics.get("score", 0),
                "validation_true_recall": validation_base_metrics.get("true_recall", 0),
                "validation_true_precision": validation_base_metrics.get("true_precision", 0),
                "validation_false_recall": validation_base_metrics.get("false_recall", 0),
                "validation_false_precision": validation_base_metrics.get("false_precision", 0),
                "validation_flips": 0,
            },
        }

    return {
        "train_base_metrics": train_base_metrics,
        "validation_base_metrics": validation_base_metrics,
        "candidates": rows,
        "best": best,
    }


def build_guardrail_candidates(config: Any) -> list[dict]:
    false_scores = parse_int_list(
        config.get("runtime", "route_score_autotune_false_scores", fallback="4,5,6,7,8,9,10")
    )
    margins = parse_int_list(config.get("runtime", "route_score_autotune_margins", fallback="1,2,3,4,5,6"))
    routes = [
        route.strip()
        for route in config.get("runtime", "route_score_autotune_routes", fallback="pipe_support,leak_contact").split(",")
        if route.strip()
    ]

    candidates = []
    for false_score in false_scores:
        for margin in margins:
            candidates.append(
                {
                    "name": f"autotuned_guardrail_f{false_score}_m{margin}",
                    "min_false_score": false_score,
                    "min_margin": margin,
                    "allowed_routes": routes,
                }
            )
    return candidates


def apply_guardrail_candidate(pred_df: pd.DataFrame, candidate: dict) -> pd.DataFrame:
    tuned = pred_df.copy(deep=True)
    if tuned.empty:
        tuned["route_autotune_applied"] = False
        return tuned

    ensure_column(tuned, "reason", "")
    ensure_column(tuned, "applied_step", "")
    ensure_column(tuned, "route_original_pred", "")

    applied = []
    reasons = []
    for _, row in tuned.iterrows():
        should_flip = should_flip_row(row, candidate)
        applied.append(should_flip)
        if should_flip:
            reasons.append(
                f"route_autotune: {row.get('route_primary', '')} "
                f"false={row.get('route_false_score', 0)} true={row.get('route_true_score', 0)} "
                f"threshold false>={candidate['min_false_score']} margin>={candidate['min_margin']}"
            )
        else:
            reasons.append("")

    tuned["route_autotune_applied"] = applied
    tuned["route_autotune_reason"] = reasons
    mask = tuned["route_autotune_applied"].map(bool)
    if mask.any():
        tuned.loc[mask, "route_original_pred"] = tuned.loc[mask, "pred"]
        tuned.loc[mask, "pred"] = "가성"
        tuned.loc[mask, "applied_step"] = "가성조건"
        tuned.loc[mask, "reason"] = tuned.loc[mask, "reason"].astype(str) + " / " + tuned.loc[mask, "route_autotune_reason"].astype(str)
        tuned.loc[mask, "correct"] = tuned.loc[mask, "label"] == tuned.loc[mask, "pred"]
    return tuned


def ensure_column(df: pd.DataFrame, column: str, default: Any) -> None:
    if column not in df.columns:
        df[column] = default


def should_flip_row(row: pd.Series, candidate: dict) -> bool:
    if row.get("pred", "") != "진성":
        return False
    if normalize_bool(row.get("exclude_from_metrics", False)):
        return False

    route = str(row.get("route_primary", "")).strip()
    allowed_routes = set(candidate.get("allowed_routes", []))
    if allowed_routes and route not in allowed_routes:
        return False

    true_score = parse_int(row.get("route_true_score", 0))
    false_score = parse_int(row.get("route_false_score", 0))
    min_false_score = int(candidate.get("min_false_score", 6))
    min_margin = int(candidate.get("min_margin", 3))
    return false_score >= min_false_score and (false_score - true_score) >= min_margin


def passes_train_gate(base_metrics: dict, metrics: dict, config: Any) -> tuple[bool, str]:
    min_accuracy_gain = config.getfloat("runtime", "route_score_autotune_min_accuracy_gain", fallback=0.0)
    min_tp_gain = config.getfloat("runtime", "route_score_autotune_min_true_precision_gain", fallback=0.01)
    max_tr_loss = config.getfloat("runtime", "route_score_autotune_max_true_recall_loss", fallback=0.04)
    max_score_loss = config.getfloat("runtime", "route_score_autotune_max_score_loss", fallback=0.0)

    accuracy_gain = metrics.get("accuracy", 0) - base_metrics.get("accuracy", 0)
    tp_gain = metrics.get("true_precision", 0) - base_metrics.get("true_precision", 0)
    tr_loss = base_metrics.get("true_recall", 0) - metrics.get("true_recall", 0)
    score_loss = base_metrics.get("score", 0) - metrics.get("score", 0)

    if accuracy_gain < min_accuracy_gain:
        return False, f"accuracy_gain {accuracy_gain:.4f} < {min_accuracy_gain:.4f}"
    if tp_gain < min_tp_gain:
        return False, f"true_precision_gain {tp_gain:.4f} < {min_tp_gain:.4f}"
    if tr_loss > max_tr_loss:
        return False, f"true_recall_loss {tr_loss:.4f} > {max_tr_loss:.4f}"
    if score_loss > max_score_loss:
        return False, f"score_loss {score_loss:.4f} > {max_score_loss:.4f}"
    return True, (
        f"accepted: accuracy_gain={accuracy_gain:.4f}, true_precision_gain={tp_gain:.4f}, "
        f"true_recall_loss={tr_loss:.4f}, score_loss={score_loss:.4f}"
    )


def is_better_candidate(row: dict, best: dict | None) -> bool:
    if best is None:
        return True
    current_key = candidate_sort_key(row)
    best_key = candidate_sort_key(best["summary_row"])
    return current_key > best_key


def candidate_sort_key(row: dict) -> tuple:
    return (
        float(row.get("validation_score_delta", 0)),
        float(row.get("validation_accuracy_delta", 0)),
        float(row.get("validation_true_precision_delta", 0)),
        -abs(float(row.get("validation_true_recall_delta", 0))),
        float(row.get("train_score_delta", 0)),
    )


def metric_delta(base: dict, current: dict) -> dict:
    keys = [
        "accuracy",
        "score",
        "true_recall",
        "true_precision",
        "false_recall",
        "false_precision",
        "fn_true_as_false",
        "fp_false_as_true",
    ]
    return {key: current.get(key, 0) - base.get(key, 0) for key in keys}


def parse_int_list(text: str) -> list[int]:
    values = []
    for item in str(text).split(","):
        item = item.strip()
        if not item:
            continue
        try:
            values.append(int(item))
        except ValueError:
            continue
    return values


def parse_int(value: Any) -> int:
    try:
        return int(value)
    except Exception:
        return 0


def normalize_bool(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n", "", "nan", "none", "null"}:
        return False
    return False
