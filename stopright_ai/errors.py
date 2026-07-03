from __future__ import annotations

import json
from collections import defaultdict
from typing import Any

import pandas as pd

from .llm_json import invoke_json
from .policy import build_policy_with_addendum
from .prompts import CANDIDATE_SYSTEM, candidate_user


def build_error_clusters(pred_df: pd.DataFrame, max_clusters: int = 12) -> list[dict]:
    excluded = pred_df.get("exclude_from_metrics", False)
    if not isinstance(excluded, pd.Series):
        excluded = pd.Series([bool(excluded)] * len(pred_df), index=pred_df.index)

    eval_df = pred_df[~excluded.map(normalize_bool)].copy()
    errors = eval_df[eval_df["correct"] == False].copy()
    if errors.empty:
        return []

    errors["error_type"] = errors.apply(classify_error_type, axis=1)
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)

    for row in errors.to_dict("records"):
        key = (str(row.get("error_type", "")), str(row.get("major", "")))
        grouped[key].append(row)

    clusters = []
    for (error_type, major), rows in sorted(grouped.items(), key=lambda item: len(item[1]), reverse=True)[:max_clusters]:
        ranked_rows = rank_learning_rows(rows)
        sample_rows = ranked_rows[:5]
        correct_reference_rows = select_correct_reference_rows(eval_df, error_type, major, limit=5)
        regression_guard_rows = select_regression_guard_rows(eval_df, error_type, major, limit=5)
        clusters.append(
            {
                "error_type": error_type,
                "major": major,
                "count": len(rows),
                "improvement_goal": describe_improvement_goal(error_type),
                "samples": [compact_case(r) for r in sample_rows],
                "correct_reference_samples": [compact_case(r) for r in correct_reference_rows],
                "regression_guard_samples": [compact_case(r) for r in regression_guard_rows],
            }
        )

    return clusters


def classify_error_type(row: pd.Series) -> str:
    label = row.get("label")
    pred = row.get("pred")
    if label == "진성" and pred == "가성":
        return "FN_실제진성_가성오판"
    if label == "가성" and pred == "진성":
        return "FP_실제가성_진성오판"
    return "OTHER"


def describe_improvement_goal(error_type: str) -> str:
    if error_type.startswith("FN_"):
        return "실제 진성인데 가성으로 낮춘 오답을 줄이되, 기존 가성 정답을 진성으로 뒤집지 않아야 한다."
    if error_type.startswith("FP_"):
        return "실제 가성인데 진성으로 올린 오답을 줄이되, 기존 진성 정답을 가성으로 뒤집지 않아야 한다."
    return "오답 원인을 좁게 설명할 수 있을 때만 보정한다."


def rank_learning_rows(rows: list[dict]) -> list[dict]:
    def key(row: dict) -> tuple[int, int, str]:
        try:
            confidence = int(row.get("confidence", 0))
        except Exception:
            confidence = 0
        review_needed = bool(row.get("review_needed", False))
        return (0 if not review_needed else 1, -confidence, str(row.get("id", "")))

    return sorted(rows, key=key)


def select_correct_reference_rows(eval_df: pd.DataFrame, error_type: str, major: str, limit: int = 5) -> list[dict]:
    correct = eval_df[eval_df["correct"] == True].copy()
    if correct.empty:
        return []

    if error_type.startswith("FN_"):
        target = correct[(correct["label"] == "진성") & (correct["pred"] == "진성")]
    elif error_type.startswith("FP_"):
        target = correct[(correct["label"] == "가성") & (correct["pred"] == "가성")]
    else:
        target = correct

    same_major = target[target.get("major", "") == major] if "major" in target.columns else target
    selected = same_major if not same_major.empty else target
    return rank_learning_rows(selected.to_dict("records"))[:limit]


def select_regression_guard_rows(eval_df: pd.DataFrame, error_type: str, major: str, limit: int = 5) -> list[dict]:
    correct = eval_df[eval_df["correct"] == True].copy()
    if correct.empty:
        return []

    # FN 보정은 진성 쪽으로 당기기 쉬우므로 기존 TN(가성 정답)을 보호한다.
    # FP 보정은 가성 쪽으로 당기기 쉬우므로 기존 TP(진성 정답)을 보호한다.
    if error_type.startswith("FN_"):
        guard = correct[(correct["label"] == "가성") & (correct["pred"] == "가성")]
    elif error_type.startswith("FP_"):
        guard = correct[(correct["label"] == "진성") & (correct["pred"] == "진성")]
    else:
        guard = correct

    same_major = guard[guard.get("major", "") == major] if "major" in guard.columns else guard
    selected = same_major if not same_major.empty else guard
    return rank_learning_rows(selected.to_dict("records"))[:limit]


def compact_case(row: dict) -> dict:
    return {
        "id": clean_scalar(row.get("id", "")),
        "title": clean_scalar(row.get("title", "")),
        "major": clean_scalar(row.get("major", "")),
        "middle": clean_scalar(row.get("middle", "")),
        "label": clean_scalar(row.get("label", "")),
        "pred": clean_scalar(row.get("pred", "")),
        "confidence": clean_scalar(row.get("confidence", "")),
        "review_needed": bool(clean_scalar(row.get("review_needed", False))),
        "reason": clean_scalar(row.get("reason", "")),
        "applied_step": clean_scalar(row.get("applied_step", "")),
        "decisive_evidence": compact_json(clean_structured(row.get("decisive_evidence", []), []), limit=600),
        "evidence": compact_json(clean_structured(row.get("evidence", {}), {}), limit=800),
        "true_argument": compact_json(clean_structured(row.get("true_argument", {}), {}), limit=600),
        "false_argument": compact_json(clean_structured(row.get("false_argument", {}), {}), limit=600),
        "critic": compact_json(clean_structured(row.get("critic", {}), {}), limit=600),
    }


def generate_candidate_policies(
    llm: Any,
    current_policy: str,
    error_clusters: list[dict],
    candidate_count: int,
    config: Any | None = None,
) -> list[dict]:
    if candidate_count <= 0:
        return []
    if not error_clusters:
        return []

    try:
        response = invoke_json(
            llm,
            CANDIDATE_SYSTEM,
            candidate_user(current_policy=current_policy, error_clusters=error_clusters, candidate_count=candidate_count),
        )
    except Exception as exc:
        print(f"[candidate-generation] failed, skip candidates: {exc}", flush=True)
        return []

    candidates = response.get("candidates", [])
    if not isinstance(candidates, list):
        return []

    clean = []
    for idx, candidate in enumerate(candidates, start=1):
        if not isinstance(candidate, dict):
            continue
        candidate_worthy = candidate.get("candidate_worthy", True)
        if isinstance(candidate_worthy, str):
            candidate_worthy = candidate_worthy.strip().lower() not in {"false", "no", "0", "아니오", "없음"}
        if not candidate_worthy:
            continue

        name = str(candidate.get("name") or f"candidate_{idx}")
        addendum_title = str(candidate.get("addendum_title") or name).strip()
        addendum_text = str(candidate.get("addendum_text") or "").strip()
        if addendum_text:
            try:
                policy_text, rendered_addendum = build_policy_with_addendum(
                    current_policy,
                    addendum_title,
                    addendum_text,
                    config or _AddendumConfig(),
                )
            except Exception as exc:
                print(f"[candidate-generation] skip invalid addendum {name}: {exc}", flush=True)
                continue
            clean.append(
                {
                    "name": name,
                    "hypothesis": str(candidate.get("hypothesis", "")),
                    "target_error_cluster": str(candidate.get("target_error_cluster", "")),
                    "policy_diagnosis": str(candidate.get("policy_diagnosis", "")),
                    "why_wrong": str(candidate.get("why_wrong", "")),
                    "why_correct_cases_remain_safe": str(candidate.get("why_correct_cases_remain_safe", "")),
                    "regression_risk": str(candidate.get("regression_risk", "")),
                    "addendum_title": addendum_title,
                    "addendum_text": addendum_text,
                    "rendered_addendum": rendered_addendum,
                    "policy_text": policy_text,
                }
            )
    return clean


class _AddendumConfig:
    def getint(self, section: str, option: str, fallback: int) -> int:
        return fallback


def clean_scalar(value: Any) -> Any:
    if is_missing(value):
        return ""
    return value


def clean_structured(value: Any, default: Any) -> Any:
    if is_missing(value):
        return default
    return value


def is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        result = pd.isna(value)
    except Exception:
        return False
    if isinstance(result, bool):
        return result
    return False


def normalize_bool(value: Any) -> bool:
    if is_missing(value):
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"true", "1", "yes", "y"}:
            return True
        if text in {"false", "0", "no", "n", ""}:
            return False
    return bool(value)


def compact_json(value: Any, limit: int = 1000) -> str:
    try:
        text = json.dumps(value, ensure_ascii=False)
    except Exception:
        text = str(value)
    if len(text) > limit:
        return text[:limit] + "..."
    return text
