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

    errors = pred_df[(pred_df["correct"] == False) & (~excluded.fillna(False).astype(bool))].copy()
    if errors.empty:
        return []

    errors["error_type"] = errors.apply(classify_error_type, axis=1)
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)

    for row in errors.to_dict("records"):
        key = (str(row.get("error_type", "")), str(row.get("major", "")))
        grouped[key].append(row)

    clusters = []
    for (error_type, major), rows in sorted(grouped.items(), key=lambda item: len(item[1]), reverse=True)[:max_clusters]:
        sample_rows = rows[:5]
        clusters.append(
            {
                "error_type": error_type,
                "major": major,
                "count": len(rows),
                "samples": [
                    {
                        "id": r.get("id", ""),
                        "title": r.get("title", ""),
                        "label": r.get("label", ""),
                        "pred": r.get("pred", ""),
                        "reason": r.get("reason", ""),
                        "evidence": compact_json(r.get("evidence", {}), limit=800),
                    }
                    for r in sample_rows
                ],
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


def generate_candidate_policies(llm: Any, current_policy: str, error_clusters: list[dict], candidate_count: int) -> list[dict]:
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
        name = str(candidate.get("name") or f"candidate_{idx}")
        addendum_title = str(candidate.get("addendum_title") or name).strip()
        addendum_text = str(candidate.get("addendum_text") or "").strip()
        if addendum_text:
            try:
                policy_text, rendered_addendum = build_policy_with_addendum(
                    current_policy,
                    addendum_title,
                    addendum_text,
                    _AddendumConfig(),
                )
            except Exception as exc:
                print(f"[candidate-generation] skip invalid addendum {name}: {exc}", flush=True)
                continue
            clean.append(
                {
                    "name": name,
                    "hypothesis": str(candidate.get("hypothesis", "")),
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


def compact_json(value: Any, limit: int = 1000) -> str:
    try:
        text = json.dumps(value, ensure_ascii=False)
    except Exception:
        text = str(value)
    if len(text) > limit:
        return text[:limit] + "..."
    return text
