from __future__ import annotations

from typing import Any

from .llm_json import invoke_json
from .prompts import (
    ADVOCATE_FALSE_SYSTEM,
    ADVOCATE_TRUE_SYSTEM,
    ARBITER_SYSTEM,
    CRITIC_SYSTEM,
    EVIDENCE_SYSTEM,
    advocate_user,
    arbiter_user,
    critic_user,
    evidence_user,
)


def judge_case(case: dict, llm: Any, policy: str, mode: str = "tournament") -> dict:
    evidence = invoke_json(llm, EVIDENCE_SYSTEM, evidence_user(case))

    if mode == "fast":
        final = invoke_json(llm, ARBITER_SYSTEM, arbiter_user(case, evidence, policy))
        return normalize_decision(case, evidence, final)

    true_argument = invoke_json(llm, ADVOCATE_TRUE_SYSTEM, advocate_user(case, evidence, policy, "진성"))
    false_argument = invoke_json(llm, ADVOCATE_FALSE_SYSTEM, advocate_user(case, evidence, policy, "가성"))
    critic = invoke_json(llm, CRITIC_SYSTEM, critic_user(case, evidence, true_argument, false_argument))
    final = invoke_json(llm, ARBITER_SYSTEM, arbiter_user(case, evidence, policy, true_argument, false_argument, critic))

    result = normalize_decision(case, evidence, final)
    result["true_argument"] = true_argument
    result["false_argument"] = false_argument
    result["critic"] = critic
    return result


def normalize_decision(case: dict, evidence: dict, final: dict) -> dict:
    pred = str(final.get("판정", "")).strip()
    if pred not in {"진성", "가성"}:
        pred = "진성" if "진" in pred else "가성"

    try:
        confidence = int(final.get("확신도", 0))
    except Exception:
        confidence = 0

    return {
        "id": case.get("id", ""),
        "label": case.get("label", ""),
        "pred": pred,
        "correct": pred == case.get("label", ""),
        "confidence": max(0, min(100, confidence)),
        "reason": str(final.get("판단근거", "")),
        "applied_step": str(final.get("applied_step", "")),
        "review_needed": bool(final.get("review_needed", False)),
        "decisive_evidence": final.get("decisive_evidence", []),
        "evidence": evidence,
        "major": case.get("major", ""),
        "middle": case.get("middle", ""),
        "title": case.get("title", ""),
    }

