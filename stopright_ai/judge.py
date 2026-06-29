from __future__ import annotations

import time
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


def judge_case(case: dict, llm: Any, policy: str, mode: str = "tournament", trace: bool = False) -> dict:
    trace_id = str(case.get("id", ""))
    started = time.monotonic()
    trace_log(trace, trace_id, f"start mode={mode}")

    evidence = invoke_json(llm, EVIDENCE_SYSTEM, evidence_user(case))
    trace_log(trace, trace_id, f"evidence done elapsed={time.monotonic() - started:.1f}s")

    if mode == "fast":
        final = invoke_json(llm, ARBITER_SYSTEM, arbiter_user(case, evidence, policy))
        trace_log(trace, trace_id, f"arbiter done elapsed={time.monotonic() - started:.1f}s")
        return normalize_decision(case, evidence, final)

    true_argument = invoke_json(llm, ADVOCATE_TRUE_SYSTEM, advocate_user(case, evidence, policy, "진성"))
    trace_log(trace, trace_id, f"true-advocate done elapsed={time.monotonic() - started:.1f}s")
    false_argument = invoke_json(llm, ADVOCATE_FALSE_SYSTEM, advocate_user(case, evidence, policy, "가성"))
    trace_log(trace, trace_id, f"false-advocate done elapsed={time.monotonic() - started:.1f}s")
    critic = invoke_json(llm, CRITIC_SYSTEM, critic_user(case, evidence, true_argument, false_argument))
    trace_log(trace, trace_id, f"critic done elapsed={time.monotonic() - started:.1f}s")
    final = invoke_json(llm, ARBITER_SYSTEM, arbiter_user(case, evidence, policy, true_argument, false_argument, critic))
    trace_log(trace, trace_id, f"arbiter done elapsed={time.monotonic() - started:.1f}s")

    result = normalize_decision(case, evidence, final)
    result["true_argument"] = true_argument
    result["false_argument"] = false_argument
    result["critic"] = critic
    return result


def trace_log(trace: bool, trace_id: str, message: str) -> None:
    if trace:
        print(f"[judge:{trace_id}] {message}", flush=True)


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
