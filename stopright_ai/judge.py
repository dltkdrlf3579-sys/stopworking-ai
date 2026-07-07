from __future__ import annotations

import re
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
from .route_score import apply_route_guardrail, build_route_scorecard


def judge_case(
    case: dict,
    llm: Any,
    policy: str,
    mode: str = "tournament",
    trace: bool = False,
    route_score_mode: str = "record",
) -> dict:
    normalized_mode = str(mode or "tournament").strip().lower()
    if normalized_mode in {"stable_vote", "vote", "best_of_3", "bo3", "majority"}:
        return judge_case_stable_vote(
            case=case,
            llm=llm,
            policy=policy,
            trace=trace,
            route_score_mode=route_score_mode,
        )
    if normalized_mode in {"true_confirm", "confirm_true", "double_true", "true_2of2"}:
        return judge_case_true_confirm(
            case=case,
            llm=llm,
            policy=policy,
            trace=trace,
            route_score_mode=route_score_mode,
        )

    return judge_case_once(
        case=case,
        llm=llm,
        policy=policy,
        mode=normalized_mode,
        trace=trace,
        route_score_mode=route_score_mode,
    )


def judge_case_once(
    case: dict,
    llm: Any,
    policy: str,
    mode: str = "tournament",
    trace: bool = False,
    route_score_mode: str = "record",
) -> dict:
    trace_id = str(case.get("id", ""))
    started = time.monotonic()
    trace_log(trace, trace_id, f"start mode={mode}")

    images = case.get("image_data_urls", []) or []
    evidence = invoke_json(llm, EVIDENCE_SYSTEM, evidence_user(case), images=images)
    trace_log(trace, trace_id, f"evidence done images={len(images)} elapsed={time.monotonic() - started:.1f}s")
    route_score_mode = normalize_route_score_mode(route_score_mode)
    route_scorecard = build_route_scorecard(case, evidence) if route_score_mode != "off" else {}
    prompt_route_scorecard = route_scorecard if route_score_mode in {"assist", "guardrail"} else None
    apply_guardrail = route_score_mode == "guardrail"
    trace_log(
        trace,
        trace_id,
        "route-score "
        f"route={route_scorecard.get('primary_route')} "
        f"rec={route_scorecard.get('recommendation')} "
        f"true={route_scorecard.get('true_score')} "
        f"false={route_scorecard.get('false_score')}",
    )

    if mode == "fast":
        final = invoke_json(llm, ARBITER_SYSTEM, arbiter_user(case, evidence, policy, route_scorecard=prompt_route_scorecard))
        trace_log(trace, trace_id, f"arbiter done elapsed={time.monotonic() - started:.1f}s")
        return normalize_decision(case, evidence, final, route_scorecard, apply_guardrail)

    true_argument = invoke_json(llm, ADVOCATE_TRUE_SYSTEM, advocate_user(case, evidence, policy, "진성", prompt_route_scorecard))
    trace_log(trace, trace_id, f"true-advocate done elapsed={time.monotonic() - started:.1f}s")
    false_argument = invoke_json(llm, ADVOCATE_FALSE_SYSTEM, advocate_user(case, evidence, policy, "가성", prompt_route_scorecard))
    trace_log(trace, trace_id, f"false-advocate done elapsed={time.monotonic() - started:.1f}s")
    critic = invoke_json(llm, CRITIC_SYSTEM, critic_user(case, evidence, true_argument, false_argument))
    trace_log(trace, trace_id, f"critic done elapsed={time.monotonic() - started:.1f}s")
    final = invoke_json(
        llm,
        ARBITER_SYSTEM,
        arbiter_user(case, evidence, policy, true_argument, false_argument, critic, prompt_route_scorecard),
    )
    trace_log(trace, trace_id, f"arbiter done elapsed={time.monotonic() - started:.1f}s")

    result = normalize_decision(case, evidence, final, route_scorecard, apply_guardrail)
    result["true_argument"] = true_argument
    result["false_argument"] = false_argument
    result["critic"] = critic
    return result


def judge_case_stable_vote(
    case: dict,
    llm: Any,
    policy: str,
    trace: bool = False,
    route_score_mode: str = "record",
) -> dict:
    trace_id = str(case.get("id", ""))
    started = time.monotonic()
    trace_log(trace, trace_id, "stable-vote start")

    votes: list[dict] = []
    first = judge_case_once(case, llm, policy, mode="tournament", trace=trace, route_score_mode=route_score_mode)
    votes.append(first)
    trace_log(trace, trace_id, f"stable-vote round=1 pred={first.get('pred')} elapsed={time.monotonic() - started:.1f}s")

    second = judge_case_once(case, llm, policy, mode="tournament", trace=trace, route_score_mode=route_score_mode)
    votes.append(second)
    trace_log(trace, trace_id, f"stable-vote round=2 pred={second.get('pred')} elapsed={time.monotonic() - started:.1f}s")

    if first.get("pred") != second.get("pred"):
        third = judge_case_once(case, llm, policy, mode="tournament", trace=trace, route_score_mode=route_score_mode)
        votes.append(third)
        trace_log(trace, trace_id, f"stable-vote round=3 pred={third.get('pred')} elapsed={time.monotonic() - started:.1f}s")

    final = select_vote_winner(votes)
    return attach_vote_metadata(final, votes)


def select_vote_winner(votes: list[dict]) -> dict:
    valid_labels = {"진성", "가성"}
    counts = {label: 0 for label in valid_labels}
    for vote in votes:
        pred = vote.get("pred")
        if pred in counts:
            counts[pred] += 1

    best_label, best_count = max(counts.items(), key=lambda item: item[1])
    if best_count >= 2:
        candidates = [vote for vote in votes if vote.get("pred") == best_label]
        return max(candidates, key=lambda vote: int(vote.get("confidence", 0))).copy()

    valid_votes = [vote for vote in votes if vote.get("pred") in valid_labels]
    if valid_votes:
        return max(valid_votes, key=lambda vote: int(vote.get("confidence", 0))).copy()

    return votes[-1].copy()


def attach_vote_metadata(result: dict, votes: list[dict]) -> dict:
    preds = [str(vote.get("pred", "")) for vote in votes]
    confidences = [int(vote.get("confidence", 0) or 0) for vote in votes]
    true_count = preds.count("진성")
    false_count = preds.count("가성")
    margin = abs(true_count - false_count)
    disagreement = len(set(preds)) > 1
    summary = f"rounds={len(votes)}, votes={'|'.join(preds)}, final={result.get('pred', '')}"

    result["vote_mode"] = "stable_vote"
    result["vote_rounds"] = len(votes)
    result["vote_results"] = preds
    result["vote_confidences"] = confidences
    result["vote_true_count"] = true_count
    result["vote_false_count"] = false_count
    result["vote_margin"] = margin
    result["vote_disagreement"] = disagreement
    result["vote_summary"] = summary
    result["vote_details"] = [
        {
            "round": idx,
            "pred": vote.get("pred", ""),
            "confidence": vote.get("confidence", 0),
            "reason": vote.get("reason", ""),
            "applied_step": vote.get("applied_step", ""),
        }
        for idx, vote in enumerate(votes, start=1)
    ]

    base_reason = str(result.get("reason", "")).strip()
    vote_line = f"[stable_vote] {summary}"
    result["reason"] = f"{base_reason}\n{vote_line}" if base_reason else vote_line
    result["correct"] = result.get("pred") == result.get("label", "")
    return result


def judge_case_true_confirm(
    case: dict,
    llm: Any,
    policy: str,
    trace: bool = False,
    route_score_mode: str = "record",
) -> dict:
    trace_id = str(case.get("id", ""))
    started = time.monotonic()
    trace_log(trace, trace_id, "true-confirm start")

    first = judge_case_once(case, llm, policy, mode="tournament", trace=trace, route_score_mode=route_score_mode)
    trace_log(trace, trace_id, f"true-confirm round=1 pred={first.get('pred')} elapsed={time.monotonic() - started:.1f}s")

    second = judge_case_once(case, llm, policy, mode="tournament", trace=trace, route_score_mode=route_score_mode)
    trace_log(trace, trace_id, f"true-confirm round=2 pred={second.get('pred')} elapsed={time.monotonic() - started:.1f}s")

    votes = [first, second]
    if first.get("pred") == "진성" and second.get("pred") == "진성":
        final = max(votes, key=lambda vote: int(vote.get("confidence", 0))).copy()
    else:
        false_votes = [vote for vote in votes if vote.get("pred") == "가성"]
        if false_votes:
            final = max(false_votes, key=lambda vote: int(vote.get("confidence", 0))).copy()
        else:
            final = first.copy()
            final["pred"] = "가성"
            final["confidence"] = min(int(final.get("confidence", 0) or 0), 50)
            final["applied_step"] = "가성조건"
            final["decisive_evidence"] = final.get("decisive_evidence", [])

    return attach_true_confirm_metadata(final, votes)


def attach_true_confirm_metadata(result: dict, votes: list[dict]) -> dict:
    preds = [str(vote.get("pred", "")) for vote in votes]
    confidences = [int(vote.get("confidence", 0) or 0) for vote in votes]
    confirmed = preds == ["진성", "진성"]
    summary = f"rounds=2, votes={'|'.join(preds)}, true_confirmed={confirmed}, final={result.get('pred', '')}"

    result["vote_mode"] = "true_confirm"
    result["vote_rounds"] = len(votes)
    result["vote_results"] = preds
    result["vote_confidences"] = confidences
    result["vote_true_count"] = preds.count("진성")
    result["vote_false_count"] = preds.count("가성")
    result["vote_margin"] = abs(preds.count("진성") - preds.count("가성"))
    result["vote_disagreement"] = len(set(preds)) > 1
    result["true_confirmed"] = confirmed
    result["vote_summary"] = summary
    result["vote_details"] = [
        {
            "round": idx,
            "pred": vote.get("pred", ""),
            "confidence": vote.get("confidence", 0),
            "reason": vote.get("reason", ""),
            "applied_step": vote.get("applied_step", ""),
        }
        for idx, vote in enumerate(votes, start=1)
    ]

    base_reason = str(result.get("reason", "")).strip()
    vote_line = f"[true_confirm] {summary}"
    result["reason"] = f"{base_reason}\n{vote_line}" if base_reason else vote_line
    result["correct"] = result.get("pred") == result.get("label", "")
    return result


def trace_log(trace: bool, trace_id: str, message: str) -> None:
    if trace:
        print(f"[judge:{trace_id}] {message}", flush=True)


def normalize_decision(
    case: dict,
    evidence: dict,
    final: dict,
    route_scorecard: dict | None = None,
    apply_guardrail: bool = False,
) -> dict:
    pred = str(final.get("판정", "")).strip()
    if pred not in {"진성", "가성"}:
        return invalid_decision(case, evidence, final, f"invalid judgement value: {pred!r}")

    try:
        confidence = parse_int(final.get("확신도", 0))
    except Exception:
        confidence = 0

    result = {
        "id": case.get("id", ""),
        "label": case.get("label", ""),
        "pred": pred,
        "correct": pred == case.get("label", ""),
        "confidence": max(0, min(100, confidence)),
        "reason": str(final.get("판단근거", "")),
        "applied_step": str(final.get("applied_step", "")),
        "review_needed": normalize_bool(final.get("review_needed", False)),
        "decisive_evidence": final.get("decisive_evidence", []),
        "evidence": evidence,
        "route_scorecard": route_scorecard or {},
        "route_recommendation": (route_scorecard or {}).get("recommendation", ""),
        "route_primary": (route_scorecard or {}).get("primary_route", ""),
        "route_true_score": (route_scorecard or {}).get("true_score", 0),
        "route_false_score": (route_scorecard or {}).get("false_score", 0),
        "route_guardrail_applied": False,
        "major": case.get("major", ""),
        "middle": case.get("middle", ""),
        "title": case.get("title", ""),
        "phenomenon_truncated": case.get("phenomenon_truncated", False),
        "action_truncated": case.get("action_truncated", False),
        "image_count": case.get("image_count", 0),
        "original_image_count": case.get("original_image_count", case.get("image_count", 0)),
        "omitted_image_count": case.get("omitted_image_count", 0),
    }
    if apply_guardrail:
        return apply_route_guardrail(result)
    return result


def normalize_route_score_mode(value: str) -> str:
    mode = str(value or "record").strip().lower()
    if mode in {"off", "none", "false", "0"}:
        return "off"
    if mode in {"assist", "prompt"}:
        return "assist"
    if mode in {"guardrail", "override", "false_guardrail"}:
        return "guardrail"
    return "record"


def invalid_decision(case: dict, evidence: dict, final: dict, reason: str) -> dict:
    return {
        "id": case.get("id", ""),
        "label": case.get("label", ""),
        "pred": "보류",
        "correct": False,
        "confidence": 0,
        "reason": f"판정값 오류로 판정 보류: {reason}",
        "applied_step": str(final.get("applied_step", "INVALID_OUTPUT")),
        "review_needed": True,
        "decisive_evidence": final.get("decisive_evidence", []),
        "evidence": evidence,
        "major": case.get("major", ""),
        "middle": case.get("middle", ""),
        "title": case.get("title", ""),
        "phenomenon_truncated": case.get("phenomenon_truncated", False),
        "action_truncated": case.get("action_truncated", False),
        "image_count": case.get("image_count", 0),
        "original_image_count": case.get("original_image_count", case.get("image_count", 0)),
        "omitted_image_count": case.get("omitted_image_count", 0),
        "error": reason,
        "exclude_from_metrics": True,
    }


def parse_int(value: Any) -> int:
    if isinstance(value, int):
        return value
    text = str(value)
    match = re.search(r"-?\d+", text)
    if not match:
        return 0
    return int(match.group(0))


def normalize_bool(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y", "필요", "검토필요"}:
        return True
    if text in {"false", "0", "no", "n", "", "없음", "불필요"}:
        return False
    return False
