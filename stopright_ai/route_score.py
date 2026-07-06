from __future__ import annotations

import json
from pathlib import Path
from typing import Any


JIN = "진성"
GA = "가성"
REVIEW = "경계"
ACTIVE_PROFILE: dict[str, Any] = {}


def set_active_route_score_profile(profile: dict | None) -> None:
    global ACTIVE_PROFILE
    ACTIVE_PROFILE = profile or {}


def get_active_route_score_profile() -> dict:
    return ACTIVE_PROFILE or {}


def configure_route_score_profile_from_config(config: Any) -> dict:
    profile = load_route_score_profile_from_config(config)
    set_active_route_score_profile(profile)
    return profile


def load_route_score_profile_from_config(config: Any) -> dict:
    if not config.getboolean("runtime", "route_score_profile_enabled", fallback=True):
        return {}
    profile_path = Path(config.get("runtime", "route_score_profile_path", fallback="artifacts/route_score_profile.json"))
    if not profile_path.exists():
        return {}
    try:
        profile = json.loads(profile_path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"[route_score_profile] failed to load {profile_path}: {exc}", flush=True)
        return {}
    if not isinstance(profile, dict):
        return {}
    return profile.get("profile", profile)


def build_route_scorecard(case: dict, evidence: dict, profile: dict | None = None) -> dict:
    """Build a lightweight, auditable scorecard for recurring hard routes.

    The scorecard does not use labels. It only summarizes signals from the case
    text and extracted evidence so the final judge has a stable middle layer
    between raw free text and policy prose.
    """
    case_text = flatten_values(case)
    evidence_text = flatten_values(evidence)
    text = f"{case_text}\n{evidence_text}".lower()

    profile = profile if profile is not None else get_active_route_score_profile()

    route_cards = []
    pipe_card = score_pipe_support_route(text, profile)
    if pipe_card["triggered"]:
        route_cards.append(pipe_card)

    leak_card = score_leak_contact_route(text, profile)
    if leak_card["triggered"]:
        route_cards.append(leak_card)

    standard_card = score_standard_deviation_route(text, profile)
    if standard_card["triggered"]:
        route_cards.append(standard_card)

    if not route_cards:
        return {
            "primary_route": "general",
            "recommendation": REVIEW,
            "true_score": 0,
            "false_score": 0,
            "margin": 0,
            "review_needed": False,
            "guardrail_override": False,
            "reason": "특수 점수판 대상 아님",
            "routes": [],
        }

    primary = max(route_cards, key=lambda card: card["true_score"] + card["false_score"])
    recommendations = {card["recommendation"] for card in route_cards}
    conflicting_routes = len(recommendations - {REVIEW}) > 1

    recommendation = primary["recommendation"]
    review_needed = primary["review_needed"] or conflicting_routes
    reason = primary["reason"]
    if conflicting_routes:
        recommendation = REVIEW
        reason = "복수 특수 루트의 권고가 충돌함"

    return {
        "primary_route": primary["route"],
        "recommendation": recommendation,
        "true_score": primary["true_score"],
        "false_score": primary["false_score"],
        "margin": primary["true_score"] - primary["false_score"],
        "review_needed": review_needed,
        "guardrail_override": should_guardrail_override_to_false(primary),
        "profile_name": profile.get("name", "base") if profile else "base",
        "reason": reason,
        "routes": route_cards,
    }


def apply_route_guardrail(result: dict) -> dict:
    """Apply only high-confidence false guardrails.

    This intentionally avoids broad true-side overrides. The current failure
    mode is low true precision, so we only correct a 진성 prediction to 가성
    when the route scorecard has strong, explicit false signals.
    """
    scorecard = result.get("route_scorecard") or {}
    if result.get("pred") != JIN:
        return result
    if not scorecard.get("guardrail_override"):
        return result

    original_pred = result.get("pred")
    result["pred"] = GA
    result["correct"] = result.get("label", "") == GA
    result["applied_step"] = "가성조건"
    result["review_needed"] = bool(scorecard.get("review_needed", False))
    result["route_guardrail_applied"] = True
    result["route_original_pred"] = original_pred
    result["reason"] = (
        f"{result.get('reason', '')} / route_score_guardrail: "
        f"{scorecard.get('primary_route')}에서 명확한 가성 제외조건이 진성 신호보다 강함 "
        f"(true={scorecard.get('true_score')}, false={scorecard.get('false_score')})."
    ).strip()
    return result


def score_pipe_support_route(text: str, profile: dict | None = None) -> dict:
    true_score = 0
    false_score = 0
    true_signals: list[str] = []
    false_signals: list[str] = []

    route_terms = [
        "배관",
        "서포트",
        "support",
        "발판",
        "사다리",
        "비계",
        "난간",
        "개구부",
        "공간 협소",
        "공간부족",
        "동선",
        "밟",
        "고소",
        "추락",
        "낙하",
        "grating",
        "toxic duct",
        "덕트",
    ]
    triggered = has_any(text, route_terms)

    add_true = score_adder("true", true_signals)
    add_false = score_adder("false", false_signals)

    if has_any(text, ["밟음", "밟고", "밟아", "밟아야", "밟을", "밟는", "밟고 작업"]):
        true_score += add_true(3, "밟으면 안 되는 설비를 밟거나 밟을 필요")
    if has_any(text, ["케미컬", "정제질소", "toxic", "가스 배관", "가동 설비", "설비 프레임", "덕트"]):
        true_score += add_true(2, "밟으면 안 되는 설비 유형")
    if has_any(text, ["발판 없음", "발판 부족", "동선 없음", "이동경로 없음", "접근 불가", "작업 위치 접근", "공간 협소", "공간부족", "협소"]):
        true_score += add_true(2, "안전한 발판 또는 이동경로 부족")
    if has_any(text, ["비계", "임시발판", "보강판", "커버", "난간", "덮개", "작업방법 변경", "시공그룹", "기술적", "부서 협의", "손들기"]):
        true_score += add_true(2, "물리적 보강 또는 기술적 협의 필요")
    if has_any(text, ["추락", "낙하", "끼임", "충돌", "파손", "누출", "감전", "접촉", "협착"]):
        true_score += add_true(2, "구체적 물리적 결과 위험")
    if has_any(text, ["작업중", "작업 중", "진행 중", "작업진행", "시공시", "상부고소작업", "작업 개시"]):
        true_score += add_true(1, "작업 중 또는 개시 직전 표현")

    if has_any(text, ["향후", "작업 예정", "작업예정", "예정일", "계획 단계", "계획 검토", "장거리 시공", "기간 연장", "연장 요청", "허가 요청", "승인 요청", "설계 검토", "일정 조율", "dri"]):
        false_score += add_false(4, "향후 계획·허가·일정 조율 중심")
    if has_any(text, ["향후", "작업 예정", "작업예정", "계획", "기간 연장", "허가 요청", "승인 요청"]) and has_any(text, ["협의", "요청", "승인", "연장"]):
        false_score += add_false(2, "실제 작업중지보다 행정 요청 성격이 강함")
    if has_any(text, ["사전 협의", "사전 검토", "사전 확인", "작업 전 협의", "작업전 협의"]):
        false_score += add_false(2, "사전 협의 또는 사전 확인 중심")
    if has_any(text, ["미실시", "미진행", "작업 전", "작업전", "일반 점검", "순회점검"]):
        false_score += add_false(2, "위험 노출 전 점검 가능성")
    if has_any(text, ["통로확보", "통로 확보", "정리정돈", "위치 변경", "자재 이동", "교육", "표지", "보호구", "재체결"]):
        false_score += add_false(2, "단순 시정 가능 신호")

    card = finalize_route_card("pipe_support", triggered, true_score, false_score, true_signals, false_signals)
    return apply_profile_to_route_card(card, profile)


def score_leak_contact_route(text: str, profile: dict | None = None) -> dict:
    true_score = 0
    false_score = 0
    true_signals: list[str] = []
    false_signals: list[str] = []

    route_terms = [
        "누출",
        "누설",
        "접액",
        "접촉",
        "흡입",
        "냄새",
        "가스",
        "미상",
        "응축수",
        "diw",
        "액체",
        "연기",
        "분진",
        "약품",
        "leak",
        "밸브",
    ]
    triggered = has_any(text, route_terms)

    add_true = score_adder("true", true_signals)
    add_false = score_adder("false", false_signals)

    if has_any(text, ["액체", "물", "누수", "누출", "응축수", "냄새", "가스", "약품 흔적", "접액"]):
        true_score += add_true(3, "작업 중 실제 물질 또는 접액 우려 발견")
    if has_any(text, ["미상", "성분 미상", "정체불명", "확인 필요", "성분 확인", "화학물질 가능", "케미컬", "약품", "유해", "안전성", "확신"]):
        true_score += add_true(2, "작업중지 당시 안전성 불확실성")
    if has_any(text, ["접액", "접촉", "흡입", "노출", "냄새", "비산", "작업자 노출"]):
        true_score += add_true(2, "작업자 접촉·흡입·노출 가능성")
    if has_any(text, ["운영부서", "담당부서", "유선전달", "전달", "확인 후 재개", "재개승인", "누출원 확인", "원인 확인", "차단", "격리", "방제", "환기", "가스측정", "ert", "신고", "현장 이탈", "대피", "전문 조치", "성분 분석"]):
        true_score += add_true(3, "확인·차단·격리·방제 등 전문 조치 필요")
    if has_any(text, ["화재", "폭발", "감전", "질식", "미끄러짐", "추가 누출", "설비 손상", "분출", "활성 누출"]):
        true_score += add_true(2, "구체적 2차 위험")
    if has_any(text, ["사후", "이후", "추후"]) and has_any(text, ["diw", "응축수", "물", "무해"]):
        true_score += add_true(2, "사후 무해 확인이나 당시 불확실성 가능성")

    if has_any(text, ["실제 액체 없음", "실제 물질 없음", "센서 테스트", "sw 경고", "테스트 알람", "계획정비 알람"]):
        false_score += add_false(4, "실제 물질 없는 테스트·알람 신호")
    if has_any(text, ["실제 누출 없음", "누출 없음", "접촉 없음", "흡입 없음", "노출 없음", "단순 알람", "점검 수준"]):
        false_score += add_false(3, "실제 노출·활성 위험 없음")
    if has_any(text, ["계획된", "유지보수", "예상된 반응", "단순 점검"]) and not has_any(text, ["액체", "누수", "누출", "응축수", "접액", "냄새"]):
        false_score += add_false(2, "실제 물질 발견 없는 계획 유지보수·점검")

    card = finalize_route_card("leak_contact", triggered, true_score, false_score, true_signals, false_signals)
    return apply_profile_to_route_card(card, profile)


def score_standard_deviation_route(text: str, profile: dict | None = None) -> dict:
    true_score = 0
    false_score = 0
    true_signals: list[str] = []
    false_signals: list[str] = []

    route_terms = [
        "지정 도구",
        "특정 도구",
        "지정 작업방법",
        "대체 도구",
        "다른 도구",
        "대체 작업방법",
        "규정위반",
        "안전기준",
        "손들기",
    ]
    triggered = has_any(text, route_terms)

    add_true = score_adder("true", true_signals)
    add_false = score_adder("false", false_signals)

    if has_any(text, ["지정 도구", "특정 도구", "지정 작업방법", "안전기준", "규정", "표준"]):
        true_score += add_true(2, "회사 지정 안전도구·작업방법 기준 존재")
    if has_any(text, ["대체 도구", "다른 도구", "대체 작업방법", "방법 변경", "작업방법 변경"]):
        true_score += add_true(2, "대체 도구·대체 방법 사용 필요")
    if has_any(text, ["공간 협소", "협소", "간섭", "접근 불가", "작업환경", "컨디션"]):
        true_score += add_true(2, "현장 조건 때문에 기준 준수 곤란")
    if has_any(text, ["삼성전자", "ds", "sec", "담당부서", "시공그룹", "허락", "승인", "손들기", "협의"]):
        true_score += add_true(2, "담당부서 허락·승인·협의 신호")

    if has_any(text, ["일정 조율", "작업순서", "일반 허가", "허가서", "담당자 확인", "기간 연장"]):
        false_score += add_false(3, "일반 행정·일정·허가 협의 신호")
    if has_any(text, ["편의", "생산성", "효율", "공구 종류", "단순 방법 논의"]):
        false_score += add_false(3, "안전기준 이탈보다 편의·생산성 목적 가능성")
    if has_any(text, ["협력회사 자체", "자체 판단"]) and not has_any(text, ["삼성전자", "ds", "sec", "담당부서"]):
        false_score += add_false(2, "담당부서 승인보다 협력회사 자체 판단")

    card = finalize_route_card("standard_deviation", triggered, true_score, false_score, true_signals, false_signals)
    return apply_profile_to_route_card(card, profile)


def finalize_route_card(
    route: str,
    triggered: bool,
    true_score: int,
    false_score: int,
    true_signals: list[str],
    false_signals: list[str],
) -> dict:
    margin = true_score - false_score
    if true_score >= 5 and margin >= 2:
        recommendation = JIN
        reason = "진성 신호가 가성 제외조건보다 강함"
    elif false_score >= 5 and margin <= -2:
        recommendation = GA
        reason = "명확한 가성 제외조건이 진성 신호보다 강함"
    else:
        recommendation = REVIEW
        reason = "진성/가성 신호가 혼재하거나 점수 차이가 작음"

    return {
        "route": route,
        "triggered": triggered,
        "true_score": true_score,
        "false_score": false_score,
        "margin": margin,
        "recommendation": recommendation,
        "review_needed": recommendation == REVIEW and triggered,
        "reason": reason,
        "true_signals": true_signals,
        "false_signals": false_signals,
    }


def apply_profile_to_scorecard(scorecard: dict, profile: dict | None) -> dict:
    """Re-score an existing scorecard with a candidate profile.

    This makes route-score evolution cheap: once the expensive LLM judgement
    produced evidence and a base scorecard, we can test many score profiles
    without calling the LLM again.
    """
    if not isinstance(scorecard, dict):
        return {}
    profile = profile or {}
    routes = [
        apply_profile_to_route_card(route, profile)
        for route in scorecard.get("routes", [])
        if isinstance(route, dict)
    ]
    if not routes:
        result = dict(scorecard)
        result["profile_name"] = profile.get("name", "base") if profile else scorecard.get("profile_name", "base")
        return result

    primary = max(routes, key=lambda card: card.get("true_score", 0) + card.get("false_score", 0))
    recommendations = {card.get("recommendation") for card in routes}
    conflicting_routes = len(recommendations - {REVIEW}) > 1
    recommendation = primary.get("recommendation", REVIEW)
    reason = primary.get("reason", "")
    review_needed = bool(primary.get("review_needed", False)) or conflicting_routes
    if conflicting_routes:
        recommendation = REVIEW
        reason = "배관/접액 등 복수 특수 루트의 권고가 충돌함"

    return {
        **scorecard,
        "primary_route": primary.get("route", scorecard.get("primary_route", "general")),
        "recommendation": recommendation,
        "true_score": int(primary.get("true_score", 0)),
        "false_score": int(primary.get("false_score", 0)),
        "margin": int(primary.get("true_score", 0)) - int(primary.get("false_score", 0)),
        "review_needed": review_needed,
        "guardrail_override": should_guardrail_override_to_false(primary),
        "profile_name": profile.get("name", scorecard.get("profile_name", "base")) if profile else scorecard.get("profile_name", "base"),
        "reason": reason,
        "routes": routes,
    }


def apply_profile_to_route_card(route_card: dict, profile: dict | None) -> dict:
    if not profile:
        return route_card

    card = dict(route_card)
    route = str(card.get("route", ""))
    route_profile = (profile.get("routes") or {}).get(route, {})
    if not route_profile:
        card["profile_name"] = profile.get("name", "base")
        return card

    true_score = int(card.get("true_score", 0)) + int(route_profile.get("true_bonus", 0))
    false_score = int(card.get("false_score", 0)) + int(route_profile.get("false_bonus", 0))
    true_signals = list(card.get("true_signals", []))
    false_signals = list(card.get("false_signals", []))

    true_score += apply_signal_bonuses(true_signals, route_profile.get("true_signal_bonuses", []), true_signals)
    false_score += apply_signal_bonuses(false_signals, route_profile.get("false_signal_bonuses", []), false_signals)
    true_score = max(0, true_score)
    false_score = max(0, false_score)

    true_threshold = int(route_profile.get("true_threshold", profile.get("true_threshold", 5)))
    true_margin = int(route_profile.get("true_margin", profile.get("true_margin", 2)))
    false_threshold = int(route_profile.get("false_threshold", profile.get("false_threshold", 5)))
    false_margin = int(route_profile.get("false_margin", profile.get("false_margin", 2)))

    margin = true_score - false_score
    if true_score >= true_threshold and margin >= true_margin:
        recommendation = JIN
        reason = "프로필 적용 후 진성 신호가 가성 제외조건보다 강함"
    elif false_score >= false_threshold and margin <= -false_margin:
        recommendation = GA
        reason = "프로필 적용 후 명확한 가성 제외조건이 진성 신호보다 강함"
    else:
        recommendation = REVIEW
        reason = "프로필 적용 후 진성/가성 신호가 혼재하거나 점수 차이가 작음"

    card.update(
        {
            "true_score": true_score,
            "false_score": false_score,
            "margin": margin,
            "recommendation": recommendation,
            "review_needed": recommendation == REVIEW and bool(card.get("triggered", False)),
            "reason": reason,
            "true_signals": true_signals,
            "false_signals": false_signals,
            "profile_name": profile.get("name", "base"),
            "guardrail_min_false_score": int(
                route_profile.get("guardrail_min_false_score", profile.get("guardrail_min_false_score", 6))
            ),
            "guardrail_min_margin": int(route_profile.get("guardrail_min_margin", profile.get("guardrail_min_margin", 3))),
        }
    )
    return card


def apply_signal_bonuses(source_signals: list[str], bonuses: list[dict], output_signals: list[str]) -> int:
    total = 0
    text = "\n".join(source_signals).lower()
    for bonus in bonuses or []:
        terms = bonus.get("contains", [])
        if isinstance(terms, str):
            terms = [terms]
        if not terms:
            continue
        if any(str(term).lower() in text for term in terms):
            score = int(bonus.get("score", 0))
            if score:
                total += score
                output_signals.append(f"{score:+d} profile:{bonus.get('reason', 'signal bonus')}")
    return total


def should_guardrail_override_to_false(route_card: dict) -> bool:
    if route_card.get("recommendation") != GA:
        return False
    true_score = int(route_card.get("true_score", 0))
    false_score = int(route_card.get("false_score", 0))
    false_signals = " ".join(route_card.get("false_signals", []))
    has_hard_guardrail = has_any(
        false_signals.lower(),
        [
            "계획",
            "허가",
            "일정",
            "무해성",
            "테스트",
            "실제 노출",
            "단순 청소",
            "단순 시정",
        ],
    )
    min_false_score = int(route_card.get("guardrail_min_false_score", 6))
    min_margin = int(route_card.get("guardrail_min_margin", 3))
    return has_hard_guardrail and false_score >= min_false_score and false_score >= true_score + min_margin


def score_adder(prefix: str, signals: list[str]):
    def add(score: int, signal: str) -> int:
        signals.append(f"+{score} {signal}")
        return score

    return add


def has_any(text: str, terms: list[str]) -> bool:
    return any(term.lower() in text for term in terms)


def flatten_values(value: Any) -> str:
    parts: list[str] = []
    collect_values(value, parts)
    return "\n".join(parts)


def collect_values(value: Any, parts: list[str]) -> None:
    if value is None:
        return
    if isinstance(value, str):
        if value.strip():
            parts.append(value.strip())
        return
    if isinstance(value, (int, float, bool)):
        parts.append(str(value))
        return
    if isinstance(value, dict):
        for item in value.values():
            collect_values(item, parts)
        return
    if isinstance(value, (list, tuple, set)):
        for item in value:
            collect_values(item, parts)
        return
    try:
        parts.append(json.dumps(value, ensure_ascii=False))
    except Exception:
        parts.append(str(value))
