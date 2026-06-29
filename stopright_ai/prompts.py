from __future__ import annotations


EVIDENCE_SYSTEM = """너는 작업중지권 진성/가성 판단을 위한 사건 증거 추출기다.
결론을 먼저 내리지 말고, 입력 사건에서 판정에 필요한 증거만 구조화한다.
반드시 JSON 하나만 출력한다."""


def evidence_user(case: dict) -> str:
    return f"""
아래 작업중지권 사건에서 판정 근거를 추출하라.

[사건]
출원번호: {case.get("id", "")}
제목: {case.get("title", "")}
대분류: {case.get("major", "")}
중분류: {case.get("middle", "")}
현상 텍스트: {case.get("phenomenon_text", "")}
사용 이미지 개수: {case.get("image_count", 0)}
원본 이미지 개수: {case.get("original_image_count", case.get("image_count", 0))}
생략 이미지 개수: {case.get("omitted_image_count", 0)}
현상 사용 이미지 개수: {case.get("phenomenon_image_count", 0)}
조치 사용 이미지 개수: {case.get("action_image_count", 0)}
이미지 파일: {case.get("image_paths", [])}
조치: {case.get("action", "")}

[출력 JSON 스키마]
{{
  "normalized_summary": "사건 요약",
  "work_timing": "작업전|작업중|작업후|불명",
  "physical_risk": "있음|없음|불명",
  "risk_type": ["낙하", "협착", "감전", "화재", "폭발", "붕괴", "유해가스", "전도", "기타"],
  "unexpected_emergency": "있음|없음|불명",
  "imminent_severe_accident": "있음|없음|불명",
  "controlled_by_standard_rules": "가능|불가능|불명",
  "worker_initiated_stop": "있음|없음|불명",
  "false_positive_signals": ["작업전점검", "일상수칙위반", "행정절차", "계획정비", "자체시정", "긴급성부족"],
  "key_evidence": ["핵심 근거 문장 또는 관찰"],
  "missing_evidence": ["부족한 증거"],
  "image_evidence_needed": true
}}
"""


ADVOCATE_TRUE_SYSTEM = """너는 진성 판정 측 검토자다.
사건이 진성일 가능성을 엄격하게 검토하되, 증거가 부족하면 부족하다고 쓴다.
반드시 JSON 하나만 출력한다."""


ADVOCATE_FALSE_SYSTEM = """너는 가성 판정 측 검토자다.
사건이 가성일 가능성을 엄격하게 검토하되, 증거가 부족하면 부족하다고 쓴다.
반드시 JSON 하나만 출력한다."""


def advocate_user(case: dict, evidence: dict, policy: str, side: str) -> str:
    return f"""
[정책]
{policy}

[사건]
{case}

[추출 증거]
{evidence}

{side} 관점에서 판단 근거를 작성하라.

[출력 JSON 스키마]
{{
  "position": "{side}",
  "strength": 0,
  "reasons": ["근거"],
  "weaknesses": ["반대 근거 또는 불확실성"],
  "critical_evidence": ["가장 중요한 증거"]
}}
"""


CRITIC_SYSTEM = """너는 작업중지권 판정의 모순 검토자다.
진성 측 주장과 가성 측 주장이 충돌하는 부분을 찾아 최종 판정자가 볼 수 있게 정리한다.
반드시 JSON 하나만 출력한다."""


def critic_user(case: dict, evidence: dict, true_argument: dict, false_argument: dict) -> str:
    return f"""
[사건]
{case}

[추출 증거]
{evidence}

[진성 측 주장]
{true_argument}

[가성 측 주장]
{false_argument}

[출력 JSON 스키마]
{{
  "contradictions": ["충돌 지점"],
  "decisive_questions": ["판정을 가르는 질문"],
  "likely_failure_modes": ["모델이 오판하기 쉬운 이유"]
}}
"""


ARBITER_SYSTEM = """너는 작업중지권 최종 판정자다.
정책과 추출 증거를 우선하고, 주장 문구보다 실제 증거를 더 신뢰한다.
최종 판정은 반드시 진성 또는 가성 중 하나다.
반드시 JSON 하나만 출력한다."""


def arbiter_user(case: dict, evidence: dict, policy: str, true_argument: dict | None = None, false_argument: dict | None = None, critic: dict | None = None) -> str:
    return f"""
[정책]
{policy}

[사건]
{case}

[추출 증거]
{evidence}

[진성 측 주장]
{true_argument or {}}

[가성 측 주장]
{false_argument or {}}

[모순 검토]
{critic or {}}

[출력 JSON 스키마]
{{
  "판정": "진성 또는 가성",
  "판단근거": "적용된 Step번호와 핵심 사유를 간결히 서술",
  "확신도": 0,
  "review_needed": false,
  "applied_step": "Step 1|Step 2|Step 3|Step 4",
  "decisive_evidence": ["결정 근거"]
}}
"""


CANDIDATE_SYSTEM = """너는 작업중지권 판정 정책 개선 연구원이다.
오답 군집을 보고 현재 정책 뒤에 붙일 작은 추가 지침 후보를 만든다.
중요: 현재 정책 전문을 다시 쓰거나 수정하지 않는다.
후보는 현재 정책 뒤에 부록처럼 추가될 addendum만 작성한다.
addendum은 오답을 줄이기 위한 적용 조건, 판단 지침, 제외 조건 중심의 1~3개 bullet로 제한한다.
출력 JSON 형식, 핵심 정의, 입력 데이터 설명을 직접 재작성하지 않는다.
반드시 JSON 하나만 출력한다."""


def candidate_user(current_policy: str, error_clusters: list[dict], candidate_count: int) -> str:
    return f"""
[현재 정책]
{current_policy}

[오답 군집]
{error_clusters}

현재 정책 뒤에 붙일 후보 추가 지침 {candidate_count}개를 생성하라.
각 후보는 왜 개선될지 짧게 설명하고, addendum_text만 작성한다.

[후보 추가 지침 작성 제한]
- 현재 정책 전문을 출력하지 않는다.
- addendum_text는 현재 정책 뒤에 붙일 추가 지침만 포함한다.
- addendum_text는 1~3개 bullet로 제한한다.
- 가능하면 "적용 조건", "판단 지침", "제외 조건"이 드러나게 쓴다.
- 기존 정책 적용 시 반복된 오답 군집을 보완하기 위한 추가 판단 기준으로 작성한다.
- 기존 정책의 출력 JSON 형식, 필수 필드명, 입력 데이터 형식은 언급하지 않는다.
- 기존 정책을 삭제/교체/재배열하라는 지시는 쓰지 않는다.
- 후보가 {candidate_count}개 이상이면 서로 다른 개선 방향을 가져야 한다.
- 가능하면 후보별 관점을 분리한다: FN 감소, FP 감소, 경계사례 분리.

[출력 JSON 스키마]
{{
  "candidates": [
    {{
      "name": "candidate_name",
      "hypothesis": "개선 가설",
      "addendum_title": "추가 지침 제목",
      "addendum_text": "- 적용 조건: ...\\n- 판단 지침: ...\\n- 제외 조건: ..."
    }}
  ]
}}
"""
