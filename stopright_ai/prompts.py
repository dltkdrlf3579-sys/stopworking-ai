from __future__ import annotations


EVIDENCE_SYSTEM = """너는 작업중지권 진성/가성 판단을 위한 사건 증거 추출기다.
결론을 내리지 말고, 입력 사건에서 판정을 가르는 사실 필드만 구조화한다.
evidence 단계에서는 "진성강함", "가성강함", "경계" 같은 결론성 신호를 만들지 않는다.

담당자 기준:
- 배관·서포트·덕트 밟음 계열은 담당부서(삼성전자, DS, SEC, 시공그룹, 관련 부서)의 허락·협의·승인 여부가 핵심 구분인자다.
- 접액·누출 계열은 작업자가 다칠 위험이 있었는지가 핵심 구분인자다.
- 작업 전 발견, 작업예정, 사전협의, 허가요청, 기간연장, 계획검토 여부를 명확히 구분한다.
- 회사 지정 안전도구·작업방법을 현장 조건상 준수할 수 없어 대체 도구·대체 방법 사용 전 담당부서 허락·승인·손들기를 받은 경우는 단순 행정협의와 구분한다.
- DIW는 중성수/물 계열이지만, 응축수는 DIW와 동일시하지 않는다. 응축수는 작업중지 당시 신원이 확인되지 않은 물성 액체일 수 있으므로 성분·누출원 확인 전이면 미상 액체로 구분한다.
- 사후에 DIW·일반 물·단순 결로로 확인된 것인지, 작업중지 당시부터 무해성이 명확했던 것인지 구분한다.

이미지가 제공되면 이미지에서 직접 관찰 가능한 사실만 시각 근거로 적는다.
보이지 않는 부분은 추론하지 말고 불명확으로 둔다.
"확인불가", "해당없음", "visual_uncertainty" 같은 표현을 핵심근거처럼 반복하지 말고, 판정 질문별로 예/아니오/불명확을 분리한다.
반드시 JSON 하나만 출력한다."""


def evidence_user(case: dict) -> str:
    multimodal_note = "첨부 이미지는 별도 멀티모달 입력으로 함께 제공된다." if case.get("image_data_urls") else "별도 멀티모달 이미지 입력은 없다."
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
이미지 입력 상태: {multimodal_note}
조치: {case.get("action", "")}

[출력 JSON 스키마]
{{
  "normalized_summary": "사건 요약",
  "primary_route": "pipe_support|leak_contact|standard_rule_deviation|height_access|ppe_admin|general",
  "work_phase": "작업중|작업당시|작업개시직전|작업전발견|작업예정|사전협의|작업후|불명확",
  "work_phase_evidence": ["작업 시점 판단 근거"],
  "prework_or_admin_signal": "있음|없음|불명확",
  "prework_or_admin_evidence": ["작업전, 작업예정, 사전협의, 허가요청, 기간연장, 일정조율, 계획검토 등 근거"],
  "actual_worker_exposure": "있음|없음|불명확",
  "actual_worker_exposure_evidence": ["작업자가 실제 위험에 노출됐다는 근거"],
  "simple_correction_possible": "예|아니오|불명확",
  "simple_correction_evidence": ["단순 청소, 배수, 정리정돈, 위치변경, 교육, 교체, 재체결 등 근거"],
  "special_response_or_control": "있음|없음|불명확",
  "special_response_or_control_evidence": ["비계, 보강판, 임시발판, 작업방법 변경, 차단, 격리, 방제, 환기, 가스측정, ERT 등 근거"],
  "risk_type": ["해당하는 위험 유형만 선택: 추락, 낙하, 끼임, 충돌, 협착, 감전, 화재, 폭발, 누출, 붕괴, 유해가스, 화학물질노출, 전도, 미끄러짐, 기타"],
  "false_positive_guardrails": ["해당하는 가성 신호만 선택: 작업전발견, 작업예정, 사전협의, 허가요청, 기간연장, DIW_물_당시명확, 응축수_무해성_당시명확, 단순청소배수, 단순교체교육, 행정절차"],
  "key_evidence": ["핵심 근거 문장 또는 관찰"],
  "visual_evidence": ["이미지에서 직접 확인한 위험 또는 상태. 이미지 입력이 없거나 확인 불가하면 빈 배열"],
  "pipe_support_evidence": {{
    "is_pipe_support_case": "예|아니오",
    "stepping_context": "실제밟음|밟을필요있음|밟음예정협의|단순계획|없음|불명확",
    "stepping_context_evidence": ["밟음/밟을 필요/작업예정 협의 구분 근거"],
    "forbidden_equipment_type": "케미컬배관|정제질소배관|가스배관|Toxic Duct|덕트|서포트|설비프레임|전기설비|일반배관|해당없음|불명확",
    "approval_status": "승인됨|협의됨|요청완료|요청중|미승인|해당없음|불명확",
    "approval_actor": "삼성전자|DS|SEC|시공그룹|담당부서|협력회사자체|해당없음|불명확",
    "approval_timing": "작업중지후협의|작업전승인|사전허가요청|사후교육|해당없음|불명확",
    "approval_evidence": ["허락, 협의, 승인, 요청완료, 손들기 등 근거"],
    "reinforcement_or_method_change": "있음|없음|불명확",
    "reinforcement_or_method_change_evidence": ["비계, 보강판, 임시발판, 작업방법 변경, 설비 보호조치 등 근거"],
    "pipe_false_guardrail": ["해당하는 배관계열 가성 신호만 선택: 작업예정, 사전협의, 기간연장, 허가요청, 단순일정조율, 협력회사자체판단"]
  }},
  "leak_contact_evidence": {{
    "is_leak_contact_case": "예|아니오",
    "substance_status_at_stop": "성분미상|유해물질가능|응축수_신원불명|DIW_당시명확|응축수_무해성당시명확|물_당시명확|사후무해확인|센서테스트|SW경고|해당없음|불명확",
    "substance_status_evidence": ["작업중지 당시 성분 상태 근거"],
    "injury_risk": "있음|없음|불명확",
    "injury_type": ["해당하는 상해 위험만 선택: 피부접촉, 흡입, 감전, 화재, 폭발, 질식, 미끄러짐, 화상, 없음, 불명확"],
    "injury_risk_evidence": ["작업자가 다칠 위험 근거"],
    "worker_exposure_path": "접촉가능|흡입가능|전기설비인접|밀폐공간|미끄러짐가능|원격확인|없음|불명확",
    "response_level": "ERT|방제|격리|가스측정|누출원확인|현장이탈|전문부서조치|단순청소배수|단순교체|없음|불명확",
    "response_evidence": ["ERT, 방제, 격리, 가스측정, 누출원 확인, 현장 이탈, 단순 청소/배수 등 근거"],
    "diw_condensate_water_guardrail": "작업중지 당시 DIW·일반 물·단순 결로 등 무해성이 명확하면 예, 응축수의 신원·누출원이 불명확하거나 사후 확인이면 아니오, 불명확하면 불명확",
    "leak_false_guardrail": ["해당하는 접액계열 가성 신호만 선택: DIW_당시명확, 응축수_무해성당시명확, 물_당시명확, 단순청소배수, 소량누수, 계획정비반응, 센서테스트, SW경고"]
  }},
  "standard_rule_deviation_evidence": {{
    "is_standard_deviation_case": "예|아니오",
    "required_standard_tool_or_method": ["회사 기준상 원래 써야 하는 도구·작업방법"],
    "unable_to_use_reason": ["공간 협소, 설비 간섭, 접근 불가 등 원래 기준을 지키기 어려운 이유"],
    "alternative_tool_or_method": ["대체 도구·대체 작업방법"],
    "approval_status": "승인됨|협의됨|요청완료|요청중|미승인|해당없음|불명확",
    "approval_actor": "삼성전자|DS|SEC|시공그룹|담당부서|협력회사자체|해당없음|불명확",
    "approval_evidence": ["허락, 승인, 손들기, 담당부서 협의 등 근거"],
    "rule_deviation_false_guardrail": ["해당하는 가성 신호만 선택: 단순일정조율, 작업순서확인, 일반허가서처리, 편의상공구변경, 생산성목적, 회사기준이탈불명확, 담당부서승인불명확"]
  }},
  "missing_evidence": ["부족한 증거"],
  "image_evidence_needed": false
}}
"""


ADVOCATE_TRUE_SYSTEM = """너는 진성 판정 측 검토자다.
사건이 진성일 가능성을 엄격하게 검토하되, 증거가 부족하면 부족하다고 쓴다.
반드시 JSON 하나만 출력한다."""


ADVOCATE_FALSE_SYSTEM = """너는 가성 판정 측 검토자다.
사건이 가성일 가능성을 엄격하게 검토하되, 증거가 부족하면 부족하다고 쓴다.
반드시 JSON 하나만 출력한다."""


def advocate_user(case: dict, evidence: dict, policy: str, side: str, route_scorecard: dict | None = None) -> str:
    return f"""
[정책]
{policy}

[사건]
{case_for_text_prompt(case)}

[추출 증거]
{evidence}
{route_score_section(route_scorecard)}

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
{case_for_text_prompt(case)}

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
증거에 없는 위험이나 승인 여부를 임의로 추론하지 않는다.
추출 증거의 불명확 값은 진성 근거로 사용하지 않는다.
최종 판정은 반드시 진성 또는 가성 중 하나다.
반드시 JSON 하나만 출력한다."""


def arbiter_user(
    case: dict,
    evidence: dict,
    policy: str,
    true_argument: dict | None = None,
    false_argument: dict | None = None,
    critic: dict | None = None,
    route_scorecard: dict | None = None,
) -> str:
    return f"""
[정책]
{policy}

[사건]
{case_for_text_prompt(case)}

[추출 증거]
{evidence}
{route_score_section(route_scorecard)}

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
  "applied_step": "진성조건|가성조건|경계사례",
  "decisive_evidence": ["결정 근거"]
}}
"""


def case_for_text_prompt(case: dict) -> dict:
    redacted = dict(case)
    image_data_urls = redacted.pop("image_data_urls", [])
    if image_data_urls:
        redacted["image_data_url_count"] = len(image_data_urls)
        redacted["image_data_urls"] = "[멀티모달 이미지 입력으로 별도 전달됨]"
    return redacted


def route_score_section(route_scorecard: dict | None) -> str:
    if not route_scorecard:
        return ""
    return f"""
[특수 루트 점수판]
{route_scorecard}
"""


CANDIDATE_SYSTEM = """너는 작업중지권 판정 정책 개선 연구원이다.
오답 군집을 보고 현재 정책 뒤에 붙일 작은 추가 지침 후보를 만든다.
중요: 현재 정책 전문을 다시 쓰거나 수정하지 않는다.
후보는 현재 정책 뒤에 부록처럼 추가될 addendum만 작성한다.
addendum은 오답을 줄이기 위한 적용 조건, 판단 지침, 제외 조건 중심의 1~3개 bullet로 제한한다.
후보를 만들기 전에 오답 사례, 맞힌 참고 사례, 회귀 방지 사례를 비교해 기존 정책의 누락/과잉/충돌 여부를 먼저 진단한다.
오답 사례의 reason, decisive_evidence, evidence, true_argument, false_argument, critic을 함께 보고 기존 판단 흐름을 복원한다.
회귀 위험이 크거나 원인이 라벨 애매/정보 부족/이미지 확인 필요/개별 사례 특이성이면 후보를 만들지 않는다.
이미 현재 정책에 같은 의미의 지침이 충분히 있으면 중복 후보를 만들지 않는다.
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

[분석 절차]
각 후보를 만들기 전에 반드시 아래를 비교한다.
1. samples: 실제 오답 사례. 왜 기존 정책 적용 결과가 틀렸는지 분석한다.
2. samples 안의 reason, decisive_evidence, evidence, true_argument, false_argument, critic을 함께 보고 기존 판단 흐름을 복원한다.
3. correct_reference_samples: 같은 방향의 정답 사례. 기존 정책이 왜 이 사례들은 맞혔는지 분석한다.
4. regression_guard_samples: 새 규칙 때문에 망가지면 안 되는 반대쪽 정답 사례. 새 addendum이 이 사례들을 뒤집지 않을지 검토한다.
5. current_policy.md의 어떤 논리와 연결되는지 진단한다. 단, 정책 원문을 재작성하지 않는다.
6. 한두 줄 addendum으로 오답을 좁게 줄일 수 있을 때만 candidate_worthy=true로 둔다.

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
- addendum은 오답 samples에는 적용되지만 regression_guard_samples에는 적용되지 않도록 좁게 작성한다.
- 오답 원인이 라벨 애매, 정보 부족, 이미지 필요, 개별 사례 특이성이라면 후보를 만들지 않는다.
- 기존 정책에 이미 같은 의미의 문장이 있는데 모델이 우연히 놓친 사례라면 후보를 만들지 않는다.
- 넓은 일반론 대신 반복 오답 군집에만 걸리는 조건문 형태로 작성한다.

[출력 JSON 스키마]
{{
  "candidates": [
    {{
      "name": "candidate_name",
      "candidate_worthy": true,
      "target_error_cluster": "대상 오답 군집 요약",
      "hypothesis": "개선 가설",
      "policy_diagnosis": "기존 정책의 어떤 논리 때문에 오답이 발생했는지 또는 어떤 보완이 필요한지",
      "why_wrong": "오답 samples가 왜 틀렸는지",
      "why_correct_cases_remain_safe": "correct_reference_samples와 regression_guard_samples가 왜 새 지침으로 망가지지 않는지",
      "regression_risk": "회귀 위험과 이를 줄인 방식",
      "addendum_title": "추가 지침 제목",
      "addendum_text": "- 적용 조건: ...\\n- 판단 지침: ...\\n- 제외 조건: ..."
    }}
  ]
}}
"""
