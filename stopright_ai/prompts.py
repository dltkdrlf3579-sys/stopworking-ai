from __future__ import annotations


EVIDENCE_SYSTEM = """너는 작업중지권 진성/가성 판단을 위한 사건 증거 추출기다.
결론을 먼저 내리지 말고, 입력 사건에서 판정에 필요한 증거만 구조화한다.
이미지가 제공되면 이미지에서 직접 관찰 가능한 위험, 설비 상태, 작업 상태를 확인해 시각 근거로 분리한다.
특히 배관, 서포트, 발판, 사다리, 비계, 난간, 개구부, 공간 협소, 동선 간섭 사례는 안전한 발판·이동경로 유무와 밟으면 안 되는 설비 접촉 여부를 자세히 관찰한다.
누출, 접촉, 접액, 흡입, 냄새, 미상 액체, 응축수, DIW, 약품 흔적, 가스, 연기 사례는 작업중지 당시 성분·위험성·누출원 확인 여부와 작업자 노출 가능성을 자세히 관찰한다.
이미지에서 보이는 사실과 텍스트에만 있는 사실을 구분하고, 보이지 않는 부분은 확인불가로 둔다.
이미지에 보이지 않는 위험은 이미지 근거처럼 쓰지 않는다.
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
  "cluster": "forced_pipe_or_support_stepping|leak_or_contact_uncertainty|height_or_access_fall|none",
  "work_timing": "작업전|작업중|작업후|불명",
  "physical_risk": "있음|없음|불명",
  "actual_physical_hazard": true,
  "actual_physical_hazard_evidence": ["직접 확인된 물리적 위험 근거"],
  "simple_admin_or_precheck": false,
  "simple_admin_or_precheck_evidence": ["계획/협의/허가/일정/일반점검 근거"],
  "simple_correction_possible": false,
  "simple_correction_evidence": ["단순 청소, 배수, 정리정돈, 위치변경, 교육 등 즉시 시정 근거"],
  "requires_physical_or_process_control": false,
  "control_evidence": ["비계, 보강판, 임시발판, 차단, 격리, 방제, 환기, 가스측정, ERT 등 근거"],
  "risk_type": ["추락", "낙하", "끼임", "충돌", "협착", "감전", "화재", "폭발", "누출", "붕괴", "유해가스", "화학물질노출", "전도", "기타"],
  "unexpected_emergency": "있음|없음|불명",
  "imminent_severe_accident": "있음|없음|불명",
  "controlled_by_standard_rules": "가능|불가능|불명",
  "worker_initiated_stop": "있음|없음|불명",
  "false_positive_signals": ["작업전점검", "일상수칙위반", "행정절차", "계획정비", "자체시정", "긴급성부족"],
  "key_evidence": ["핵심 근거 문장 또는 관찰"],
  "visual_evidence": ["이미지에서 직접 확인한 위험 또는 상태. 이미지 입력이 없거나 확인 불가하면 빈 배열"],
  "pipe_support_evidence": {{
    "stepping_required_on_forbidden_equipment": false,
    "actual_stepping_observed": false,
    "forbidden_equipment_type": "케미컬라인|Toxic Duct|가동설비|배관|서포트|덕트|전기설비|해당없음|불명확",
    "access_reinforcement_needed": false,
    "access_reinforcement_evidence": ["발판 부족, 비계, 보강판, 커버, 임시발판, 작업방법 변경, 위험 제거 협의 근거"],
    "safe_foothold_or_path": "안전한 발판·이동경로가 보이는지. 확인불가면 확인불가",
    "pipe_or_support_stepping": "배관·서포트·설비 프레임을 밟거나 밟아야 하는 정황이 보이는지. 확인불가면 확인불가",
    "work_height_or_fall_context": "고소·개구부·난간·사다리·비계·추락 관련 시각 정황",
    "interference_or_contact_context": "공간 협소, 동선 간섭, 끼임·충돌·접촉·설비 파손 가능 시각 정황",
    "needed_physical_controls": ["이미지나 텍스트상 필요해 보이는 물리적 조치: 임시발판, 비계, 난간, 덮개, 작업방법 변경, 부서 협의 등"],
    "visual_uncertainty": ["이미지만으로 확인 불가하거나 추가 확인이 필요한 사항"]
  }},
  "leak_contact_evidence": {{
    "leak_signal_type": "실제누출|미상액체|냄새|가스|연기|분진|약품흔적|센서알람|SW경고|해당없음|불명확",
    "leak_material_status_at_stop": "미상|DIW|응축수|물|화학물질|가스|센서테스트|SW경고|해당없음|불명확",
    "post_harmless_confirmation": false,
    "active_release_or_physical_event": false,
    "active_release_or_physical_event_evidence": ["분출, 활성누출, 스파크, 감전충격, 유해가스경보 등 근거"],
    "worker_exposure_path": "접촉|흡입|전기설비인접|밀폐공간|원격/개방|해당없음|불명확",
    "emergency_or_special_response": false,
    "emergency_or_special_response_evidence": ["ERT, 현장이탈, 전문방제, 격리, 환기, 가스측정 등 근거"],
    "benign_alarm_or_harmless_leak_guardrail": false,
    "benign_alarm_or_harmless_leak_evidence": ["센서테스트, SW경고, DIW, 응축수, 물, 단순결로, 청소/배수 등 근거"],
    "observed_material_or_signal": "액체, 가스, 냄새, 연기, 분진, 약품 흔적, 누출 흔적, 고임, 젖음, 변색 등 관찰된 물질 또는 신호",
    "material_identity_at_stop_time": "작업중지 당시 성분이 미상인지, DIW·응축수·물 등 무해성이 이미 확인됐는지, 사후 확인인지 구분",
    "source_or_boundary": "누출원, 배관, 밸브, 펌프, 탱크, 장비 하부, 차단·격리 경계가 확인되는지",
    "worker_exposure_path": "작업자가 접촉·흡입·비산·미끄러짐·화학물질 노출 위치에 있었는지 또는 접근 예정인지",
    "immediate_risk_context": "화재, 폭발, 질식, 감전, 접액, 흡입, 미끄러짐, 설비 파손에 따른 추가 누출 등 즉시 위험 정황",
    "needed_confirmation_or_controls": ["성분 확인, 누출원 확인, 차단, 격리, 방제, 환기, 가스측정, 압력 확인, 관련 부서 확인 등 필요한 조치"],
    "harmless_or_minor_signals": ["작업중지 당시 이미 무해성이 확인됐거나 단순 청소·닦음·배수로 통제 가능한 근거"],
    "visual_uncertainty": ["이미지만으로 확인 불가하거나 추가 확인이 필요한 사항"]
  }},
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
특수 루트 점수판이 제공되면 배관·서포트·발판·이동경로 및 누출·접액·미상액체 사례에서 진성/가성 신호의 균형을 확인한다.
점수판의 recommendation이 "가성"이고 false_score가 true_score보다 충분히 높으면, 단어만 보고 진성으로 끌어올리는 오류를 경계한다.
점수판의 recommendation이 "경계"이면 진성/가성 신호가 섞인 것이므로 decisive_evidence에 어떤 신호를 더 중요하게 보았는지 적는다.
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
  "applied_step": "Step 1|Step 2|Step 3|Step 4",
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
