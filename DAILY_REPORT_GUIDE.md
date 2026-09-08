# 업체별 작업중지권 일일 보고서

운영 PC의 사내 DB와 기존 AI API를 이용한다. 메일 발송은 이 프로그램에서 수행하지 않으며,
사용자 메일 코드가 결과의 제목과 HTML 본문을 받아 발송하면 된다.

## 준비

1. `queries/daily_company_report.sql`에 일별 조회 쿼리를 넣는다.
2. `daily_report.ini`의 컬럼명만 실제 쿼리와 맞춘다. 기본 업체 컬럼은 `업체명`이다.
3. API와 DB 연결은 기존 `config.ini`, `sql_adapter.py`를 그대로 사용한다.

필수 조회 컬럼: `출원번호`, `업체명`, `등록일`, `AI판정`.
내용 분석을 위해 `제목`, `대분류`, `현상`, `조치`, `판단근거`도 포함하는 것을 권장한다.
오류 판정이 섞이면 `에러여부`도 조회한다. 판정값은 `진성`/`가성`을 사용한다.
그 외 값, NULL, 에러여부=true는 보류/미판정으로 집계한다.
`NEW_대분류`를 집계하려면 `daily_report.ini`의 `major = NEW_대분류`로 설정한다.

업체명 등 원본 데이터와 기존 AI 판정 결과를 출원번호로 조인하여 조회한다.
같은 출원번호는 최신 결과 한 행을 권장하며, 중복이 있으면 `판정일시` 기준으로 최신 행을 선택한다.
최신 순서를 결정할 수 없으면 보고서를 생성하지 않고 오류를 알린다.

## 실행

```powershell
python run_daily_company_report.py --target-date 2026-09-08
```

날짜를 생략하면 한국 시각 기준 전날을 사용한다.
SQL의 `{target_date}`, `{next_date}`는 자동 치환하며, 조회 후에도 등록일을 확인해 해당 날짜만 집계한다.
따라서 등록일 대신 다른 업무 기준일을 쓰려면 `daily_report.ini`의 `date` 컬럼을 바꾼다.

```powershell
python run_daily_company_report.py
```

DB/API 연결 없이 보고서 모양만 확인:

```powershell
python run_daily_company_report.py --demo --target-date 2026-09-08
```

`--demo`는 가상 데이터와 고정된 예시 문장으로 작성하며 화면에 시연 자료임을 표시한다.

## 프롬프트에 전달하는 데이터

1. 코드에서 전체 건수, 업체별·분류별 진성/가성/보류 건수를 전수 집계한다.
2. 각 업체의 게시물을 묶어 출원번호, 업체명, 기존 AI 판정, 제목, 현상, 조치, 판단근거를 AI에 전달한다.
3. 업체 자료가 길면 여러 호출로 나눈다. 각 행은 한 분할 입력에 포함된다.
4. 업체별 관찰과 전체 집계를 종합해 최종 보고서를 작성한다.

`max_case_chars`는 제목·현상·조치·판단근거 각각의 길이 제한이다(기본 800자).
`max_batch_chars`는 한 분할 입력의 문자 수 제한이다(기본 24,000자, 토큰 수가 아님).
첨부 이미지와 base64는 본문에서 제거한다. 이번 보고서는 기존 텍스트와 AI 판단근거를 요약하며,
이미지 재판정이나 진성·가성 재분류를 하지 않는다. AI 관찰은 주요 사례 요약이며 새로운 전수 유형 집계가 아니다.

대략 API 호출 수는 업체별 분할 수의 합 + 최종 종합 1회이다. 기본 4개 병렬 실행이며,
호출 제한·재시도는 기존 `config.ini`의 운영 설정을 따른다. 별도 프로세스 간 호출 제한은 공유되지 않는다.

## 결과와 메일 연동

`outputs/daily_reports/날짜_실행시각/`에 저장한다.

- `report.html`: 브라우저로 확인하는 메일용 보고서
- `mail_content.json`: `subject`, `html_body`, `text_summary`, `demo`
- `summary.json`: 코드로 계산한 전체·업체별·등록유형별 통계 및 중복/날짜 필터 내역
- `partial_analysis.json`: 업체별 분할 분석 결과, 완료되는 대로 저장
- `analysis.json`: AI 최종 보고서 서술
- `error.json`: 분석 실패 시 오류 내역. 실패한 분석을 정상 보고서로 발송하지 않는다.

메일 코드에서 다음 값을 사용한다.

```python
import json
from stopright_ai.daily_company_report import run_report

report_path = run_report()  # 전날 데이터, 생성 실패 시 예외 발생
content = json.loads((report_path.parent / "mail_content.json").read_text(encoding="utf-8"))
subject = content["subject"]
html_body = content["html_body"]
# 사용자의 사내 메일 함수에 subject와 html_body를 전달한다.
```

HTML은 숫자와 업체별 표를 코드로 생성하고 AI 문장은 이스케이프하여 삽입한다.
최종 보고서의 출원번호와 업체 일치 여부도 검사한다. AI의 문장 해석 자체의 정확성까지 보장하는 검증은 아니다.
일일 정기 실행은 사용자의 메일 발송 스크립트를 기존 Windows 작업 스케줄러에 등록하면 된다.
