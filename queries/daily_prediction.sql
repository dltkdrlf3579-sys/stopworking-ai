-- 운영 예측용 쿼리입니다.
-- 평가용 main.sql에서 정답_판단결과 조인만 제거한 형태로 넣으면 됩니다.
-- {target_date}는 run_daily_prediction.py가 YYYY-MM-DD 형식으로 치환합니다.
--
-- 예시:
-- SELECT
--   t1.*
-- FROM your_source_table t1
-- WHERE t1.기준일자 = '{target_date}'
;
