CREATE TABLE IF NOT EXISTS {history_table} (
    run_id TEXT,
    source_date DATE,
    judged_at TIMESTAMP,
    application_no TEXT,
    title TEXT,
    major_category TEXT,
    middle_category TEXT,
    ai_pred TEXT,
    confidence INTEGER,
    reason TEXT,
    applied_step TEXT,
    decisive_evidence TEXT,
    review_needed BOOLEAN,
    error_yn BOOLEAN,
    error_message TEXT,
    model_name TEXT,
    policy_version TEXT,
    judge_mode TEXT,
    created_at TIMESTAMP
)
DISTRIBUTED BY (application_no);

DROP VIEW IF EXISTS {latest_view};

CREATE VIEW {latest_view} AS
SELECT
    run_id,
    source_date,
    judged_at,
    application_no,
    title,
    major_category,
    middle_category,
    ai_pred,
    confidence,
    reason,
    applied_step,
    decisive_evidence,
    review_needed,
    error_yn,
    error_message,
    model_name,
    policy_version,
    judge_mode,
    created_at
FROM (
    SELECT
        h.*,
        ROW_NUMBER() OVER (
            PARTITION BY h.source_date, h.application_no
            ORDER BY h.judged_at DESC, h.created_at DESC, h.run_id DESC
        ) AS rn
    FROM {history_table} h
) x
WHERE rn = 1;
