from __future__ import annotations

import json
import time
import uuid
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from .case_prepare import row_to_case
from .config import load_config
from .db_adapter import load_execute_dml, load_execute_sql
from .judge import judge_case
from .llm_factory import create_llm
from .llm_json import configure_llm_runtime
from .logging import install_timestamped_print
from .policy import load_policy


def run_daily_prediction(config_path: str = "config.ini", target_date: str | None = None, llm: Any | None = None) -> dict:
    install_timestamped_print()
    config = load_config(config_path)
    configure_llm_runtime_from_config(config)

    if llm is None:
        llm = create_llm(config)

    run_id = datetime.now().strftime("daily_%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:8]
    source_date = resolve_target_date(config, target_date)
    print(f"[daily] run_id={run_id}")
    print(f"[daily] source_date={source_date}")

    ensure_result_objects(config)
    df = load_daily_df(config, source_date)
    df = dedupe_daily_df(df, config)
    print(f"[daily] prediction rows={len(df)}")

    policy = load_policy(config)
    pred_df = predict_daily_df(df, config, llm, policy)
    rows = build_insert_rows(pred_df, config, run_id, source_date)
    inserted = insert_judgement_rows(config, rows)

    error_count = int(pred_df.get("error_yn", pd.Series(dtype=bool)).fillna(False).sum())
    review_count = int(pred_df.get("review_needed", pd.Series(dtype=bool)).fillna(False).sum())
    result = {
        "run_id": run_id,
        "source_date": source_date,
        "rows": int(len(pred_df)),
        "inserted": inserted,
        "error_count": error_count,
        "review_needed_count": review_count,
    }
    print(
        f"[daily] done: rows={result['rows']}, inserted={inserted}, "
        f"errors={error_count}, review_needed={review_count}"
    )
    return result


def configure_llm_runtime_from_config(config: Any) -> None:
    configure_llm_runtime(
        calls_per_minute=config.getint(
            "daily_prediction",
            "llm_calls_per_minute",
            fallback=config.getint("runtime", "llm_calls_per_minute", fallback=25),
        ),
        retry_wait_seconds=config.getint(
            "daily_prediction",
            "llm_retry_wait_seconds",
            fallback=config.getint("runtime", "llm_retry_wait_seconds", fallback=300),
        ),
        max_attempts=config.getint(
            "daily_prediction",
            "llm_max_attempts",
            fallback=config.getint("runtime", "llm_max_attempts", fallback=20),
        ),
    )


def resolve_target_date(config: Any, target_date: str | None) -> str:
    if target_date:
        return target_date

    configured = config.get("daily_prediction", "target_date", fallback="auto_yesterday").strip()
    if configured and configured.lower() not in {"auto", "auto_yesterday", "yesterday"}:
        return configured

    return (date.today() - timedelta(days=1)).strftime("%Y-%m-%d")


def ensure_result_objects(config: Any) -> None:
    auto_create = config.getboolean("daily_prediction", "auto_create_result_table", fallback=True)
    if not auto_create:
        return

    ddl_file = config.get("daily_prediction", "ddl_file", fallback="queries/create_daily_result_tables.sql")
    history_table = config.get("daily_prediction", "result_history_table", fallback="ai_stopwork_judgement_history")
    latest_view = config.get("daily_prediction", "latest_view", fallback="v_ai_stopwork_judgement_latest")
    ddl = Path(ddl_file).read_text(encoding="utf-8")
    ddl = ddl.format(history_table=history_table, latest_view=latest_view)

    execute_dml = load_execute_dml(config)
    for statement in split_sql_statements(ddl):
        execute_dml(statement)
    print(f"[daily] result objects ready: table={history_table}, view={latest_view}")


def load_daily_df(config: Any, source_date: str) -> pd.DataFrame:
    query_file = config.get("daily_prediction", "query_file", fallback="queries/daily_prediction.sql")
    query = Path(query_file).read_text(encoding="utf-8").format(target_date=source_date)
    print(f"[daily] query loaded: {query_file}, chars={len(query)}")
    execute_sql = load_execute_sql(config)
    df = execute_sql(query)
    print(f"[daily] SQL completed: rows={len(df)}, cols={len(df.columns)}")
    return df


def dedupe_daily_df(df: pd.DataFrame, config: Any) -> pd.DataFrame:
    if df.empty:
        return df

    cols = config["columns"]
    key_col = config.get("daily_prediction", "dedupe_key", fallback=cols.get("id", "출원번호"))
    order_col = config.get("daily_prediction", "dedupe_order_column", fallback="").strip()

    if key_col not in df.columns:
        print(f"[daily] dedupe skipped: key column not found: {key_col}")
        return df

    before = len(df)
    if order_col and order_col in df.columns:
        df = df.sort_values(order_col)
    df = df.drop_duplicates([key_col], keep="last").copy()
    removed = before - len(df)
    if removed:
        print(f"[daily] dedupe removed={removed}, key={key_col}, order={order_col or 'input_order'}")
    return df


def predict_daily_df(df: pd.DataFrame, config: Any, llm: Any, policy: str) -> pd.DataFrame:
    max_workers = config.getint("daily_prediction", "max_workers", fallback=config.getint("runtime", "max_workers", fallback=8))
    mode = config.get("daily_prediction", "judge_mode", fallback=config.get("runtime", "judge_mode", fallback="tournament"))
    progress_every = config.getint("runtime", "progress_every", fallback=10)
    heartbeat_seconds = config.getint("runtime", "heartbeat_seconds", fallback=30)
    trace_first_n = config.getint("runtime", "trace_first_n", fallback=0)

    records = list(df.to_dict("records"))
    total = len(records)
    results: list[dict] = []
    started = time.monotonic()
    print(f"[daily] start prediction: rows={total}, mode={mode}, max_workers={max_workers}")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for idx, row in enumerate(records, start=1):
            case = row_to_case(row, config)
            futures.append(executor.submit(safe_daily_judge, case, llm, policy, mode, idx <= trace_first_n))

        pending = set(futures)
        done_count = 0
        while pending:
            completed, pending = wait(pending, timeout=heartbeat_seconds, return_when=FIRST_COMPLETED)
            if not completed:
                print(f"[daily] heartbeat: completed={done_count}/{total}, pending={len(pending)}")
                continue

            for future in completed:
                results.append(future.result())
                done_count += 1
                if done_count == total or (progress_every > 0 and done_count % progress_every == 0):
                    print(f"[daily] progress: {done_count}/{total}")

    elapsed = time.monotonic() - started
    print(f"[daily] prediction done: elapsed={format_elapsed(elapsed)}")
    return pd.DataFrame(results)


def safe_daily_judge(case: dict, llm: Any, policy: str, mode: str, trace: bool = False) -> dict:
    try:
        result = judge_case(case=case, llm=llm, policy=policy, mode=mode, trace=trace)
        result["error_yn"] = False
        result["error_message"] = ""
        return result
    except Exception as exc:
        return {
            "id": case.get("id", ""),
            "title": case.get("title", ""),
            "major": case.get("major", ""),
            "middle": case.get("middle", ""),
            "pred": "보류",
            "confidence": 0,
            "reason": "AI 판정 중 오류가 발생했습니다.",
            "applied_step": "ERROR",
            "decisive_evidence": [],
            "review_needed": True,
            "error_yn": True,
            "error_message": str(exc),
        }


def build_insert_rows(pred_df: pd.DataFrame, config: Any, run_id: str, source_date: str) -> list[dict]:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    model_name = config.get("llm", "model_name", fallback=config.get("llm", "model", fallback=""))
    policy_version = config.get("daily_prediction", "policy_version", fallback=Path(config.get("policy", "current_policy_path", fallback="policies/current_policy.md")).name)
    judge_mode = config.get("daily_prediction", "judge_mode", fallback=config.get("runtime", "judge_mode", fallback="tournament"))

    rows = []
    for row in pred_df.to_dict("records"):
        rows.append(
            {
                "run_id": run_id,
                "source_date": source_date,
                "judged_at": now,
                "application_no": row.get("id", ""),
                "title": row.get("title", ""),
                "major_category": row.get("major", ""),
                "middle_category": row.get("middle", ""),
                "ai_pred": row.get("pred", ""),
                "confidence": row.get("confidence", 0),
                "reason": row.get("reason", ""),
                "applied_step": row.get("applied_step", ""),
                "decisive_evidence": json.dumps(row.get("decisive_evidence", []), ensure_ascii=False),
                "review_needed": bool(row.get("review_needed", False)),
                "error_yn": bool(row.get("error_yn", False)),
                "error_message": row.get("error_message", ""),
                "model_name": model_name,
                "policy_version": policy_version,
                "judge_mode": judge_mode,
                "created_at": now,
            }
        )
    return rows


def insert_judgement_rows(config: Any, rows: list[dict]) -> int:
    if not rows:
        return 0

    table = config.get("daily_prediction", "result_history_table", fallback="ai_stopwork_judgement_history")
    batch_size = config.getint("daily_prediction", "insert_batch_size", fallback=100)
    execute_dml = load_execute_dml(config)

    inserted = 0
    columns = [
        "run_id",
        "source_date",
        "judged_at",
        "application_no",
        "title",
        "major_category",
        "middle_category",
        "ai_pred",
        "confidence",
        "reason",
        "applied_step",
        "decisive_evidence",
        "review_needed",
        "error_yn",
        "error_message",
        "model_name",
        "policy_version",
        "judge_mode",
        "created_at",
    ]

    for start in range(0, len(rows), batch_size):
        chunk = rows[start : start + batch_size]
        values_sql = ",\n".join("(" + ", ".join(sql_literal(row.get(col)) for col in columns) + ")" for row in chunk)
        sql = f"INSERT INTO {table} ({', '.join(columns)}) VALUES\n{values_sql}"
        execute_dml(sql)
        inserted += len(chunk)
        print(f"[daily] inserted: {inserted}/{len(rows)}")

    return inserted


def sql_literal(value: Any) -> str:
    if value is None:
        return "NULL"
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return str(value)

    text = str(value).replace("'", "''")
    return f"'{text}'"


def split_sql_statements(sql: str) -> list[str]:
    statements = []
    current = []
    in_quote = False
    idx = 0
    while idx < len(sql):
        ch = sql[idx]
        current.append(ch)
        if ch == "'":
            if in_quote and idx + 1 < len(sql) and sql[idx + 1] == "'":
                current.append(sql[idx + 1])
                idx += 1
            else:
                in_quote = not in_quote
        elif ch == ";" and not in_quote:
            statement = "".join(current).strip().rstrip(";").strip()
            if statement:
                statements.append(statement)
            current = []
        idx += 1

    tail = "".join(current).strip()
    if tail:
        statements.append(tail)
    return statements


def format_elapsed(seconds: float) -> str:
    total = int(seconds)
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h {minutes}m {secs}s"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"
