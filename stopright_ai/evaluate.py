from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import time
from typing import Any

import pandas as pd

from .case_prepare import row_to_case
from .judge import judge_case
from .llm_json import configure_llm_runtime


def evaluate_policy(df: pd.DataFrame, llm: Any, config: Any, policy: str, label: str = "policy") -> tuple[pd.DataFrame, dict]:
    started = time.monotonic()
    max_workers = config.getint("runtime", "max_workers", fallback=8)
    mode = config.get("runtime", "judge_mode", fallback="tournament")
    progress_every = config.getint("runtime", "progress_every", fallback=10)
    heartbeat_seconds = config.getint("runtime", "heartbeat_seconds", fallback=30)
    trace_first_n = config.getint("runtime", "trace_first_n", fallback=0)
    configure_llm_runtime(
        calls_per_minute=config.getint("runtime", "llm_calls_per_minute", fallback=25),
        retry_wait_seconds=config.getint("runtime", "llm_retry_wait_seconds", fallback=300),
        max_attempts=config.getint("runtime", "llm_max_attempts", fallback=20),
    )

    records = list(df.to_dict("records"))
    results: list[dict] = []
    total = len(records)

    print(
        f"[{label}] start: rows={total}, mode={mode}, max_workers={max_workers}",
        flush=True,
    )

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for idx, row in enumerate(records, start=1):
            case = row_to_case(row, config)
            futures.append(executor.submit(_safe_judge_case, case, llm, policy, mode, idx <= trace_first_n))

        pending = set(futures)
        done_count = 0
        while pending:
            completed, pending = wait(pending, timeout=heartbeat_seconds, return_when=FIRST_COMPLETED)
            if not completed:
                print(f"[{label}] heartbeat: completed={done_count}/{total}, pending={len(pending)} {format_live_metrics(results)}", flush=True)
                continue

            for future in completed:
                results.append(future.result())
                done_count += 1
                if done_count == total or (progress_every > 0 and done_count % progress_every == 0):
                    print(f"[{label}] progress: {done_count}/{total} {format_live_metrics(results)}", flush=True)

    pred_df = pd.DataFrame(results)
    metrics = compute_metrics(pred_df)
    elapsed_seconds = time.monotonic() - started
    metrics["elapsed_seconds"] = elapsed_seconds
    print(
        f"[{label}] done: accuracy={metrics.get('accuracy', 0):.4f}, "
        f"score={metrics.get('score', 0):.4f}, elapsed={format_elapsed(elapsed_seconds)}",
        flush=True,
    )
    return pred_df, metrics


def format_live_metrics(results: list[dict]) -> str:
    metrics = compute_metrics(pd.DataFrame(results))
    n = metrics.get("n", 0)
    excluded = metrics.get("excluded_n", 0)
    return (
        f"n={n}, excluded={excluded}, "
        f"acc={metrics.get('accuracy', 0):.4f}, "
        f"TR={metrics.get('true_recall', 0):.4f}, "
        f"TP={metrics.get('true_precision', 0):.4f}, "
        f"FR={metrics.get('false_recall', 0):.4f}, "
        f"FP={metrics.get('false_precision', 0):.4f}, "
        f"FN={metrics.get('fn_true_as_false', 0)}, "
        f"FPerr={metrics.get('fp_false_as_true', 0)}"
    )


def _safe_judge_case(case: dict, llm: Any, policy: str, mode: str, trace: bool = False) -> dict:
    try:
        return judge_case(case=case, llm=llm, policy=policy, mode=mode, trace=trace)
    except Exception as exc:
        if is_context_length_error(exc):
            return {
                "id": case.get("id", ""),
                "label": case.get("label", ""),
                "pred": "보류",
                "correct": False,
                "confidence": 0,
                "reason": f"토큰 한계로 판정 보류: {exc}",
                "applied_step": "CONTEXT_LIMIT",
                "review_needed": True,
                "exclude_from_metrics": True,
                "decisive_evidence": [],
                "evidence": {},
                "major": case.get("major", ""),
                "middle": case.get("middle", ""),
                "title": case.get("title", ""),
                "error": str(exc),
            }

        return {
            "id": case.get("id", ""),
            "label": case.get("label", ""),
            "pred": "보류",
            "correct": False,
            "confidence": 0,
            "reason": f"판정 실패로 판정 보류: {exc}",
            "applied_step": "ERROR",
            "review_needed": True,
            "decisive_evidence": [],
            "evidence": {},
            "major": case.get("major", ""),
            "middle": case.get("middle", ""),
            "title": case.get("title", ""),
            "error": str(exc),
            "exclude_from_metrics": True,
        }


def compute_metrics(pred_df: pd.DataFrame) -> dict:
    if pred_df.empty:
        return {"total_n": 0, "n": 0, "excluded_n": 0, "score": 0}

    excluded = pred_df.get("exclude_from_metrics", False)
    if not isinstance(excluded, pd.Series):
        excluded = pd.Series([bool(excluded)] * len(pred_df), index=pred_df.index)

    eval_df = pred_df[~excluded.map(normalize_bool)].copy()
    excluded_n = int(len(pred_df) - len(eval_df))

    if eval_df.empty:
        return {
            "total_n": int(len(pred_df)),
            "n": 0,
            "excluded_n": excluded_n,
            "accuracy": 0,
            "true_precision": 0,
            "true_recall": 0,
            "false_precision": 0,
            "false_recall": 0,
            "fn_true_as_false": 0,
            "fp_false_as_true": 0,
            "tp_true": 0,
            "tn_false": 0,
            "category_accuracy_gap": 0,
            "score": 0,
        }

    labels = eval_df["label"]
    preds = eval_df["pred"]

    tp_true = int(((labels == "진성") & (preds == "진성")).sum())
    fn_true = int(((labels == "진성") & (preds == "가성")).sum())
    fp_true = int(((labels == "가성") & (preds == "진성")).sum())
    tn_true = int(((labels == "가성") & (preds == "가성")).sum())

    accuracy = float((labels == preds).mean())
    true_precision = safe_div(tp_true, tp_true + fp_true)
    true_recall = safe_div(tp_true, tp_true + fn_true)
    false_precision = safe_div(tn_true, tn_true + fn_true)
    false_recall = safe_div(tn_true, tn_true + fp_true)

    category_gap = category_accuracy_gap(eval_df, "major")
    fn_rate = safe_div(fn_true, max(1, int((labels == "진성").sum())))
    fp_rate = safe_div(fp_true, max(1, int((labels == "가성").sum())))

    score = (
        accuracy
        + 0.35 * true_recall
        + 0.15 * true_precision
        + 0.10 * false_precision
        - 0.40 * fn_rate
        - 0.20 * fp_rate
        - 0.10 * category_gap
    )

    return {
        "total_n": int(len(pred_df)),
        "n": int(len(eval_df)),
        "excluded_n": excluded_n,
        "accuracy": accuracy,
        "true_precision": true_precision,
        "true_recall": true_recall,
        "false_precision": false_precision,
        "false_recall": false_recall,
        "fn_true_as_false": fn_true,
        "fp_false_as_true": fp_true,
        "tp_true": tp_true,
        "tn_false": tn_true,
        "category_accuracy_gap": category_gap,
        "score": score,
    }


def category_accuracy_gap(pred_df: pd.DataFrame, col: str) -> float:
    if col not in pred_df.columns:
        return 0.0
    grouped = pred_df.groupby(col)["correct"].mean()
    if len(grouped) <= 1:
        return 0.0
    return float(grouped.max() - grouped.min())


def safe_div(num: int | float, den: int | float) -> float:
    if den == 0:
        return 0.0
    return float(num / den)


def normalize_bool(value: Any) -> bool:
    if value is None:
        return False
    try:
        is_missing = pd.isna(value)
    except Exception:
        is_missing = False
    if isinstance(is_missing, bool) and is_missing:
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"true", "1", "yes", "y"}:
            return True
        if text in {"false", "0", "no", "n", ""}:
            return False
    return bool(value)


def format_elapsed(seconds: float) -> str:
    total = int(seconds)
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h {minutes}m {secs}s"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def is_context_length_error(exc: Exception) -> bool:
    text = str(exc).lower()
    markers = [
        "maximum context length",
        "context length",
        "token limit",
        "too many tokens",
        "requested 0 output tokens",
        "input tokens",
    ]
    return any(marker in text for marker in markers)
