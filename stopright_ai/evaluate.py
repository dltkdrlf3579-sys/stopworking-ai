from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import pandas as pd

from .case_prepare import row_to_case
from .judge import judge_case
from .llm_json import configure_llm_runtime


def evaluate_policy(df: pd.DataFrame, llm: Any, config: Any, policy: str, label: str = "policy") -> tuple[pd.DataFrame, dict]:
    max_workers = config.getint("runtime", "max_workers", fallback=8)
    mode = config.get("runtime", "judge_mode", fallback="tournament")
    progress_every = config.getint("runtime", "progress_every", fallback=10)
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
        for row in records:
            case = row_to_case(row, config)
            futures.append(executor.submit(_safe_judge_case, case, llm, policy, mode))

        for done, future in enumerate(as_completed(futures), start=1):
            results.append(future.result())
            if done == total or (progress_every > 0 and done % progress_every == 0):
                print(f"[{label}] progress: {done}/{total}", flush=True)

    pred_df = pd.DataFrame(results)
    metrics = compute_metrics(pred_df)
    print(
        f"[{label}] done: accuracy={metrics.get('accuracy', 0):.4f}, score={metrics.get('score', 0):.4f}",
        flush=True,
    )
    return pred_df, metrics


def _safe_judge_case(case: dict, llm: Any, policy: str, mode: str) -> dict:
    try:
        return judge_case(case=case, llm=llm, policy=policy, mode=mode)
    except Exception as exc:
        return {
            "id": case.get("id", ""),
            "label": case.get("label", ""),
            "pred": "가성",
            "correct": case.get("label", "") == "가성",
            "confidence": 0,
            "reason": f"판정 실패로 보수적 가성 처리: {exc}",
            "applied_step": "ERROR",
            "review_needed": True,
            "decisive_evidence": [],
            "evidence": {},
            "major": case.get("major", ""),
            "middle": case.get("middle", ""),
            "title": case.get("title", ""),
            "error": str(exc),
        }


def compute_metrics(pred_df: pd.DataFrame) -> dict:
    if pred_df.empty:
        return {"n": 0, "score": 0}

    labels = pred_df["label"]
    preds = pred_df["pred"]

    tp_true = int(((labels == "진성") & (preds == "진성")).sum())
    fn_true = int(((labels == "진성") & (preds == "가성")).sum())
    fp_true = int(((labels == "가성") & (preds == "진성")).sum())
    tn_true = int(((labels == "가성") & (preds == "가성")).sum())

    accuracy = float((labels == preds).mean())
    true_precision = safe_div(tp_true, tp_true + fp_true)
    true_recall = safe_div(tp_true, tp_true + fn_true)
    false_precision = safe_div(tn_true, tn_true + fn_true)
    false_recall = safe_div(tn_true, tn_true + fp_true)

    category_gap = category_accuracy_gap(pred_df, "major")
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
        "n": int(len(pred_df)),
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
