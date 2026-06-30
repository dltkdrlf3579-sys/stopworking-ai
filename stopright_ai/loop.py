from __future__ import annotations

from datetime import datetime
import time
from pathlib import Path
from typing import Any

import pandas as pd

from .artifacts import append_hard_cases, make_run_dir, save_json, save_predictions
from .errors import build_error_clusters, generate_candidate_policies
from .evaluate import evaluate_policy
from .llm_factory import create_llm
from .llm_json import configure_llm_runtime
from .policy import load_policy, make_policy_diff, save_policy, should_promote, validate_candidate_policy
from .sampling import build_train_validation_dfs


def run_improvement_loop(df: pd.DataFrame, config: Any, llm: Any | None = None) -> None:
    if llm is None:
        llm = create_llm(config)
    configure_llm_runtime_from_config(config)

    max_cycles = config.getint("runtime", "max_cycles", fallback=1)
    sleep_seconds = config.getint("runtime", "sleep_seconds", fallback=0)

    cycle = 0
    while max_cycles <= 0 or cycle < max_cycles:
        cycle += 1
        result = run_one_cycle(df=df, config=config, llm=llm, cycle=cycle)
        print(
            f"[cycle {cycle}] base_score={result['base_metrics'].get('score', 0):.4f} "
            f"winner={result.get('winner_name')} promoted={result.get('promoted')}",
            flush=True,
        )

        if sleep_seconds > 0:
            time.sleep(sleep_seconds)


def run_one_cycle(df: pd.DataFrame, config: Any, llm: Any, cycle: int = 1) -> dict:
    started_at = datetime.now()
    started = time.monotonic()
    configure_llm_runtime_from_config(config)
    run_dir = make_run_dir(config)
    train_df, validation_df = build_train_validation_dfs(df, config, cycle=cycle)
    current_policy = load_policy(config)
    (run_dir / "current_policy.md").write_text(current_policy, encoding="utf-8")

    print(f"[cycle {cycle}] run_dir={run_dir}", flush=True)
    print(f"[cycle {cycle}] train rows={len(train_df)}, validation rows={len(validation_df)}", flush=True)

    train_pred_df, train_base_metrics = evaluate_policy(train_df, llm, config, current_policy, label="train-baseline")
    save_predictions(run_dir / "train_baseline_predictions.csv", train_pred_df)
    save_json(run_dir / "train_baseline_metrics.json", train_base_metrics)
    append_hard_cases(config, train_pred_df)

    error_clusters = build_error_clusters(train_pred_df)
    save_json(run_dir / "error_clusters.json", error_clusters)
    print(f"[cycle {cycle}] error_clusters={len(error_clusters)}", flush=True)

    candidate_count = config.getint("policy", "candidate_count", fallback=3)
    print(f"[cycle {cycle}] generating candidates: count={candidate_count}", flush=True)
    candidates = generate_candidate_policies(llm, current_policy, error_clusters, candidate_count, config=config)
    save_json(run_dir / "candidate_manifest.json", candidates)
    print(f"[cycle {cycle}] generated candidates={len(candidates)}", flush=True)

    validation_base_pred_df, validation_base_metrics = evaluate_policy(
        validation_df,
        llm,
        config,
        current_policy,
        label="validation-baseline",
    )
    save_predictions(run_dir / "validation_baseline_predictions.csv", validation_base_pred_df)
    save_json(run_dir / "validation_baseline_metrics.json", validation_base_metrics)

    winner = {
        "name": "baseline",
        "policy_text": current_policy,
        "metrics": validation_base_metrics,
        "predictions_path": str(run_dir / "validation_baseline_predictions.csv"),
    }
    promoted = False
    promotion_reason = "no candidate promoted"

    for candidate in candidates:
        name = sanitize_name(candidate["name"])
        policy_text = candidate["policy_text"]
        candidate_dir = run_dir / name
        candidate_dir.mkdir(parents=True, exist_ok=True)
        (candidate_dir / "policy.md").write_text(policy_text, encoding="utf-8")
        (candidate_dir / "hypothesis.txt").write_text(candidate.get("hypothesis", ""), encoding="utf-8")
        (candidate_dir / "addendum.md").write_text(candidate.get("rendered_addendum", ""), encoding="utf-8")
        (candidate_dir / "diagnosis.json").write_text(
            json_dump(
                {
                    "target_error_cluster": candidate.get("target_error_cluster", ""),
                    "policy_diagnosis": candidate.get("policy_diagnosis", ""),
                    "why_wrong": candidate.get("why_wrong", ""),
                    "why_correct_cases_remain_safe": candidate.get("why_correct_cases_remain_safe", ""),
                    "regression_risk": candidate.get("regression_risk", ""),
                    "hypothesis": candidate.get("hypothesis", ""),
                }
            ),
            encoding="utf-8",
        )
        (candidate_dir / "policy.diff").write_text(make_policy_diff(current_policy, policy_text), encoding="utf-8")

        valid_policy, validation_reason = validate_candidate_policy(current_policy, policy_text, config)
        (candidate_dir / "validation.txt").write_text(validation_reason, encoding="utf-8")
        if not valid_policy:
            print(f"[candidate:{candidate['name']}] skipped: {validation_reason}", flush=True)
            continue

        pred_df, metrics = evaluate_policy(validation_df, llm, config, policy_text, label=f"candidate:{candidate['name']}")
        save_predictions(candidate_dir / "predictions.csv", pred_df)
        save_json(candidate_dir / "metrics.json", metrics)

        ok, reason = should_promote(validation_base_metrics, metrics, config)
        if ok and metrics.get("score", 0) > winner["metrics"].get("score", 0):
            winner = {
                "name": candidate["name"],
                "policy_text": policy_text,
                "metrics": metrics,
                "predictions_path": str(candidate_dir / "predictions.csv"),
            }
            promoted = True
            promotion_reason = reason

    if promoted:
        save_policy(config, winner["policy_text"])
        Path(run_dir / "PROMOTED.txt").write_text(promotion_reason, encoding="utf-8")

    ended_at = datetime.now()
    elapsed_seconds = time.monotonic() - started
    result = {
        "cycle": cycle,
        "run_dir": str(run_dir),
        "started_at": started_at.strftime("%Y-%m-%d %H:%M:%S"),
        "ended_at": ended_at.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_seconds": elapsed_seconds,
        "train_base_metrics": train_base_metrics,
        "base_metrics": validation_base_metrics,
        "validation_base_metrics": validation_base_metrics,
        "winner_name": winner["name"],
        "winner_metrics": winner["metrics"],
        "promoted": promoted,
        "promotion_reason": promotion_reason,
    }
    save_json(run_dir / "cycle_result.json", result)
    print(f"[cycle {cycle}] done: elapsed={format_elapsed(elapsed_seconds)}", flush=True)
    return result


def configure_llm_runtime_from_config(config: Any) -> None:
    configure_llm_runtime(
        calls_per_minute=config.getint("runtime", "llm_calls_per_minute", fallback=25),
        retry_wait_seconds=config.getint("runtime", "llm_retry_wait_seconds", fallback=300),
        max_attempts=config.getint("runtime", "llm_max_attempts", fallback=20),
    )


def sanitize_name(name: str) -> str:
    keep = []
    for ch in name:
        if ch.isalnum() or ch in {"-", "_"}:
            keep.append(ch)
        else:
            keep.append("_")
    return "".join(keep)[:80] or "candidate"


def json_dump(value: Any) -> str:
    import json

    return json.dumps(value, ensure_ascii=False, indent=2)


def format_elapsed(seconds: float) -> str:
    total = int(seconds)
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h {minutes}m {secs}s"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"
