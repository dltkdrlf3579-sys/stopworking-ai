from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import pandas as pd

from .artifacts import append_hard_cases, make_run_dir, save_json, save_predictions
from .errors import build_error_clusters, generate_candidate_policies
from .evaluate import evaluate_policy
from .llm_factory import create_llm
from .policy import load_policy, save_policy, should_promote
from .sampling import build_eval_df


def run_improvement_loop(df: pd.DataFrame, config: Any, llm: Any | None = None) -> None:
    if llm is None:
        llm = create_llm(config)

    max_cycles = config.getint("runtime", "max_cycles", fallback=1)
    sleep_seconds = config.getint("runtime", "sleep_seconds", fallback=0)

    cycle = 0
    while max_cycles <= 0 or cycle < max_cycles:
        cycle += 1
        result = run_one_cycle(df=df, config=config, llm=llm, cycle=cycle)
        print(
            f"[cycle {cycle}] base_score={result['base_metrics'].get('score', 0):.4f} "
            f"winner={result.get('winner_name')} promoted={result.get('promoted')}"
        )

        if sleep_seconds > 0:
            time.sleep(sleep_seconds)


def run_one_cycle(df: pd.DataFrame, config: Any, llm: Any, cycle: int = 1) -> dict:
    run_dir = make_run_dir(config)
    eval_df = build_eval_df(df, config)
    current_policy = load_policy(config)

    print(f"[cycle {cycle}] run_dir={run_dir}", flush=True)
    print(f"[cycle {cycle}] evaluation rows={len(eval_df)}", flush=True)

    base_pred_df, base_metrics = evaluate_policy(eval_df, llm, config, current_policy, label="baseline")
    save_predictions(run_dir / "baseline_predictions.csv", base_pred_df)
    save_json(run_dir / "baseline_metrics.json", base_metrics)
    append_hard_cases(config, base_pred_df)

    error_clusters = build_error_clusters(base_pred_df)
    save_json(run_dir / "error_clusters.json", error_clusters)
    print(f"[cycle {cycle}] error_clusters={len(error_clusters)}", flush=True)

    candidate_count = config.getint("policy", "candidate_count", fallback=3)
    print(f"[cycle {cycle}] generating candidates: count={candidate_count}", flush=True)
    candidates = generate_candidate_policies(llm, current_policy, error_clusters, candidate_count)
    save_json(run_dir / "candidate_manifest.json", candidates)
    print(f"[cycle {cycle}] generated candidates={len(candidates)}", flush=True)

    winner = {
        "name": "baseline",
        "policy_text": current_policy,
        "metrics": base_metrics,
        "predictions_path": str(run_dir / "baseline_predictions.csv"),
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

        pred_df, metrics = evaluate_policy(eval_df, llm, config, policy_text, label=f"candidate:{candidate['name']}")
        save_predictions(candidate_dir / "predictions.csv", pred_df)
        save_json(candidate_dir / "metrics.json", metrics)

        ok, reason = should_promote(base_metrics, metrics, config)
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

    result = {
        "cycle": cycle,
        "run_dir": str(run_dir),
        "base_metrics": base_metrics,
        "winner_name": winner["name"],
        "winner_metrics": winner["metrics"],
        "promoted": promoted,
        "promotion_reason": promotion_reason,
    }
    save_json(run_dir / "cycle_result.json", result)
    return result


def sanitize_name(name: str) -> str:
    keep = []
    for ch in name:
        if ch.isalnum() or ch in {"-", "_"}:
            keep.append(ch)
        else:
            keep.append("_")
    return "".join(keep)[:80] or "candidate"
