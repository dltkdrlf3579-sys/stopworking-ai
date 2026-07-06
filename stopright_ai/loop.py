from __future__ import annotations

from datetime import datetime
import time
from pathlib import Path
from typing import Any

import pandas as pd

from .artifacts import append_hard_cases, make_run_dir, save_error_slices, save_json, save_predictions
from .errors import build_error_clusters, generate_candidate_policies
from .evaluate import evaluate_policy
from .llm_factory import create_llm
from .llm_json import configure_llm_runtime
from .policy import load_policy, make_policy_diff, save_policy, should_promote, validate_candidate_policy
from .route_score_evolution import run_route_score_evolution, save_promoted_profile
from .route_tuning import run_guardrail_autotune
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

    route_score_mode = config.get("runtime", "route_score_mode", fallback="record").strip().lower()
    if route_score_mode in {"sweep", "all", "compare"}:
        return run_route_score_sweep_cycle(
            config=config,
            llm=llm,
            cycle=cycle,
            run_dir=run_dir,
            train_df=train_df,
            validation_df=validation_df,
            current_policy=current_policy,
            started_at=started_at,
            started=started,
        )

    train_pred_df, train_base_metrics = evaluate_policy(train_df, llm, config, current_policy, label="train-baseline")
    save_predictions(run_dir / "train_baseline_predictions.csv", train_pred_df)
    save_error_slices(run_dir, "train_baseline", train_pred_df)
    save_json(run_dir / "train_baseline_metrics.json", train_base_metrics)
    append_hard_cases(config, train_pred_df)

    error_clusters = build_error_clusters(train_pred_df)
    save_json(run_dir / "error_clusters.json", error_clusters)
    print(f"[cycle {cycle}] error_clusters={len(error_clusters)}", flush=True)

    candidate_count = config.getint("policy", "candidate_count", fallback=3)
    if candidate_count <= 0:
        print(f"[cycle {cycle}] candidate generation skipped: candidate_count={candidate_count}", flush=True)
        candidates = []
    else:
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
    save_error_slices(run_dir, "validation_baseline", validation_base_pred_df)
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


def run_route_score_sweep_cycle(
    config: Any,
    llm: Any,
    cycle: int,
    run_dir: Path,
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    current_policy: str,
    started_at: datetime,
    started: float,
) -> dict:
    modes = get_route_score_sweep_modes(config)
    original_mode = config.get("runtime", "route_score_mode", fallback="record")
    mode_results = []

    print(f"[cycle {cycle}] route_score_mode sweep: modes={','.join(modes)}", flush=True)
    print(f"[cycle {cycle}] candidate generation skipped during route_score sweep", flush=True)

    try:
        for mode in modes:
            config.set("runtime", "route_score_mode", mode)
            mode_dir = run_dir / f"route_score_{sanitize_name(mode)}"
            mode_dir.mkdir(parents=True, exist_ok=True)
            (mode_dir / "current_policy.md").write_text(current_policy, encoding="utf-8")

            train_pred_df, train_metrics = evaluate_policy(
                train_df,
                llm,
                config,
                current_policy,
                label=f"train-route-{mode}",
            )
            save_predictions(mode_dir / "train_predictions.csv", train_pred_df)
            save_json(mode_dir / "train_metrics.json", train_metrics)

            validation_pred_df, validation_metrics = evaluate_policy(
                validation_df,
                llm,
                config,
                current_policy,
                label=f"validation-route-{mode}",
            )
            save_predictions(mode_dir / "validation_predictions.csv", validation_pred_df)
            save_json(mode_dir / "validation_metrics.json", validation_metrics)

            mode_result = {
                "mode": mode,
                "train_metrics": train_metrics,
                "validation_metrics": validation_metrics,
                "train_predictions_path": str(mode_dir / "train_predictions.csv"),
                "validation_predictions_path": str(mode_dir / "validation_predictions.csv"),
            }
            mode_results.append(mode_result)
            print_route_mode_summary(cycle, mode, validation_metrics)

        if config.getboolean("runtime", "route_score_autotune", fallback=True):
            autotune_result = maybe_run_route_score_autotune(
                config=config,
                run_dir=run_dir,
                mode_results=mode_results,
                cycle=cycle,
            )
            if autotune_result:
                mode_results.append(autotune_result)

        if config.getboolean("runtime", "route_score_evolve", fallback=True):
            evolution_result = maybe_run_route_score_evolution(
                config=config,
                run_dir=run_dir,
                mode_results=mode_results,
                cycle=cycle,
            )
            if evolution_result:
                mode_results.append(evolution_result)
    finally:
        config.set("runtime", "route_score_mode", original_mode)

    comparison_df = build_route_mode_comparison(mode_results)
    comparison_df.to_csv(run_dir / "route_score_mode_comparison.csv", index=False, encoding="utf-8-sig")
    save_json(run_dir / "route_score_mode_comparison.json", mode_results)

    winner = max(mode_results, key=lambda item: item["validation_metrics"].get("score", 0)) if mode_results else None
    ended_at = datetime.now()
    elapsed_seconds = time.monotonic() - started
    result = {
        "cycle": cycle,
        "run_dir": str(run_dir),
        "started_at": started_at.strftime("%Y-%m-%d %H:%M:%S"),
        "ended_at": ended_at.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_seconds": elapsed_seconds,
        "sweep": True,
        "route_score_modes": modes,
        "mode_results": mode_results,
        "train_base_metrics": mode_results[0]["train_metrics"] if mode_results else {},
        "base_metrics": mode_results[0]["validation_metrics"] if mode_results else {},
        "validation_base_metrics": mode_results[0]["validation_metrics"] if mode_results else {},
        "winner_name": f"route_score_mode:{winner['mode']}" if winner else "none",
        "winner_metrics": winner["validation_metrics"] if winner else {},
        "promoted": False,
        "promotion_reason": "route_score sweep only; policy promotion disabled",
    }
    save_json(run_dir / "cycle_result.json", result)
    print(f"[cycle {cycle}] route_score sweep done: winner={result['winner_name']} elapsed={format_elapsed(elapsed_seconds)}", flush=True)
    return result


def maybe_run_route_score_autotune(config: Any, run_dir: Path, mode_results: list[dict], cycle: int) -> dict | None:
    record = next((item for item in mode_results if item.get("mode") == "record"), None)
    if not record:
        print(f"[cycle {cycle}] route_score autotune skipped: record mode result not found", flush=True)
        return None

    train_path = Path(record["train_predictions_path"])
    validation_path = Path(record["validation_predictions_path"])
    if not train_path.exists() or not validation_path.exists():
        print(f"[cycle {cycle}] route_score autotune skipped: record prediction files not found", flush=True)
        return None

    print(f"[cycle {cycle}] route_score autotune: searching guardrail thresholds", flush=True)
    train_pred_df = pd.read_csv(train_path)
    validation_pred_df = pd.read_csv(validation_path)
    tuning = run_guardrail_autotune(train_pred_df, validation_pred_df, config)
    best = tuning["best"]

    tune_dir = run_dir / "route_score_autotuned_guardrail"
    tune_dir.mkdir(parents=True, exist_ok=True)
    save_predictions(tune_dir / "train_predictions.csv", best["train_predictions"])
    save_predictions(tune_dir / "validation_predictions.csv", best["validation_predictions"])
    save_json(tune_dir / "train_metrics.json", best["train_metrics"])
    save_json(tune_dir / "validation_metrics.json", best["validation_metrics"])
    save_json(
        tune_dir / "selected_guardrail.json",
        {
            "candidate": best["candidate"],
            "train_delta": best["train_delta"],
            "validation_delta": best["validation_delta"],
            "summary_row": best["summary_row"],
        },
    )
    pd.DataFrame(tuning["candidates"]).to_csv(tune_dir / "candidate_thresholds.csv", index=False, encoding="utf-8-sig")

    metrics = best["validation_metrics"]
    print(
        f"[cycle {cycle}] route_score autotuned_guardrail "
        f"candidate={best['candidate'].get('name')} "
        f"validation acc={metrics.get('accuracy', 0):.4f} "
        f"score={metrics.get('score', 0):.4f} "
        f"TR={metrics.get('true_recall', 0):.4f} "
        f"TP={metrics.get('true_precision', 0):.4f}",
        flush=True,
    )

    return {
        "mode": "autotuned_guardrail",
        "train_metrics": best["train_metrics"],
        "validation_metrics": best["validation_metrics"],
        "train_predictions_path": str(tune_dir / "train_predictions.csv"),
        "validation_predictions_path": str(tune_dir / "validation_predictions.csv"),
        "autotune_candidate": best["candidate"],
        "autotune_train_delta": best["train_delta"],
        "autotune_validation_delta": best["validation_delta"],
    }


def maybe_run_route_score_evolution(config: Any, run_dir: Path, mode_results: list[dict], cycle: int) -> dict | None:
    record = next((item for item in mode_results if item.get("mode") == "record"), None)
    if not record:
        print(f"[cycle {cycle}] route_score evolution skipped: record mode result not found", flush=True)
        return None

    train_path = Path(record["train_predictions_path"])
    validation_path = Path(record["validation_predictions_path"])
    if not train_path.exists() or not validation_path.exists():
        print(f"[cycle {cycle}] route_score evolution skipped: record prediction files not found", flush=True)
        return None

    print(f"[cycle {cycle}] route_score evolution: testing scorecard profile candidates", flush=True)
    train_pred_df = pd.read_csv(train_path)
    validation_pred_df = pd.read_csv(validation_path)
    evolution = run_route_score_evolution(train_pred_df, validation_pred_df, config)
    best = evolution["best"]

    evolve_dir = run_dir / "route_score_evolved_profile"
    evolve_dir.mkdir(parents=True, exist_ok=True)
    save_predictions(evolve_dir / "train_predictions.csv", best["train_predictions"])
    save_predictions(evolve_dir / "validation_predictions.csv", best["validation_predictions"])
    save_json(evolve_dir / "train_metrics.json", best["train_metrics"])
    save_json(evolve_dir / "validation_metrics.json", best["validation_metrics"])
    save_json(
        evolve_dir / "selected_profile.json",
        {
            "candidate": best["candidate"],
            "train_delta": best["train_delta"],
            "validation_delta": best["validation_delta"],
            "summary_row": best["summary_row"],
        },
    )
    pd.DataFrame(evolution["candidates"]).to_csv(evolve_dir / "profile_candidates.csv", index=False, encoding="utf-8-sig")

    promoted = bool(best["summary_row"].get("validation_gate")) and best["candidate"].get("name") != "route_profile_noop"
    profile_path = None
    if promoted and config.getboolean("runtime", "route_score_evolve_save_profile", fallback=True):
        profile_path = save_promoted_profile(
            config,
            best["candidate"].get("profile", {}),
            {
                "cycle": cycle,
                "run_dir": str(run_dir),
                "candidate_name": best["candidate"].get("name"),
                "train_delta": best["train_delta"],
                "validation_delta": best["validation_delta"],
                "summary_row": best["summary_row"],
            },
        )
        (evolve_dir / "PROMOTED_PROFILE.txt").write_text(str(profile_path), encoding="utf-8")

    metrics = best["validation_metrics"]
    print(
        f"[cycle {cycle}] route_score evolved_profile "
        f"candidate={best['candidate'].get('name')} promoted={promoted} "
        f"validation acc={metrics.get('accuracy', 0):.4f} "
        f"score={metrics.get('score', 0):.4f} "
        f"TR={metrics.get('true_recall', 0):.4f} "
        f"TP={metrics.get('true_precision', 0):.4f}",
        flush=True,
    )

    return {
        "mode": "evolved_profile",
        "train_metrics": best["train_metrics"],
        "validation_metrics": best["validation_metrics"],
        "train_predictions_path": str(evolve_dir / "train_predictions.csv"),
        "validation_predictions_path": str(evolve_dir / "validation_predictions.csv"),
        "evolved_profile_candidate": best["candidate"],
        "evolved_profile_promoted": promoted,
        "evolved_profile_path": str(profile_path) if profile_path else "",
        "evolved_profile_train_delta": best["train_delta"],
        "evolved_profile_validation_delta": best["validation_delta"],
    }


def get_route_score_sweep_modes(config: Any) -> list[str]:
    raw = config.get("runtime", "route_score_modes", fallback="record,assist,guardrail")
    modes = []
    for item in raw.split(","):
        mode = item.strip().lower()
        if not mode:
            continue
        if mode in {"sweep", "all", "compare"}:
            continue
        if mode not in {"off", "record", "assist", "guardrail"}:
            print(f"[route_score_sweep] ignored unknown mode: {mode}", flush=True)
            continue
        if mode not in modes:
            modes.append(mode)
    return modes or ["record", "assist", "guardrail"]


def build_route_mode_comparison(mode_results: list[dict]) -> pd.DataFrame:
    rows = []
    for item in mode_results:
        train = item.get("train_metrics", {})
        validation = item.get("validation_metrics", {})
        rows.append(
            {
                "mode": item.get("mode", ""),
                "train_accuracy": train.get("accuracy", 0),
                "train_score": train.get("score", 0),
                "train_true_recall": train.get("true_recall", 0),
                "train_true_precision": train.get("true_precision", 0),
                "train_false_recall": train.get("false_recall", 0),
                "train_false_precision": train.get("false_precision", 0),
                "train_fn_true_as_false": train.get("fn_true_as_false", 0),
                "train_fp_false_as_true": train.get("fp_false_as_true", 0),
                "validation_accuracy": validation.get("accuracy", 0),
                "validation_score": validation.get("score", 0),
                "validation_true_recall": validation.get("true_recall", 0),
                "validation_true_precision": validation.get("true_precision", 0),
                "validation_false_recall": validation.get("false_recall", 0),
                "validation_false_precision": validation.get("false_precision", 0),
                "validation_fn_true_as_false": validation.get("fn_true_as_false", 0),
                "validation_fp_false_as_true": validation.get("fp_false_as_true", 0),
                "validation_excluded_n": validation.get("excluded_n", 0),
            }
        )
    return pd.DataFrame(rows)


def print_route_mode_summary(cycle: int, mode: str, metrics: dict) -> None:
    print(
        f"[cycle {cycle}] route_score_mode={mode} "
        f"validation acc={metrics.get('accuracy', 0):.4f} "
        f"score={metrics.get('score', 0):.4f} "
        f"TR={metrics.get('true_recall', 0):.4f} "
        f"TP={metrics.get('true_precision', 0):.4f} "
        f"FR={metrics.get('false_recall', 0):.4f} "
        f"FP={metrics.get('false_precision', 0):.4f}",
        flush=True,
    )


def configure_llm_runtime_from_config(config: Any) -> None:
    configure_llm_runtime(
        calls_per_minute=config.getint("runtime", "llm_calls_per_minute", fallback=25),
        retry_wait_seconds=config.getint("runtime", "llm_retry_wait_seconds", fallback=300),
        max_attempts=config.getint("runtime", "llm_max_attempts", fallback=20),
        json_parse_retries=config.getint("runtime", "llm_json_parse_retries", fallback=1),
        log_rate_limit_waits=config.getboolean("runtime", "llm_log_rate_limit_waits", fallback=False),
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
