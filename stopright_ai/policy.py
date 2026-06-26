from __future__ import annotations

from pathlib import Path
from typing import Any


def load_policy(config: Any) -> str:
    path = Path(config.get("policy", "current_policy_path", fallback="policies/current_policy.md"))
    if not path.exists():
        raise FileNotFoundError(f"Policy file not found: {path.resolve()}")
    return path.read_text(encoding="utf-8")


def save_policy(config: Any, policy_text: str) -> None:
    path = Path(config.get("policy", "current_policy_path", fallback="policies/current_policy.md"))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(policy_text, encoding="utf-8")


def should_promote(base_metrics: dict, candidate_metrics: dict, config: Any) -> tuple[bool, str]:
    min_gain = config.getfloat("policy", "promotion_min_score_gain", fallback=0.01)
    max_fn_regression = config.getint("policy", "promotion_max_fn_regression", fallback=0)
    max_fp_regression = config.getint("policy", "promotion_max_fp_regression", fallback=5)

    score_gain = candidate_metrics.get("score", 0) - base_metrics.get("score", 0)
    fn_regression = candidate_metrics.get("fn_true_as_false", 0) - base_metrics.get("fn_true_as_false", 0)
    fp_regression = candidate_metrics.get("fp_false_as_true", 0) - base_metrics.get("fp_false_as_true", 0)

    if score_gain < min_gain:
        return False, f"score_gain {score_gain:.4f} < {min_gain:.4f}"
    if fn_regression > max_fn_regression:
        return False, f"FN regression {fn_regression} > {max_fn_regression}"
    if fp_regression > max_fp_regression:
        return False, f"FP regression {fp_regression} > {max_fp_regression}"
    return True, f"promoted: score_gain={score_gain:.4f}, fn_regression={fn_regression}, fp_regression={fp_regression}"

