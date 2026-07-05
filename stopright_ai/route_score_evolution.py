from __future__ import annotations

from copy import deepcopy
from datetime import datetime
import json
from pathlib import Path
from typing import Any

import pandas as pd

from .evaluate import compute_metrics
from .route_score import GA, JIN, apply_profile_to_scorecard, load_route_score_profile_from_config


def run_route_score_evolution(
    train_pred_df: pd.DataFrame,
    validation_pred_df: pd.DataFrame,
    config: Any,
) -> dict:
    """Evolve route-score profiles on train and validate on holdout.

    This is the scorecard analogue of policy candidates. It does not call the
    LLM again: it re-scores the saved route_scorecard with several candidate
    profiles, applies narrow post-hoc flips, and promotes only if validation
    improves against the record baseline.
    """
    current_profile = load_route_score_profile_from_config(config)
    train_base_metrics = compute_metrics(train_pred_df)
    validation_base_metrics = compute_metrics(validation_pred_df)
    candidates = build_profile_candidates(config, train_pred_df, current_profile)

    rows = []
    best = None
    for candidate in candidates:
        tuned_train = apply_profile_candidate(train_pred_df, candidate)
        train_metrics = compute_metrics(tuned_train)
        train_delta = metric_delta(train_base_metrics, train_metrics)
        train_ok, train_reason = passes_gate(train_base_metrics, train_metrics, config, prefix="route_score_evolve_train")

        tuned_validation = apply_profile_candidate(validation_pred_df, candidate)
        validation_metrics = compute_metrics(tuned_validation)
        validation_delta = metric_delta(validation_base_metrics, validation_metrics)
        validation_ok, validation_reason = passes_gate(
            validation_base_metrics,
            validation_metrics,
            config,
            prefix="route_score_evolve_validation",
        )

        row = {
            "name": candidate["name"],
            "description": candidate.get("description", ""),
            "train_gate": train_ok,
            "train_gate_reason": train_reason,
            "validation_gate": validation_ok,
            "validation_gate_reason": validation_reason,
            "train_accuracy": train_metrics.get("accuracy", 0),
            "train_score": train_metrics.get("score", 0),
            "train_true_recall": train_metrics.get("true_recall", 0),
            "train_true_precision": train_metrics.get("true_precision", 0),
            "train_false_recall": train_metrics.get("false_recall", 0),
            "train_false_precision": train_metrics.get("false_precision", 0),
            "train_flips_to_true": count_flips(tuned_train, JIN),
            "train_flips_to_false": count_flips(tuned_train, GA),
            "train_accuracy_delta": train_delta.get("accuracy", 0),
            "train_score_delta": train_delta.get("score", 0),
            "train_true_recall_delta": train_delta.get("true_recall", 0),
            "train_true_precision_delta": train_delta.get("true_precision", 0),
            "validation_accuracy": validation_metrics.get("accuracy", 0),
            "validation_score": validation_metrics.get("score", 0),
            "validation_true_recall": validation_metrics.get("true_recall", 0),
            "validation_true_precision": validation_metrics.get("true_precision", 0),
            "validation_false_recall": validation_metrics.get("false_recall", 0),
            "validation_false_precision": validation_metrics.get("false_precision", 0),
            "validation_flips_to_true": count_flips(tuned_validation, JIN),
            "validation_flips_to_false": count_flips(tuned_validation, GA),
            "validation_accuracy_delta": validation_delta.get("accuracy", 0),
            "validation_score_delta": validation_delta.get("score", 0),
            "validation_true_recall_delta": validation_delta.get("true_recall", 0),
            "validation_true_precision_delta": validation_delta.get("true_precision", 0),
        }
        rows.append(row)

        if train_ok and validation_ok and is_better_candidate(row, best):
            best = {
                "candidate": candidate,
                "train_metrics": train_metrics,
                "validation_metrics": validation_metrics,
                "train_delta": train_delta,
                "validation_delta": validation_delta,
                "train_predictions": tuned_train,
                "validation_predictions": tuned_validation,
                "summary_row": row,
            }

    if best is None:
        best = {
            "candidate": {
                "name": "route_profile_noop",
                "description": "no scorecard profile candidate passed train and validation gates",
                "profile": current_profile or {"name": "base"},
                "posthoc": {},
            },
            "train_metrics": train_base_metrics,
            "validation_metrics": validation_base_metrics,
            "train_delta": metric_delta(train_base_metrics, train_base_metrics),
            "validation_delta": metric_delta(validation_base_metrics, validation_base_metrics),
            "train_predictions": train_pred_df.copy(),
            "validation_predictions": validation_pred_df.copy(),
            "summary_row": {
                "name": "route_profile_noop",
                "description": "no scorecard profile candidate passed train and validation gates",
                "train_gate": False,
                "validation_gate": False,
                "train_gate_reason": "no candidate passed",
                "validation_gate_reason": "no candidate passed",
                "train_accuracy": train_base_metrics.get("accuracy", 0),
                "train_score": train_base_metrics.get("score", 0),
                "train_true_recall": train_base_metrics.get("true_recall", 0),
                "train_true_precision": train_base_metrics.get("true_precision", 0),
                "validation_accuracy": validation_base_metrics.get("accuracy", 0),
                "validation_score": validation_base_metrics.get("score", 0),
                "validation_true_recall": validation_base_metrics.get("true_recall", 0),
                "validation_true_precision": validation_base_metrics.get("true_precision", 0),
                "validation_flips_to_true": 0,
                "validation_flips_to_false": 0,
            },
        }

    return {
        "current_profile": current_profile,
        "train_base_metrics": train_base_metrics,
        "validation_base_metrics": validation_base_metrics,
        "candidates": rows,
        "best": best,
    }


def build_profile_candidates(config: Any, train_pred_df: pd.DataFrame, current_profile: dict) -> list[dict]:
    candidates = [
        make_candidate(
            "balanced_profile",
            "양방향 보정. 진성/가성 점수 차이가 충분히 큰 경우에만 뒤집는다.",
            current_profile,
            {
                "routes": {
                    "pipe_support": {
                        "true_signal_bonuses": [
                            {"contains": ["밟으면 안"], "score": 1, "reason": "배관/서포트 밟음 진성 신호 강화"},
                            {"contains": ["물리적 보강"], "score": 1, "reason": "보강/기술협의 진성 신호 강화"},
                        ],
                        "false_signal_bonuses": [
                            {"contains": ["계획"], "score": 1, "reason": "계획/허가성 가성 신호 강화"},
                        ],
                    },
                    "leak_contact": {
                        "true_signal_bonuses": [
                            {"contains": ["미상"], "score": 1, "reason": "성분 미상 진성 신호 강화"},
                            {"contains": ["전문 조치"], "score": 1, "reason": "ERT/전문조치 진성 신호 강화"},
                        ],
                        "false_signal_bonuses": [
                            {"contains": ["무해성"], "score": 1, "reason": "당시 무해성 명확 가성 신호 강화"},
                            {"contains": ["단순 청소"], "score": 1, "reason": "단순 청소/배수 가성 신호 강화"},
                        ],
                    },
                }
            },
            posthoc={
                "flip_true_routes": ["pipe_support", "leak_contact"],
                "min_true_score": 8,
                "min_true_margin": 3,
                "flip_false_routes": ["pipe_support", "leak_contact"],
                "min_false_score": 8,
                "min_false_margin": 3,
            },
        ),
        make_candidate(
            "pipe_true_recall_push",
            "배관/서포트 밟음과 발판 부족 FN 감소용. 가성 보정은 매우 엄격하게 둔다.",
            current_profile,
            {
                "routes": {
                    "pipe_support": {
                        "true_signal_bonuses": [
                            {"contains": ["밟으면 안"], "score": 2, "reason": "밟음 필요 내부기준 진성 강화"},
                            {"contains": ["발판"], "score": 1, "reason": "발판/이동경로 부족 강화"},
                            {"contains": ["물리적 보강"], "score": 2, "reason": "비계/보강판/기술협의 강화"},
                            {"contains": ["물리적 결과"], "score": 1, "reason": "추락/낙하/파손 결과위험 강화"},
                        ],
                        "false_signal_bonuses": [
                            {"contains": ["계획"], "score": 1, "reason": "작업예정/허가성은 가성 방어"},
                        ],
                        "true_threshold": 5,
                        "true_margin": 1,
                    }
                }
            },
            posthoc={
                "flip_true_routes": ["pipe_support"],
                "min_true_score": 7,
                "min_true_margin": 2,
                "flip_false_routes": ["pipe_support"],
                "min_false_score": 10,
                "min_false_margin": 5,
            },
        ),
        make_candidate(
            "leak_true_recall_push",
            "접액/누출 FN 감소용. 성분 미상, 작업자 노출, ERT/전문조치를 강하게 본다.",
            current_profile,
            {
                "routes": {
                    "leak_contact": {
                        "true_signal_bonuses": [
                            {"contains": ["미상"], "score": 2, "reason": "작업중지 당시 성분 미상 강화"},
                            {"contains": ["노출"], "score": 1, "reason": "작업자 노출 가능성 강화"},
                            {"contains": ["전문 조치"], "score": 2, "reason": "ERT/방제/격리 전문조치 강화"},
                            {"contains": ["2차 위험"], "score": 1, "reason": "화재/폭발/질식 등 2차 위험 강화"},
                        ],
                        "false_signal_bonuses": [
                            {"contains": ["당시 무해성"], "score": 2, "reason": "당시 무해성 명확 방어"},
                        ],
                        "true_threshold": 5,
                        "true_margin": 1,
                    }
                }
            },
            posthoc={
                "flip_true_routes": ["leak_contact"],
                "min_true_score": 7,
                "min_true_margin": 2,
                "flip_false_routes": ["leak_contact"],
                "min_false_score": 10,
                "min_false_margin": 5,
            },
        ),
        make_candidate(
            "precision_guard",
            "진성 과판정 방어용. 계획/허가/당시 무해성/단순시정 신호가 강한 경우만 가성으로 뒤집는다.",
            current_profile,
            {
                "routes": {
                    "pipe_support": {
                        "false_signal_bonuses": [
                            {"contains": ["계획"], "score": 2, "reason": "계획/허가/일정 조율 방어 강화"},
                            {"contains": ["사전"], "score": 1, "reason": "사전협의/점검 방어 강화"},
                            {"contains": ["단순 시정"], "score": 1, "reason": "단순 시정 방어 강화"},
                        ],
                        "false_threshold": 5,
                        "false_margin": 1,
                    },
                    "leak_contact": {
                        "false_signal_bonuses": [
                            {"contains": ["당시 무해성"], "score": 2, "reason": "당시 DIW/응축수/테스트 명확 방어 강화"},
                            {"contains": ["활성 위험 없음"], "score": 1, "reason": "실제 노출/활성위험 없음 방어 강화"},
                            {"contains": ["단순 청소"], "score": 1, "reason": "단순 청소/배수 방어 강화"},
                        ],
                        "false_threshold": 5,
                        "false_margin": 1,
                    },
                }
            },
            posthoc={
                "flip_true_routes": ["pipe_support", "leak_contact"],
                "min_true_score": 999,
                "min_true_margin": 999,
                "flip_false_routes": ["pipe_support", "leak_contact"],
                "min_false_score": 7,
                "min_false_margin": 2,
            },
        ),
        make_candidate(
            "hybrid_recall_with_guard",
            "배관/접액 진성 회수율을 올리되, 명확한 계획/무해성 가드레일은 유지한다.",
            current_profile,
            {
                "routes": {
                    "pipe_support": {
                        "true_signal_bonuses": [
                            {"contains": ["밟으면 안"], "score": 2, "reason": "밟음 필요 강화"},
                            {"contains": ["물리적 보강"], "score": 1, "reason": "보강/협의 강화"},
                        ],
                        "false_signal_bonuses": [
                            {"contains": ["계획"], "score": 1, "reason": "계획/허가 방어"},
                        ],
                    },
                    "leak_contact": {
                        "true_signal_bonuses": [
                            {"contains": ["미상"], "score": 2, "reason": "성분 미상 강화"},
                            {"contains": ["전문 조치"], "score": 1, "reason": "전문조치 강화"},
                        ],
                        "false_signal_bonuses": [
                            {"contains": ["당시 무해성"], "score": 2, "reason": "당시 무해성 방어"},
                        ],
                    },
                }
            },
            posthoc={
                "flip_true_routes": ["pipe_support", "leak_contact"],
                "min_true_score": 8,
                "min_true_margin": 2,
                "flip_false_routes": ["pipe_support", "leak_contact"],
                "min_false_score": 8,
                "min_false_margin": 3,
            },
        ),
    ]

    dynamic = make_dynamic_candidate(train_pred_df, current_profile)
    if dynamic:
        candidates.append(dynamic)
    return candidates


def make_dynamic_candidate(train_pred_df: pd.DataFrame, current_profile: dict) -> dict | None:
    excluded = train_pred_df.get("exclude_from_metrics")
    if excluded is None:
        excluded = pd.Series([False] * len(train_pred_df), index=train_pred_df.index)
    eval_df = train_pred_df[~excluded.map(normalize_bool)].copy()
    if eval_df.empty or "route_primary" not in eval_df.columns:
        return None

    fn = eval_df[(eval_df["label"] == JIN) & (eval_df["pred"] == GA)]
    fp = eval_df[(eval_df["label"] == GA) & (eval_df["pred"] == JIN)]
    pipe_fn = int((fn.get("route_primary", "") == "pipe_support").sum())
    leak_fn = int((fn.get("route_primary", "") == "leak_contact").sum())
    pipe_fp = int((fp.get("route_primary", "") == "pipe_support").sum())
    leak_fp = int((fp.get("route_primary", "") == "leak_contact").sum())

    routes: dict[str, Any] = {}
    posthoc = {
        "flip_true_routes": [],
        "min_true_score": 8,
        "min_true_margin": 2,
        "flip_false_routes": [],
        "min_false_score": 8,
        "min_false_margin": 3,
    }
    if pipe_fn >= max(5, pipe_fp):
        routes["pipe_support"] = {
            "true_signal_bonuses": [
                {"contains": ["밟으면 안"], "score": 2, "reason": "train FN 우세: 배관/서포트 밟음 강화"},
                {"contains": ["물리적 보강"], "score": 1, "reason": "train FN 우세: 보강/협의 강화"},
            ],
        }
        posthoc["flip_true_routes"].append("pipe_support")
    elif pipe_fp >= 5:
        routes["pipe_support"] = {
            "false_signal_bonuses": [
                {"contains": ["계획"], "score": 2, "reason": "train FP 우세: 계획/허가 방어 강화"},
                {"contains": ["단순 시정"], "score": 1, "reason": "train FP 우세: 단순 시정 방어 강화"},
            ],
        }
        posthoc["flip_false_routes"].append("pipe_support")

    if leak_fn >= max(5, leak_fp):
        routes["leak_contact"] = {
            "true_signal_bonuses": [
                {"contains": ["미상"], "score": 2, "reason": "train FN 우세: 성분 미상 강화"},
                {"contains": ["전문 조치"], "score": 1, "reason": "train FN 우세: 전문조치 강화"},
            ],
        }
        posthoc["flip_true_routes"].append("leak_contact")
    elif leak_fp >= 5:
        routes["leak_contact"] = {
            "false_signal_bonuses": [
                {"contains": ["당시 무해성"], "score": 2, "reason": "train FP 우세: 당시 무해성 방어 강화"},
                {"contains": ["단순 청소"], "score": 1, "reason": "train FP 우세: 단순 청소 방어 강화"},
            ],
        }
        posthoc["flip_false_routes"].append("leak_contact")

    if not routes:
        return None
    return make_candidate(
        "dynamic_error_balance",
        f"train 오류 분포 기반 자동 후보: pipe_fn={pipe_fn}, pipe_fp={pipe_fp}, leak_fn={leak_fn}, leak_fp={leak_fp}",
        current_profile,
        {"routes": routes},
        posthoc=posthoc,
    )


def make_candidate(
    name: str,
    description: str,
    current_profile: dict,
    profile_delta: dict,
    posthoc: dict,
) -> dict:
    profile = merge_profile(current_profile or {"name": "base"}, profile_delta)
    profile["name"] = name
    return {
        "name": name,
        "description": description,
        "profile": profile,
        "posthoc": posthoc,
    }


def apply_profile_candidate(pred_df: pd.DataFrame, candidate: dict) -> pd.DataFrame:
    tuned = pred_df.copy(deep=True)
    if tuned.empty:
        tuned["route_profile_applied"] = False
        return tuned

    ensure_column(tuned, "reason", "")
    ensure_column(tuned, "applied_step", "")
    ensure_column(tuned, "route_profile_original_pred", "")

    applied = []
    reasons = []
    updated_scorecards = []
    route_recommendations = []
    route_primaries = []
    route_true_scores = []
    route_false_scores = []
    new_preds = []

    for _, row in tuned.iterrows():
        scorecard = parse_scorecard(row.get("route_scorecard", {}))
        adjusted = apply_profile_to_scorecard(scorecard, candidate.get("profile", {}))
        updated_scorecards.append(adjusted)
        route_recommendations.append(adjusted.get("recommendation", ""))
        route_primaries.append(adjusted.get("primary_route", ""))
        route_true_scores.append(int(adjusted.get("true_score", 0)))
        route_false_scores.append(int(adjusted.get("false_score", 0)))

        new_pred, reason = maybe_flip_prediction(row, adjusted, candidate)
        was_applied = new_pred != row.get("pred", "")
        new_preds.append(new_pred)
        applied.append(was_applied)
        reasons.append(reason if was_applied else "")

    tuned["route_scorecard"] = updated_scorecards
    tuned["route_recommendation"] = route_recommendations
    tuned["route_primary"] = route_primaries
    tuned["route_true_score"] = route_true_scores
    tuned["route_false_score"] = route_false_scores
    tuned["route_profile_applied"] = applied
    tuned["route_profile_reason"] = reasons
    mask = tuned["route_profile_applied"].map(bool)
    if mask.any():
        tuned.loc[mask, "route_profile_original_pred"] = tuned.loc[mask, "pred"]
        tuned["pred"] = new_preds
        tuned.loc[mask, "applied_step"] = "route_score_profile"
        tuned.loc[mask, "reason"] = tuned.loc[mask, "reason"].astype(str) + " / " + tuned.loc[mask, "route_profile_reason"].astype(str)
        tuned.loc[mask, "correct"] = tuned.loc[mask, "label"] == tuned.loc[mask, "pred"]
    return tuned


def maybe_flip_prediction(row: pd.Series, scorecard: dict, candidate: dict) -> tuple[str, str]:
    pred = str(row.get("pred", ""))
    if normalize_bool(row.get("exclude_from_metrics", False)):
        return pred, ""

    posthoc = candidate.get("posthoc", {})
    route = str(scorecard.get("primary_route", ""))
    true_score = int(scorecard.get("true_score", 0))
    false_score = int(scorecard.get("false_score", 0))
    recommendation = str(scorecard.get("recommendation", ""))

    if (
        pred == GA
        and recommendation == JIN
        and route in set(posthoc.get("flip_true_routes", []))
        and true_score >= int(posthoc.get("min_true_score", 8))
        and true_score - false_score >= int(posthoc.get("min_true_margin", 3))
    ):
        return (
            JIN,
            f"route_profile {candidate['name']}: {route} true={true_score} false={false_score} 기준으로 진성 보정",
        )

    if (
        pred == JIN
        and recommendation == GA
        and route in set(posthoc.get("flip_false_routes", []))
        and false_score >= int(posthoc.get("min_false_score", 8))
        and false_score - true_score >= int(posthoc.get("min_false_margin", 3))
    ):
        return (
            GA,
            f"route_profile {candidate['name']}: {route} false={false_score} true={true_score} 기준으로 가성 보정",
        )

    return pred, ""


def save_promoted_profile(config: Any, profile: dict, metadata: dict) -> Path:
    profile_path = Path(config.get("runtime", "route_score_profile_path", fallback="artifacts/route_score_profile.json"))
    profile_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "profile": profile,
        "metadata": {
            **metadata,
            "promoted_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        },
    }
    profile_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return profile_path


def passes_gate(base_metrics: dict, metrics: dict, config: Any, prefix: str) -> tuple[bool, str]:
    min_score_gain = config.getfloat("runtime", f"{prefix}_min_score_gain", fallback=0.0)
    min_accuracy_gain = config.getfloat("runtime", f"{prefix}_min_accuracy_gain", fallback=0.0)
    max_tr_loss = config.getfloat("runtime", f"{prefix}_max_true_recall_loss", fallback=0.03)
    max_tp_loss = config.getfloat("runtime", f"{prefix}_max_true_precision_loss", fallback=0.03)

    score_gain = metrics.get("score", 0) - base_metrics.get("score", 0)
    accuracy_gain = metrics.get("accuracy", 0) - base_metrics.get("accuracy", 0)
    tr_loss = base_metrics.get("true_recall", 0) - metrics.get("true_recall", 0)
    tp_loss = base_metrics.get("true_precision", 0) - metrics.get("true_precision", 0)

    if score_gain < min_score_gain:
        return False, f"score_gain {score_gain:.4f} < {min_score_gain:.4f}"
    if accuracy_gain < min_accuracy_gain:
        return False, f"accuracy_gain {accuracy_gain:.4f} < {min_accuracy_gain:.4f}"
    if tr_loss > max_tr_loss:
        return False, f"true_recall_loss {tr_loss:.4f} > {max_tr_loss:.4f}"
    if tp_loss > max_tp_loss:
        return False, f"true_precision_loss {tp_loss:.4f} > {max_tp_loss:.4f}"
    return True, (
        f"accepted: score_gain={score_gain:.4f}, accuracy_gain={accuracy_gain:.4f}, "
        f"true_recall_loss={tr_loss:.4f}, true_precision_loss={tp_loss:.4f}"
    )


def is_better_candidate(row: dict, best: dict | None) -> bool:
    if best is None:
        return True
    return candidate_sort_key(row) > candidate_sort_key(best["summary_row"])


def candidate_sort_key(row: dict) -> tuple:
    return (
        float(row.get("validation_score_delta", 0)),
        float(row.get("validation_accuracy_delta", 0)),
        float(row.get("validation_true_recall_delta", 0)),
        float(row.get("validation_true_precision_delta", 0)),
        float(row.get("train_score_delta", 0)),
    )


def metric_delta(base: dict, current: dict) -> dict:
    keys = [
        "accuracy",
        "score",
        "true_recall",
        "true_precision",
        "false_recall",
        "false_precision",
        "fn_true_as_false",
        "fp_false_as_true",
    ]
    return {key: current.get(key, 0) - base.get(key, 0) for key in keys}


def count_flips(df: pd.DataFrame, target_pred: str) -> int:
    if "route_profile_applied" not in df.columns:
        return 0
    mask = df["route_profile_applied"].map(normalize_bool) & (df["pred"] == target_pred)
    return int(mask.sum())


def parse_scorecard(value: Any) -> dict:
    if isinstance(value, dict):
        return value
    if value is None:
        return {}
    try:
        if pd.isna(value):
            return {}
    except Exception:
        pass
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return {}
        try:
            parsed = json.loads(text)
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            return {}
    return {}


def ensure_column(df: pd.DataFrame, column: str, default: Any) -> None:
    if column not in df.columns:
        df[column] = default


def normalize_bool(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n", "", "nan", "none", "null"}:
        return False
    return False


def merge_profile(base: dict, delta: dict) -> dict:
    result = deepcopy(base or {})
    deep_merge(result, delta or {})
    return result


def deep_merge(target: dict, source: dict) -> None:
    for key, value in source.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            deep_merge(target[key], value)
        else:
            target[key] = deepcopy(value)
