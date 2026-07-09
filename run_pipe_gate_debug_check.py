from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd


JIN = "진성"
GA = "가성"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Directly verify pipe gate output by comparing pred_before_pipe_gate vs pred."
    )
    parser.add_argument(
        "--gated",
        required=True,
        help="Path to balanced_gated_predictions.csv, strict_gated_predictions.csv, or approval_only_gated_predictions.csv.",
    )
    parser.add_argument("--out", default="", help="Optional CSV path for debug summary.")
    args = parser.parse_args()

    path = Path(args.gated)
    df = pd.read_csv(path)
    summary = debug_summary(df)

    print(f"[debug] file={path}", flush=True)
    for row in summary:
        print(f"{row['metric']}: {row['value']}", flush=True)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(summary).to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"[debug] saved={out_path}", flush=True)


def debug_summary(df: pd.DataFrame) -> list[dict[str, Any]]:
    required = {"label", "pred", "pred_before_pipe_gate"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    work = df.copy()
    work["_label"] = work["label"].map(norm_label)
    work["_before"] = work["pred_before_pipe_gate"].map(norm_label)
    work["_after"] = work["pred"].map(norm_label)
    work["_excluded"] = work.get("exclude_from_metrics", False)
    if isinstance(work["_excluded"], pd.Series):
        work["_excluded"] = work["_excluded"].map(norm_bool)
    else:
        work["_excluded"] = False

    eval_mask = (
        ~work["_excluded"]
        & work["_label"].isin({JIN, GA})
        & work["_before"].isin({JIN, GA})
        & work["_after"].isin({JIN, GA})
    )
    changed_mask = work["_before"] != work["_after"]
    changed_eval = work[eval_mask & changed_mask].copy()

    rows: list[dict[str, Any]] = []
    add = lambda metric, value: rows.append({"metric": metric, "value": value})

    add("total_rows", len(work))
    add("excluded_rows", int(work["_excluded"].sum()))
    add("valid_label_rows", int(work["_label"].isin({JIN, GA}).sum()))
    add("valid_before_rows", int(work["_before"].isin({JIN, GA}).sum()))
    add("valid_after_rows", int(work["_after"].isin({JIN, GA}).sum()))
    add("eval_rows", int(eval_mask.sum()))
    add("changed_rows_all", int(changed_mask.sum()))
    add("changed_rows_eval", int((eval_mask & changed_mask).sum()))
    add("changed_eval_actual_false", int((changed_eval["_label"] == GA).sum()))
    add("changed_eval_actual_true", int((changed_eval["_label"] == JIN).sum()))
    add(
        "changed_eval_precision_actual_false",
        round(div(int((changed_eval["_label"] == GA).sum()), len(changed_eval)), 6),
    )

    before = confusion(work[eval_mask], "_before")
    after = confusion(work[eval_mask], "_after")
    for prefix, metrics in [("before", before), ("after", after)]:
        for key, value in metrics.items():
            add(f"{prefix}_{key}", value)
    for key in ["accuracy", "TP", "TR", "FR", "FP", "fp_false_as_true", "fn_true_as_false", "ai_true_count"]:
        add(f"delta_{key}", after[key] - before[key])

    if "pipe_gate_applied" in work.columns:
        applied = work["pipe_gate_applied"].map(norm_bool)
        add("pipe_gate_applied_rows_all", int(applied.sum()))
        add("pipe_gate_applied_rows_eval", int((applied & eval_mask).sum()))

    if "pipe_gate_reason" in work.columns:
        for reason, count in work.loc[changed_mask, "pipe_gate_reason"].value_counts(dropna=False).head(20).items():
            add(f"changed_reason={reason}", int(count))

    for prefix, col in [
        ("raw_label", "label"),
        ("raw_before", "pred_before_pipe_gate"),
        ("raw_after", "pred"),
        ("norm_label", "_label"),
        ("norm_before", "_before"),
        ("norm_after", "_after"),
        ("exclude_from_metrics", "_excluded"),
    ]:
        for value, count in work[col].astype(str).value_counts(dropna=False).head(12).items():
            add(f"{prefix}={value}", int(count))

    return rows


def confusion(df: pd.DataFrame, pred_col: str) -> dict[str, Any]:
    labels = df["_label"]
    preds = df[pred_col]
    tp = int(((labels == JIN) & (preds == JIN)).sum())
    fn = int(((labels == JIN) & (preds == GA)).sum())
    fp = int(((labels == GA) & (preds == JIN)).sum())
    tn = int(((labels == GA) & (preds == GA)).sum())
    n = len(df)
    return {
        "n": n,
        "accuracy": div(tp + tn, n),
        "TP": div(tp, tp + fp),
        "TR": div(tp, tp + fn),
        "FR": div(tn, tn + fp),
        "FP": div(tn, tn + fn),
        "tp_true": tp,
        "fn_true_as_false": fn,
        "fp_false_as_true": fp,
        "tn_false": tn,
        "ai_true_count": int((preds == JIN).sum()),
        "ai_false_count": int((preds == GA).sum()),
    }


def norm_label(value: Any) -> str:
    text = "" if value is None else str(value).strip()
    if text in {JIN, GA}:
        return text
    if "진성" in text or "진" in text or "\uf9de" in text:
        return JIN
    if "가성" in text or "가" in text or "\u5a9b" in text:
        return GA
    return text


def norm_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    text = str(value).strip().lower()
    compact = text.replace(" ", "")
    true_values = {"true", "1", "yes", "y", "t", "예", "네", "맞음", "있음", "해당"}
    false_values = {
        "false",
        "0",
        "no",
        "n",
        "f",
        "",
        "아니오",
        "아니요",
        "아님",
        "없음",
        "해당없음",
        "해당없슴",
        "없슴",
        "nan",
        "none",
        "null",
    }
    if compact in true_values:
        return True
    if compact in false_values:
        return False
    return bool(value)


def div(a: int | float, b: int | float) -> float:
    return float(a / b) if b else 0.0


if __name__ == "__main__":
    main()
