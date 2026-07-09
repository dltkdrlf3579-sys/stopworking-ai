from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd

from stopright_ai.pipe_support_gate import (
    build_gate_diagnostics,
    build_gate_diagnostics_summary,
    compute_metrics,
    load_prediction_files,
    simulate_pipe_support_gate,
    summarize_gate,
)


PROFILES = ["strict", "balanced", "approval_only"]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Simulate deterministic pipe/support false gates on saved prediction CSV files without LLM calls."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Prediction CSV or run/output directory containing train_baseline_predictions.csv etc.",
    )
    parser.add_argument(
        "--out-dir",
        default="",
        help="Output directory. Default: artifacts/pipe_support_gate/<timestamp>",
    )
    parser.add_argument(
        "--include-candidates",
        action="store_true",
        help="Also include candidate-folder predictions.csv files. Default excludes candidates.",
    )
    parser.add_argument(
        "--save-gated-predictions",
        action="store_true",
        help="Save full gated prediction CSV for each profile. Flip rows are always saved.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    out_dir = Path(args.out_dir) if args.out_dir else Path("artifacts") / "pipe_support_gate" / datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)

    base_df = load_prediction_files(input_path, include_candidates=args.include_candidates)
    base_metrics = compute_metrics(base_df)
    diagnostics = build_gate_diagnostics(base_df)
    diagnostics_summary = build_gate_diagnostics_summary(diagnostics)
    diagnostics.to_csv(out_dir / "pipe_support_gate_diagnostics_rows.csv", index=False, encoding="utf-8-sig")
    diagnostics_summary.to_csv(out_dir / "pipe_support_gate_diagnostics_summary.csv", index=False, encoding="utf-8-sig")

    summary_rows = []
    profile_outputs = {}
    for profile in PROFILES:
        gated = simulate_pipe_support_gate(base_df, profile=profile)
        gated_metrics = compute_metrics(gated)
        row = summarize_gate(base_df, gated, profile)
        row["n"] = gated_metrics.get("n", 0)
        summary_rows.append(row)
        profile_outputs[profile] = gated

        flips = gated[gated["pipe_gate_applied"].map(bool)].copy()
        flips.to_csv(out_dir / f"{profile}_flips.csv", index=False, encoding="utf-8-sig")
        if args.save_gated_predictions:
            gated.to_csv(out_dir / f"{profile}_gated_predictions.csv", index=False, encoding="utf-8-sig")

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out_dir / "pipe_support_gate_metrics.csv", index=False, encoding="utf-8-sig")

    report = build_report(input_path, base_metrics, summary)
    (out_dir / "pipe_support_gate_report.md").write_text(report, encoding="utf-8")

    print(f"[pipe_gate] input={input_path}", flush=True)
    print(f"[pipe_gate] out_dir={out_dir}", flush=True)
    print("[pipe_gate] baseline", format_metrics(base_metrics), flush=True)
    print(summary.to_string(index=False), flush=True)
    if not diagnostics_summary.empty:
        print("[pipe_gate] diagnostics summary", flush=True)
        print(diagnostics_summary.head(40).to_string(index=False), flush=True)


def build_report(input_path: Path, base_metrics: dict, summary: pd.DataFrame) -> str:
    lines = [
        "# Pipe/Support Gate Simulation",
        "",
        f"- Input: `{input_path}`",
        "- LLM calls: none",
        "- Purpose: simulate `진성 -> 가성` correction only for pipe/support over-prediction cases.",
        "",
        "## Baseline",
        "",
        metrics_bullets(base_metrics),
        "",
        "## Gate Profiles",
        "",
        "- `strict`: flips only when pipe/support evidence has no positive physical signal.",
        "- `balanced`: additionally flips prework/admin pipe cases with no force/countermeasure/approval-positive evidence.",
        "- `approval_only`: targets approval-only cases without phase/force/countermeasure support.",
        "",
        "## Metrics",
        "",
    ]
    lines.extend(markdown_table(summary))
    lines.extend(
        [
            "",
            "## How To Choose",
            "",
            "- Prefer the profile where `TP_after` rises meaningfully while `TR_after` stays acceptable.",
            "- `flip_precision` means: among rows flipped from `진성` to `가성`, how many were actually `가성`.",
            "- A useful gate should usually have `flip_precision >= 0.75`.",
            "- If `TR_delta` is too negative, the gate is catching too many real `진성` cases.",
            "- If `flips` is tiny, the gate is too weak to solve the operating issue.",
            "",
            "## Next Step",
            "",
            "Open the best profile's `*_flips.csv` and inspect a few `flip_bad_true_to_false` cases if any.",
        ]
    )
    return "\n".join(lines)


def metrics_bullets(metrics: dict) -> str:
    return "\n".join(
        [
            f"- n: {metrics.get('n', 0)}",
            f"- accuracy: {metrics.get('accuracy', 0):.4f}",
            f"- TP(true precision): {metrics.get('true_precision', 0):.4f}",
            f"- TR(true recall): {metrics.get('true_recall', 0):.4f}",
            f"- FPerr(false as true): {metrics.get('fp_false_as_true', 0)}",
            f"- FN(true as false): {metrics.get('fn_true_as_false', 0)}",
            f"- AI 진성 count: {metrics.get('ai_true_count', 0)}",
        ]
    )


def markdown_table(df: pd.DataFrame) -> list[str]:
    if df.empty:
        return ["_No rows._"]
    cols = [
        "profile",
        "flips",
        "changed_pred_rows",
        "changed_eval_rows",
        "eval_flips",
        "flip_precision",
        "eval_flip_precision",
        "accuracy_after",
        "accuracy_delta",
        "TP_after",
        "TP_delta",
        "TR_after",
        "TR_delta",
        "FPerr_after",
        "FPerr_delta",
        "FN_after",
        "FN_delta",
        "ai_true_after",
        "ai_true_delta",
    ]
    cols = [col for col in cols if col in df.columns]
    lines = ["|" + "|".join(cols) + "|", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, row in df[cols].iterrows():
        values = []
        for col in cols:
            value = row[col]
            if isinstance(value, float):
                values.append(f"{value:.4f}")
            else:
                values.append(str(value))
        lines.append("|" + "|".join(values) + "|")
    return lines


def format_metrics(metrics: dict) -> str:
    return (
        f"total_n={metrics.get('total_n', metrics.get('n', 0))} "
        f"eval_n={metrics.get('n', 0)} "
        f"acc={metrics.get('accuracy', 0):.4f} "
        f"TP={metrics.get('true_precision', 0):.4f} "
        f"TR={metrics.get('true_recall', 0):.4f} "
        f"FPerr={metrics.get('fp_false_as_true', 0)} "
        f"FN={metrics.get('fn_true_as_false', 0)}"
    )


if __name__ == "__main__":
    main()
