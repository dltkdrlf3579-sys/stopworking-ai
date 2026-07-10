from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Any

import pandas as pd

from stopright_ai.evaluate import compute_metrics


JIN = "진성"
GA = "가성"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Diagnose true_confirm vote patterns from prediction CSV files."
    )
    parser.add_argument(
        "input",
        nargs="?",
        default="outputs",
        help="Prediction CSV file or output folder. Defaults to outputs.",
    )
    parser.add_argument(
        "--prefix",
        default="",
        help="Optional filename prefix filter, e.g. train_baseline or validation_baseline.",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/vote_diagnostics",
        help="Folder for diagnostics output.",
    )
    args = parser.parse_args()

    files = discover_files(Path(args.input), args.prefix)
    if not files:
        raise FileNotFoundError(f"No prediction CSV files found under {args.input!r}")

    df = load_files(files)
    df = normalize(df)
    out_dir = Path(args.output_dir) / latest_name(Path(args.input))
    out_dir.mkdir(parents=True, exist_ok=True)

    write_outputs(df, files, out_dir)
    print(f"[vote-diagnostics] files={len(files)} rows={len(df)}")
    print(f"[vote-diagnostics] output={out_dir}")


def discover_files(path: Path, prefix: str) -> list[Path]:
    if path.is_file():
        return [path]

    candidates = sorted(path.rglob("*predictions.csv"))
    if prefix:
        candidates = [p for p in candidates if p.name.startswith(prefix)]

    # Avoid candidate policy folders unless explicitly passed as a file.
    return [
        p
        for p in candidates
        if "candidate_" not in str(p).lower()
        and p.name
        in {
            "train_baseline_predictions.csv",
            "validation_baseline_predictions.csv",
            "train_predictions.csv",
            "validation_predictions.csv",
            "predictions.csv",
        }
    ]


def load_files(files: list[Path]) -> pd.DataFrame:
    frames = []
    for order, path in enumerate(files, start=1):
        frame = pd.read_csv(path, encoding="utf-8-sig")
        frame["source_file"] = str(path)
        frame["source_order"] = order
        frames.append(frame)
    return pd.concat(frames, ignore_index=True, sort=False)


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    work["label"] = work.get("label", "").map(normalize_label)
    work["pred"] = work.get("pred", "").map(normalize_label)

    if "exclude_from_metrics" not in work.columns:
        work["exclude_from_metrics"] = False
    work["exclude_from_metrics"] = work["exclude_from_metrics"].map(normalize_bool)

    work["vote_list"] = work.get("vote_results", "").map(parse_vote_results)
    work["vote_pattern"] = work["vote_list"].map(lambda xs: "|".join(xs) if xs else "no_vote")
    work["vote_true_count_calc"] = work["vote_list"].map(lambda xs: xs.count(JIN))
    work["vote_false_count_calc"] = work["vote_list"].map(lambda xs: xs.count(GA))
    work["vote_rounds_calc"] = work["vote_list"].map(len)

    work["pred_any_true"] = work["vote_true_count_calc"].map(lambda n: JIN if n >= 1 else GA)
    work["pred_all_true"] = work.apply(
        lambda r: JIN if r["vote_rounds_calc"] > 0 and r["vote_true_count_calc"] == r["vote_rounds_calc"] else GA,
        axis=1,
    )
    work["pred_two_or_more_true"] = work["vote_true_count_calc"].map(lambda n: JIN if n >= 2 else GA)
    return work


def parse_vote_results(value: Any) -> list[str]:
    if isinstance(value, list):
        return [normalize_label(v) for v in value if normalize_label(v) in {JIN, GA}]
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    text = str(value).strip()
    if not text:
        return []

    parsed: Any
    for loader in (json.loads, ast.literal_eval):
        try:
            parsed = loader(text)
            if isinstance(parsed, list):
                return [normalize_label(v) for v in parsed if normalize_label(v) in {JIN, GA}]
        except Exception:
            pass

    if "|" in text:
        return [normalize_label(v) for v in text.split("|") if normalize_label(v) in {JIN, GA}]
    if "," in text:
        return [normalize_label(v) for v in text.split(",") if normalize_label(v) in {JIN, GA}]
    label = normalize_label(text)
    return [label] if label in {JIN, GA} else []


def write_outputs(df: pd.DataFrame, files: list[Path], out_dir: Path) -> None:
    eval_df = df[~df["exclude_from_metrics"]].copy()
    metrics_rows = []

    for name, pred_col in [
        ("saved_current", "pred"),
        ("sim_any_true", "pred_any_true"),
        ("sim_all_true", "pred_all_true"),
        ("sim_two_or_more_true", "pred_two_or_more_true"),
    ]:
        metrics_input = eval_df.copy()
        metrics_input["pred"] = metrics_input[pred_col]
        metrics = compute_metrics(metrics_input)
        metrics["rule"] = name
        metrics_rows.append(metrics)

    metrics_df = pd.DataFrame(metrics_rows)
    metric_cols = [
        "rule",
        "n",
        "accuracy",
        "true_recall",
        "true_precision",
        "false_recall",
        "false_precision",
        "fn_true_as_false",
        "fp_false_as_true",
        "tp_true",
        "tn_false",
        "score",
    ]
    metrics_df = metrics_df[[c for c in metric_cols if c in metrics_df.columns]]
    metrics_df.to_csv(out_dir / "vote_rule_metrics.csv", index=False, encoding="utf-8-sig")

    buckets = []
    for pattern, group in eval_df.groupby("vote_pattern", dropna=False):
        metrics = compute_metrics(group)
        metrics["vote_pattern"] = pattern
        metrics["rows"] = len(group)
        metrics["label_true"] = int((group["label"] == JIN).sum())
        metrics["label_false"] = int((group["label"] == GA).sum())
        metrics["saved_pred_true"] = int((group["pred"] == JIN).sum())
        buckets.append(metrics)
    bucket_df = pd.DataFrame(buckets).sort_values("rows", ascending=False)
    bucket_cols = [
        "vote_pattern",
        "rows",
        "label_true",
        "label_false",
        "saved_pred_true",
        "accuracy",
        "true_recall",
        "true_precision",
        "fn_true_as_false",
        "fp_false_as_true",
    ]
    bucket_df[[c for c in bucket_cols if c in bucket_df.columns]].to_csv(
        out_dir / "vote_pattern_metrics.csv", index=False, encoding="utf-8-sig"
    )

    recoverable_fn = eval_df[
        (eval_df["label"] == JIN)
        & (eval_df["pred"] == GA)
        & (eval_df["vote_true_count_calc"] >= 1)
    ].copy()
    hard_fn = eval_df[
        (eval_df["label"] == JIN)
        & (eval_df["pred"] == GA)
        & (eval_df["vote_true_count_calc"] == 0)
    ].copy()
    hard_fp = eval_df[
        (eval_df["label"] == GA)
        & (eval_df["pred"] == JIN)
        & (eval_df["vote_true_count_calc"] >= 2)
    ].copy()

    keep = [
        "id",
        "label",
        "pred",
        "confidence",
        "vote_pattern",
        "vote_true_count_calc",
        "major",
        "middle",
        "title",
        "route_primary",
        "pipe_support_subtype",
        "leak_contact_subtype",
        "reason",
        "decisive_evidence",
        "source_file",
    ]
    save_slice(recoverable_fn, out_dir / "recoverable_fn_one_true.csv", keep)
    save_slice(hard_fn, out_dir / "hard_fn_zero_true.csv", keep)
    save_slice(hard_fp, out_dir / "hard_fp_all_true.csv", keep)
    save_error_clusters(
        [
            ("recoverable_fn_one_true", recoverable_fn),
            ("hard_fn_zero_true", hard_fn),
            ("hard_fp_all_true", hard_fp),
        ],
        out_dir / "vote_error_clusters.csv",
    )
    write_prompt_brief(
        [
            ("recoverable_fn_one_true", recoverable_fn),
            ("hard_fn_zero_true", hard_fn),
            ("hard_fp_all_true", hard_fp),
        ],
        out_dir / "vote_prompt_brief.md",
    )

    summary = {
        "files": [str(p) for p in files],
        "rows": int(len(df)),
        "eval_rows": int(len(eval_df)),
        "recoverable_fn_one_true": int(len(recoverable_fn)),
        "hard_fn_zero_true": int(len(hard_fn)),
        "hard_fp_all_true": int(len(hard_fp)),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def save_slice(df: pd.DataFrame, path: Path, keep: list[str]) -> None:
    cols = [c for c in keep if c in df.columns]
    df[cols].to_csv(path, index=False, encoding="utf-8-sig")


def save_error_clusters(groups: list[tuple[str, pd.DataFrame]], path: Path) -> None:
    rows: list[dict[str, Any]] = []
    cluster_cols = [
        "route_primary",
        "pipe_support_subtype",
        "leak_contact_subtype",
        "major",
        "middle",
        "vote_pattern",
    ]
    for error_type, frame in groups:
        if frame.empty:
            continue
        available = [c for c in cluster_cols if c in frame.columns]
        if not available:
            rows.append({"error_type": error_type, "count": len(frame)})
            continue
        grouped = frame.groupby(available, dropna=False)
        for keys, group in grouped:
            if not isinstance(keys, tuple):
                keys = (keys,)
            row = {"error_type": error_type, "count": int(len(group))}
            row.update({col: clean_value(value) for col, value in zip(available, keys)})
            row["avg_confidence"] = safe_mean(group.get("confidence"))
            row["top_terms"] = top_terms(group)
            row["sample_titles"] = " / ".join(
                str(v)[:80] for v in group.get("title", pd.Series(dtype=str)).dropna().head(3).tolist()
            )
            rows.append(row)
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["error_type", "count"], ascending=[True, False])
    out.to_csv(path, index=False, encoding="utf-8-sig")


def write_prompt_brief(groups: list[tuple[str, pd.DataFrame]], path: Path) -> None:
    lines = [
        "# Vote Diagnostics Prompt Brief",
        "",
        "이 파일은 true_confirm 결과에서 강한 오답만 압축한 요약입니다.",
        "",
    ]
    for error_type, frame in groups:
        lines.append(f"## {error_type}")
        lines.append(f"- rows: {len(frame)}")
        if frame.empty:
            lines.append("")
            continue
        for col in ["route_primary", "pipe_support_subtype", "leak_contact_subtype", "major", "middle", "vote_pattern"]:
            if col in frame.columns:
                lines.append(f"- {col}: {value_counts_text(frame[col])}")
        lines.append(f"- top_terms: {top_terms(frame)}")
        sample_cols = [c for c in ["id", "label", "pred", "vote_pattern", "major", "middle", "title"] if c in frame.columns]
        if sample_cols:
            lines.append("")
            lines.append("|" + "|".join(sample_cols) + "|")
            lines.append("|" + "|".join(["---"] * len(sample_cols)) + "|")
            for _, row in frame.head(8).iterrows():
                lines.append("|" + "|".join(escape_md(clean_value(row.get(c, ""))) for c in sample_cols) + "|")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def safe_mean(series: Any) -> float:
    if not isinstance(series, pd.Series) or series.empty:
        return 0.0
    nums = pd.to_numeric(series, errors="coerce").dropna()
    return round(float(nums.mean()), 2) if not nums.empty else 0.0


def top_terms(frame: pd.DataFrame, limit: int = 12) -> str:
    text_cols = [
        "title",
        "major",
        "middle",
        "reason",
        "decisive_evidence",
        "route_primary",
        "pipe_support_subtype",
        "leak_contact_subtype",
    ]
    text = " ".join(
        str(value)
        for col in text_cols
        if col in frame.columns
        for value in frame[col].dropna().tolist()
    )
    tokens = []
    for raw in text.replace("[", " ").replace("]", " ").replace(",", " ").replace("|", " ").split():
        token = raw.strip("'\"(){}:;.-_")
        if len(token) >= 2 and token not in {"진성", "가성", "해당없음", "불명확", "아니오", "있음", "없음"}:
            tokens.append(token)
    counts: dict[str, int] = {}
    for token in tokens:
        counts[token] = counts.get(token, 0) + 1
    top = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:limit]
    return ", ".join(f"{term}:{count}" for term, count in top)


def value_counts_text(series: pd.Series, limit: int = 8) -> str:
    counts = series.fillna("").map(clean_value).value_counts().head(limit)
    return ", ".join(f"{idx}:{int(value)}" for idx, value in counts.items())


def clean_value(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return str(value).replace("\r", " ").replace("\n", " ").strip()


def escape_md(value: str) -> str:
    return value.replace("|", "\\|")


def normalize_label(value: Any) -> str:
    text = "" if value is None else str(value).strip()
    if text in {JIN, "O", "o", "TRUE", "True", "true", "1"}:
        return JIN
    if text in {GA, "X", "x", "FALSE", "False", "false", "0"}:
        return GA
    return text


def normalize_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return False
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n", ""}:
        return False
    return bool(value)


def latest_name(path: Path) -> str:
    if path.is_file():
        return path.stem
    return path.name or "outputs"


if __name__ == "__main__":
    main()
