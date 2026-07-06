from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


JIN = "진성"
GA = "가성"


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two prediction CSV files and find where metrics changed.")
    parser.add_argument("--before", required=True, help="Baseline predictions CSV, e.g. old train_baseline_predictions.csv")
    parser.add_argument("--after", required=True, help="New predictions CSV")
    parser.add_argument("--out-dir", default="", help="Output directory. Default: artifacts/prediction_compare/<timestamp>")
    args = parser.parse_args()

    before_path = Path(args.before)
    after_path = Path(args.after)
    before = enrich(pd.read_csv(before_path), "before")
    after = enrich(pd.read_csv(after_path), "after")

    out_dir = Path(args.out_dir) if args.out_dir else Path("artifacts") / "prediction_compare" / datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)

    overall = pd.DataFrame([metrics_row(before, "before"), metrics_row(after, "after")])
    overall.to_csv(out_dir / "overall_metrics.csv", index=False, encoding="utf-8-sig")

    route_compare = compare_groups(before, after, ["route_primary_norm"])
    route_compare.to_csv(out_dir / "by_route.csv", index=False, encoding="utf-8-sig")

    major_compare = compare_groups(before, after, ["major"])
    major_compare.to_csv(out_dir / "by_major.csv", index=False, encoding="utf-8-sig")

    route_major_compare = compare_groups(before, after, ["route_primary_norm", "major"])
    route_major_compare.to_csv(out_dir / "by_route_major.csv", index=False, encoding="utf-8-sig")

    bucket_compare = compare_groups(before, after, ["review_bucket"])
    bucket_compare.to_csv(out_dir / "by_bucket.csv", index=False, encoding="utf-8-sig")

    transitions = compare_matched_ids(before, after)
    transitions.to_csv(out_dir / "matched_id_transitions.csv", index=False, encoding="utf-8-sig")

    report = build_report(overall, route_compare, route_major_compare, bucket_compare, transitions, before_path, after_path)
    (out_dir / "compare_report.md").write_text(report, encoding="utf-8")

    print(f"[compare] out_dir={out_dir}", flush=True)
    print(overall.to_string(index=False), flush=True)
    print("[compare] worst route_major by accuracy_delta", flush=True)
    print(route_major_compare.head(15).to_string(index=False), flush=True)


def enrich(df: pd.DataFrame, source: str) -> pd.DataFrame:
    work = df.copy()
    work["source"] = source
    work["label_norm"] = work.get("label", "").map(normalize_label) if "label" in work else ""
    work["pred_norm"] = work.get("pred", "").map(normalize_label) if "pred" in work else ""
    if "correct" in work:
        work["correct_norm"] = work["correct"].map(normalize_bool)
    else:
        work["correct_norm"] = work["label_norm"] == work["pred_norm"]
    if "exclude_from_metrics" in work:
        work["exclude_norm"] = work["exclude_from_metrics"].map(normalize_bool)
    else:
        work["exclude_norm"] = False
    work["evidence_obj"] = work.get("evidence", "").map(parse_jsonish) if "evidence" in work else [{}] * len(work)
    if "route_primary" not in work:
        work["route_primary"] = ""
    work["route_primary_norm"] = work.apply(route_primary, axis=1)
    work["review_bucket"] = work.apply(assign_bucket, axis=1)
    for col in ["major", "middle", "id"]:
        if col not in work:
            work[col] = ""
    return work


def metrics_row(df: pd.DataFrame, name: str) -> dict[str, Any]:
    eval_df = df[~df["exclude_norm"]].copy()
    labels = eval_df["label_norm"]
    preds = eval_df["pred_norm"]
    tp = int(((labels == JIN) & (preds == JIN)).sum())
    fn = int(((labels == JIN) & (preds == GA)).sum())
    fp = int(((labels == GA) & (preds == JIN)).sum())
    tn = int(((labels == GA) & (preds == GA)).sum())
    n = int(len(eval_df))
    return {
        "name": name,
        "n": n,
        "excluded": int(len(df) - n),
        "accuracy": safe_div(tp + tn, n),
        "TR": safe_div(tp, tp + fn),
        "TP": safe_div(tp, tp + fp),
        "FR": safe_div(tn, tn + fp),
        "FP": safe_div(tn, tn + fn),
        "tp_true": tp,
        "fn_true_as_false": fn,
        "fp_false_as_true": fp,
        "tn_false": tn,
    }


def compare_groups(before: pd.DataFrame, after: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    left = group_metrics(before, group_cols, "before")
    right = group_metrics(after, group_cols, "after")
    merged = left.merge(right, on=group_cols, how="outer").fillna(0)
    for metric in ["n", "accuracy", "TR", "TP", "FR", "FP", "fn_true_as_false", "fp_false_as_true"]:
        merged[f"{metric}_delta"] = merged[f"{metric}_after"] - merged[f"{metric}_before"]
    sort_cols = ["accuracy_delta", "fp_false_as_true_delta", "fn_true_as_false_delta", "n_after"]
    return merged.sort_values(sort_cols, ascending=[True, False, False, False]).reset_index(drop=True)


def group_metrics(df: pd.DataFrame, group_cols: list[str], suffix: str) -> pd.DataFrame:
    eval_df = df[~df["exclude_norm"]].copy()
    if eval_df.empty:
        return pd.DataFrame(columns=group_cols)
    rows = []
    for key, group in eval_df.groupby(group_cols, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        row = dict(zip(group_cols, key))
        row.update(metrics_row(group, suffix))
        row.pop("name", None)
        row.pop("excluded", None)
        rows.append(row)
    result = pd.DataFrame(rows)
    rename = {col: f"{col}_{suffix}" for col in result.columns if col not in group_cols}
    return result.rename(columns=rename)


def compare_matched_ids(before: pd.DataFrame, after: pd.DataFrame) -> pd.DataFrame:
    if "id" not in before or "id" not in after:
        return pd.DataFrame()
    left = before.drop_duplicates("id").copy()
    right = after.drop_duplicates("id").copy()
    merged = left.merge(right, on="id", suffixes=("_before", "_after"))
    if merged.empty:
        return merged
    merged["transition"] = merged.apply(
        lambda row: f"{bool(row.get('correct_norm_before'))}->{bool(row.get('correct_norm_after'))}",
        axis=1,
    )
    cols = [
        "id",
        "transition",
        "label_norm_before",
        "pred_norm_before",
        "pred_norm_after",
        "route_primary_norm_before",
        "route_primary_norm_after",
        "review_bucket_before",
        "review_bucket_after",
        "major_before",
        "middle_before",
        "confidence_before",
        "confidence_after",
        "reason_before",
        "reason_after",
    ]
    cols = [col for col in cols if col in merged.columns]
    return merged[cols].sort_values("transition")


def build_report(
    overall: pd.DataFrame,
    route_compare: pd.DataFrame,
    route_major_compare: pd.DataFrame,
    bucket_compare: pd.DataFrame,
    transitions: pd.DataFrame,
    before_path: Path,
    after_path: Path,
) -> str:
    lines = [
        "# Prediction Compare Report",
        "",
        f"- before: `{before_path}`",
        f"- after: `{after_path}`",
        "",
        "## Overall",
        "",
    ]
    lines.extend(markdown_table(overall, ["name", "n", "accuracy", "TR", "TP", "FR", "FP", "fn_true_as_false", "fp_false_as_true"]))
    lines.extend(["", "## By Route", ""])
    lines.extend(markdown_table(route_compare, ["route_primary_norm", "n_before", "n_after", "accuracy_before", "accuracy_after", "accuracy_delta", "TR_delta", "TP_delta", "fn_true_as_false_delta", "fp_false_as_true_delta"]))
    lines.extend(["", "## Worst Route/Major", ""])
    lines.extend(markdown_table(route_major_compare.head(20), ["route_primary_norm", "major", "n_before", "n_after", "accuracy_before", "accuracy_after", "accuracy_delta", "TR_delta", "TP_delta", "fn_true_as_false_delta", "fp_false_as_true_delta"]))
    lines.extend(["", "## By Review Bucket", ""])
    lines.extend(markdown_table(bucket_compare, ["review_bucket", "n_before", "n_after", "accuracy_before", "accuracy_after", "accuracy_delta", "TR_delta", "TP_delta", "fn_true_as_false_delta", "fp_false_as_true_delta"]))
    if not transitions.empty:
        counts = transitions["transition"].value_counts().reset_index()
        counts.columns = ["transition", "count"]
        lines.extend(["", "## Matched ID Transitions", ""])
        lines.extend(markdown_table(counts, ["transition", "count"]))
        worsened = transitions[transitions["transition"] == "True->False"].head(20)
        lines.extend(["", "## Newly Broken Matched Rows", ""])
        lines.extend(markdown_table(worsened, [col for col in ["id", "pred_norm_before", "pred_norm_after", "route_primary_norm_after", "major_before", "reason_after"] if col in worsened]))
    return "\n".join(lines) + "\n"


def route_primary(row: pd.Series) -> str:
    direct = str(row.get("route_primary", "") or "").strip()
    if direct:
        return direct
    evidence = row.get("evidence_obj", {})
    from_evidence = get_nested(evidence, "primary_route")
    return str(from_evidence or "general")


def assign_bucket(row: pd.Series) -> str:
    text = " ".join(
        str(row.get(col, ""))
        for col in ["route_primary", "major", "middle", "title", "reason", "decisive_evidence", "evidence"]
    )
    if any(token in text for token in ["지정 도구", "특정 도구", "대체 도구", "다른 도구", "대체 작업방법", "규정위반", "안전기준", "손들기"]):
        return "standard_rule_deviation"
    if any(token in text for token in ["배관", "서포트", "밟", "발판", "Toxic Duct", "덕트"]):
        return "pipe_support_or_access"
    if any(token in text for token in ["누출", "접액", "DIW", "응축수", "미상", "냄새", "가스", "방제", "ERT"]):
        return "leak_contact"
    if any(token in text for token in ["추락", "낙하", "고소", "사다리", "개구부", "난간", "그레이팅"]):
        return "height_fall"
    if any(token in text.lower() for token in ["ppe", "보호구", "안전고리", "안전대"]):
        return "ppe"
    if any(token in text for token in ["허가", "SOP", "서류", "협의", "일정", "작업예정", "사전"]):
        return "admin_prework"
    return "general"


def markdown_table(df: pd.DataFrame, cols: list[str], max_rows: int = 20) -> list[str]:
    if df.empty:
        return ["없음"]
    view = df.head(max_rows).copy()
    cols = [col for col in cols if col in view.columns]
    lines = ["|" + "|".join(cols) + "|", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, row in view.iterrows():
        values = [compact(row.get(col, ""), 140).replace("|", "/") for col in cols]
        lines.append("|" + "|".join(values) + "|")
    return lines


def normalize_label(value: Any) -> str:
    text = "" if value is None else str(value).strip()
    if "진" in text or "\uf9de" in text:
        return JIN
    if "가" in text:
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


def parse_jsonish(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return value
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return {}
    text = str(value).strip()
    if not text:
        return {}
    try:
        return json.loads(text)
    except Exception:
        return {}


def get_nested(value: Any, path: str) -> Any:
    current = value
    for part in path.split("."):
        if isinstance(current, dict):
            current = current.get(part, "")
        else:
            return ""
    return current


def compact(value: Any, limit: int) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    if isinstance(value, (dict, list)):
        text = json.dumps(value, ensure_ascii=False)
    else:
        text = "" if value is None else str(value)
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > limit:
        return text[:limit].rstrip() + "..."
    return text


def safe_div(num: int | float, den: int | float) -> float:
    return round(float(num) / float(den), 4) if den else 0.0


if __name__ == "__main__":
    main()
