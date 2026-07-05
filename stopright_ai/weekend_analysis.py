from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from .artifacts import save_json
from .evaluate import compute_metrics
from .prediction_analysis import analyze_prediction_outputs, read_table


MODES = ["record", "assist", "guardrail", "autotuned_guardrail", "evolved_profile"]
SPLITS = ["train", "validation"]


def run_weekend_analysis(
    input_root: str | Path = "outputs",
    output_root: str | Path = "artifacts/weekend_analysis",
) -> dict[str, Path]:
    input_path = Path(input_root)
    out_dir = Path(output_root) / datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)

    prediction_index = build_prediction_index(input_path)
    save_prediction_index(prediction_index, out_dir / "prediction_file_index.csv")

    record_files = [
        item["path"]
        for item in prediction_index
        if item["mode"] == "record" and item["split"] in SPLITS
    ]
    if not record_files:
        raise FileNotFoundError(f"No route_score_record prediction files found under {input_path}")

    record_analysis_paths = analyze_prediction_outputs(
        record_files,
        output_root=out_dir / "record_only_error_analysis",
        include_candidates=False,
    )

    combined_metrics = build_mode_metrics(prediction_index)
    combined_metrics.to_csv(out_dir / "mode_metrics.csv", index=False, encoding="utf-8-sig")

    degradation = build_degradation_report(prediction_index)
    degradation["summary"].to_csv(out_dir / "mode_degradation_summary.csv", index=False, encoding="utf-8-sig")
    for name, df in degradation["details"].items():
        df.to_csv(out_dir / f"{name}.csv", index=False, encoding="utf-8-sig")

    paths = {
        "output_dir": out_dir,
        "prediction_file_index": out_dir / "prediction_file_index.csv",
        "mode_metrics": out_dir / "mode_metrics.csv",
        "mode_degradation_summary": out_dir / "mode_degradation_summary.csv",
        **{f"record_analysis_{key}": value for key, value in record_analysis_paths.items()},
    }
    save_json(out_dir / "analysis_manifest.json", {key: str(value) for key, value in paths.items()})
    write_readme(out_dir, paths)
    return paths


def build_prediction_index(input_root: Path) -> list[dict[str, Any]]:
    rows = []
    if not input_root.exists():
        return rows

    for run_dir in sorted([path for path in input_root.iterdir() if path.is_dir()]):
        for mode in MODES:
            mode_dir = run_dir / f"route_score_{mode}"
            if mode in {"autotuned_guardrail", "evolved_profile"}:
                mode_dir = run_dir / f"route_score_{mode}"
            if not mode_dir.exists():
                continue
            for split in SPLITS:
                path = mode_dir / f"{split}_predictions.csv"
                if path.exists():
                    rows.append(
                        {
                            "run_dir": str(run_dir),
                            "run_id": run_dir.name,
                            "mode": mode,
                            "split": split,
                            "path": path,
                        }
                    )

    return rows


def save_prediction_index(index: list[dict[str, Any]], path: Path) -> None:
    df = pd.DataFrame([{**item, "path": str(item["path"])} for item in index])
    df.to_csv(path, index=False, encoding="utf-8-sig")


def build_mode_metrics(index: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for mode in MODES:
        for split in SPLITS:
            files = [item["path"] for item in index if item["mode"] == mode and item["split"] == split]
            if not files:
                continue
            df = pd.concat([read_table(path) for path in files], ignore_index=True, sort=False)
            metrics = compute_metrics(df)
            rows.append(
                {
                    "mode": mode,
                    "split": split,
                    "file_count": len(files),
                    "total_n": metrics.get("total_n", 0),
                    "n": metrics.get("n", 0),
                    "excluded_n": metrics.get("excluded_n", 0),
                    "accuracy": metrics.get("accuracy", 0),
                    "score": metrics.get("score", 0),
                    "true_recall": metrics.get("true_recall", 0),
                    "true_precision": metrics.get("true_precision", 0),
                    "false_recall": metrics.get("false_recall", 0),
                    "false_precision": metrics.get("false_precision", 0),
                    "fn_true_as_false": metrics.get("fn_true_as_false", 0),
                    "fp_false_as_true": metrics.get("fp_false_as_true", 0),
                    "tp_true": metrics.get("tp_true", 0),
                    "tn_false": metrics.get("tn_false", 0),
                }
            )
    return pd.DataFrame(rows)


def build_degradation_report(index: list[dict[str, Any]]) -> dict[str, Any]:
    details: dict[str, pd.DataFrame] = {}
    summary_rows = []

    runs = sorted({item["run_id"] for item in index})
    for run_id in runs:
        for split in SPLITS:
            record_item = find_index_item(index, run_id, "record", split)
            if not record_item:
                continue
            record_df = normalize_for_compare(read_table(record_item["path"]), "record")
            for mode in ["assist", "guardrail", "autotuned_guardrail", "evolved_profile"]:
                mode_item = find_index_item(index, run_id, mode, split)
                if not mode_item:
                    continue
                mode_df = normalize_for_compare(read_table(mode_item["path"]), mode)
                merged = record_df.merge(mode_df, on=["id_key"], how="inner", suffixes=("_record", f"_{mode}"))
                if merged.empty:
                    continue

                new_errors = merged[(merged["correct_record"] == True) & (merged[f"correct_{mode}"] == False)].copy()
                fixed_errors = merged[(merged["correct_record"] == False) & (merged[f"correct_{mode}"] == True)].copy()
                changed = merged[merged["pred_record"] != merged[f"pred_{mode}"]].copy()

                key = f"{mode}_{split}_new_errors"
                if not new_errors.empty:
                    details.setdefault(key, []).append(new_errors)
                fixed_key = f"{mode}_{split}_fixed_errors"
                if not fixed_errors.empty:
                    details.setdefault(fixed_key, []).append(fixed_errors)

                summary_rows.append(
                    {
                        "run_id": run_id,
                        "split": split,
                        "mode": mode,
                        "paired_rows": len(merged),
                        "changed_predictions": len(changed),
                        "new_errors_vs_record": len(new_errors),
                        "fixed_errors_vs_record": len(fixed_errors),
                        "net_error_delta": len(new_errors) - len(fixed_errors),
                        "new_fp_false_as_true": int(((new_errors["label_record"] == "가성") & (new_errors[f"pred_{mode}"] == "진성")).sum()),
                        "new_fn_true_as_false": int(((new_errors["label_record"] == "진성") & (new_errors[f"pred_{mode}"] == "가성")).sum()),
                    }
                )

    detail_frames = {
        key: pd.concat(value, ignore_index=True, sort=False) if isinstance(value, list) else value
        for key, value in details.items()
    }
    return {
        "summary": pd.DataFrame(summary_rows),
        "details": detail_frames,
    }


def normalize_for_compare(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    work = df.copy()
    for col in ["id", "label", "pred", "correct", "confidence", "major", "middle", "title", "reason", "route_primary", "route_true_score", "route_false_score"]:
        if col not in work.columns:
            work[col] = ""
    work["id_key"] = work["id"].astype(str).str.strip()
    work["correct"] = work["correct"].map(normalize_bool)
    keep = ["id_key", "id", "label", "pred", "correct", "confidence", "major", "middle", "title", "reason", "route_primary", "route_true_score", "route_false_score"]
    work = work[keep].copy()
    return work.rename(columns={col: f"{col}_{mode}" for col in keep if col != "id_key"})


def find_index_item(index: list[dict[str, Any]], run_id: str, mode: str, split: str) -> dict[str, Any] | None:
    for item in index:
        if item["run_id"] == run_id and item["mode"] == mode and item["split"] == split:
            return item
    return None


def normalize_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    return text in {"true", "1", "yes", "y"}


def write_readme(out_dir: Path, paths: dict[str, Path]) -> None:
    lines = [
        "# Weekend Result Analysis",
        "",
        "## 핵심 파일",
        "",
        "- `mode_metrics.csv`: 주말 전체 결과를 mode/split별로 합친 성능표",
        "- `mode_degradation_summary.csv`: assist/guardrail/evolved가 record 대비 새로 망친 행 수",
        "- `*_new_errors.csv`: record는 맞았는데 해당 mode가 틀린 행",
        "- `record_only_error_analysis/.../report.md`: record 기준 오답군집 보고서",
        "",
        "## 해석",
        "",
        "- assist/guardrail의 `new_errors_vs_record`가 크면 점수판은 판정 입력에서 빼야 한다.",
        "- record-only report의 FN/FP 클러스터가 다음 프롬프트 개선 재료다.",
        "- `fixed_errors_vs_record`보다 `new_errors_vs_record`가 크면 해당 mode는 폐기한다.",
        "",
        "## 생성 위치",
        "",
    ]
    for key, value in paths.items():
        lines.append(f"- {key}: `{value}`")
    (out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
