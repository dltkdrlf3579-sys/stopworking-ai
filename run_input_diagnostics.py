from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from data import load_df
from stopright_ai.case_prepare import row_to_case
from stopright_ai.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check preprocessing truncation and saved prediction ellipses without calling the LLM."
    )
    parser.add_argument("--config", default="config.ini")
    parser.add_argument("--limit", type=int, default=0, help="Only inspect the first N dataframe rows. 0 means all rows.")
    parser.add_argument("--pred-csv", default="", help="Optional predictions CSV to inspect for saved ellipses.")
    parser.add_argument("--skip-df", action="store_true", help="Only inspect --pred-csv. Do not load df or run preprocessing.")
    args = parser.parse_args()

    config = load_config(args.config)

    out_dir = Path("artifacts") / "diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_df:
        print("[input-diagnostics] loading dataframe", flush=True)
        df = load_df(config)
        if args.limit and args.limit > 0:
            df = df.head(args.limit).copy()
        print(f"[input-diagnostics] rows={len(df)}", flush=True)

        rows = []
        for idx, row in enumerate(df.to_dict("records"), start=1):
            case = row_to_case(row, config)
            rows.append(
                {
                    "row_no": idx,
                    "id": case.get("id", ""),
                    "major": case.get("major", ""),
                    "middle": case.get("middle", ""),
                    "title_len": len(str(case.get("title", ""))),
                    "phenomenon_len": len(str(case.get("phenomenon_text", ""))),
                    "action_len": len(str(case.get("action", ""))),
                    "phenomenon_truncated": bool(case.get("phenomenon_truncated", False)),
                    "action_truncated": bool(case.get("action_truncated", False)),
                    "image_count": int(case.get("image_count", 0) or 0),
                    "original_image_count": int(case.get("original_image_count", 0) or 0),
                    "omitted_image_count": int(case.get("omitted_image_count", 0) or 0),
                }
            )

        diag = pd.DataFrame(rows)
        out_path = out_dir / "input_truncation_diagnostics.csv"
        diag.to_csv(out_path, index=False, encoding="utf-8-sig")

        total = len(diag)
        phenomenon_truncated = int(diag["phenomenon_truncated"].sum()) if total else 0
        action_truncated = int(diag["action_truncated"].sum()) if total else 0
        image_omitted = int((diag["omitted_image_count"] > 0).sum()) if total else 0
        omitted_images_total = int(diag["omitted_image_count"].sum()) if total else 0

        print("[input-diagnostics] summary", flush=True)
        print(f"  phenomenon_truncated_rows={phenomenon_truncated}/{total}", flush=True)
        print(f"  action_truncated_rows={action_truncated}/{total}", flush=True)
        print(f"  rows_with_omitted_images={image_omitted}/{total}", flush=True)
        print(f"  omitted_images_total={omitted_images_total}", flush=True)
        print(f"  detail_csv={out_path}", flush=True)

    if args.pred_csv:
        inspect_prediction_csv(Path(args.pred_csv), out_dir)


def inspect_prediction_csv(path: Path, out_dir: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"predictions CSV not found: {path}")

    df = pd.read_csv(path)
    cols = [
        col
        for col in [
            "reason",
            "decisive_evidence",
            "evidence",
            "true_argument",
            "false_argument",
            "critic",
        ]
        if col in df.columns
    ]
    rows = []
    for col in cols:
        text = df[col].fillna("").astype(str)
        rows.append(
            {
                "column": col,
                "rows": len(text),
                "contains_length_limit_marker": int(text.str.contains("길이제한|truncated", regex=True).sum()),
                "ends_with_ellipsis": int(text.str.rstrip().str.endswith("...").sum()),
                "contains_ellipsis": int(text.str.contains(r"\.\.\.", regex=True).sum()),
                "max_len": int(text.map(len).max()) if len(text) else 0,
                "avg_len": round(float(text.map(len).mean()), 1) if len(text) else 0,
            }
        )

    summary = pd.DataFrame(rows)
    out_path = out_dir / "prediction_text_ellipsis_diagnostics.csv"
    summary.to_csv(out_path, index=False, encoding="utf-8-sig")
    print("[input-diagnostics] prediction CSV text summary", flush=True)
    print(summary.to_string(index=False), flush=True)
    print(f"  prediction_text_csv={out_path}", flush=True)


if __name__ == "__main__":
    main()
