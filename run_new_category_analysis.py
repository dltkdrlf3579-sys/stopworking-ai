from __future__ import annotations

import argparse

from stopright_ai.new_category_analysis import run_new_category_analysis


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Classify analysis rows into NEW_대분류/NEW_소분류 and export an Excel file."
    )
    parser.add_argument("--config", default="config.ini", help="config.ini path")
    parser.add_argument(
        "--query-file",
        default="queries/new_category_analysis.sql",
        help="analysis SQL path",
    )
    parser.add_argument(
        "--prompt-file",
        default="policies/new_category_prompt.md",
        help="classification prompt path",
    )
    parser.add_argument("--output", default=None, help="output .xlsx path")
    parser.add_argument(
        "--max-rows",
        type=int,
        default=0,
        help="limit rows for a quick test; 0 processes all rows",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="override concurrent workers",
    )
    args = parser.parse_args()

    run_new_category_analysis(
        config_path=args.config,
        query_file=args.query_file,
        prompt_file=args.prompt_file,
        output_path=args.output,
        max_rows=args.max_rows,
        max_workers=args.max_workers,
    )


if __name__ == "__main__":
    main()
