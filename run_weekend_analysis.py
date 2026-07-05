from __future__ import annotations

import argparse

from stopright_ai.weekend_analysis import run_weekend_analysis


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze weekend route-score sweep results.")
    parser.add_argument("input_root", nargs="?", default="outputs", help="Folder containing cycle output directories.")
    parser.add_argument("--output-dir", default="artifacts/weekend_analysis", help="Analysis output root folder.")
    args = parser.parse_args()

    paths = run_weekend_analysis(args.input_root, args.output_dir)
    print("[weekend-analysis] done")
    for name, path in paths.items():
        print(f"[weekend-analysis] {name}: {path}")


if __name__ == "__main__":
    main()
