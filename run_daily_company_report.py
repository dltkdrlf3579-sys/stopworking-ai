from __future__ import annotations

import argparse

from stopright_ai.daily_company_report import run_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a daily company report from existing AI judgements.")
    parser.add_argument("--config", default="daily_report.ini")
    parser.add_argument("--target-date", help="YYYY-MM-DD; default: yesterday in Korea")
    parser.add_argument("--demo", action="store_true", help="Preview with labeled synthetic data, no DB/API calls")
    args = parser.parse_args()
    run_report(args.config, args.target_date, demo=args.demo)


if __name__ == "__main__":
    main()
