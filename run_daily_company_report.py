from __future__ import annotations

import argparse

from stopright_ai.daily_company_report import resolve_period, run_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a company report for a day or date range from existing AI judgements.")
    parser.add_argument("--config", default="daily_report.ini")
    dates = parser.add_mutually_exclusive_group()
    dates.add_argument("--target-date", help="Single day: YYYY-MM-DD; default: yesterday in Korea")
    dates.add_argument("--start-date", help="First day of the report period (inclusive): YYYY-MM-DD")
    parser.add_argument("--end-date", help="Last day of the report period (inclusive); requires --start-date")
    parser.add_argument("--demo", action="store_true", help="Preview with labeled synthetic data, no DB/API calls")
    args = parser.parse_args()
    try:
        start, end = resolve_period(args.target_date, args.start_date, args.end_date)
    except ValueError as exc:
        parser.error(str(exc))
    run_report(args.config, start_date=start.isoformat(), end_date=end.isoformat(), demo=args.demo)


if __name__ == "__main__":
    main()
