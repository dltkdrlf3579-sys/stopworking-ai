from __future__ import annotations

import argparse

from stopright_ai.daily import run_daily_prediction


def main() -> None:
    parser = argparse.ArgumentParser(description="Run daily stop-work AI prediction batch.")
    parser.add_argument("--config", default="config.ini", help="config.ini path")
    parser.add_argument("--target-date", default=None, help="Source date as YYYY-MM-DD. Default: yesterday")
    args = parser.parse_args()

    run_daily_prediction(config_path=args.config, target_date=args.target_date)


if __name__ == "__main__":
    main()
