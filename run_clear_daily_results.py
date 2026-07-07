from __future__ import annotations

import argparse
import re
from typing import Any

from stopright_ai.config import load_config
from stopright_ai.db_adapter import load_execute_dml, load_execute_sql


def main() -> None:
    parser = argparse.ArgumentParser(description="Clear stored daily AI judgement results.")
    parser.add_argument("--config", default="config.ini", help="config.ini path")
    parser.add_argument("--all", action="store_true", help="Delete all rows in the result history table")
    parser.add_argument("--source-date", default="", help="Delete one source_date, format YYYY-MM-DD")
    parser.add_argument("--source-date-from", default="", help="Delete rows where source_date >= this date")
    parser.add_argument("--run-id", default="", help="Delete one run_id")
    parser.add_argument("--application-no", default="", help="Delete one application_no")
    parser.add_argument("--yes", action="store_true", help="Required confirmation flag")
    args = parser.parse_args()

    if not args.yes:
        raise SystemExit("Refusing to delete without --yes")

    filters = [
        args.all,
        bool(args.source_date),
        bool(args.source_date_from),
        bool(args.run_id),
        bool(args.application_no),
    ]
    if sum(bool(item) for item in filters) != 1:
        raise SystemExit("Choose exactly one delete scope: --all, --source-date, --source-date-from, --run-id, or --application-no")

    config = load_config(args.config)
    table = config.get("daily_prediction", "result_history_table", fallback="ai_stopwork_judgement_history").strip()
    validate_identifier(table)

    where_sql = build_where_sql(args)
    count_sql = f"SELECT COUNT(*) AS cnt FROM {table}{where_sql}"
    delete_sql = f"DELETE FROM {table}{where_sql}"

    execute_sql = load_execute_sql(config)
    execute_dml = load_execute_dml(config)

    before = fetch_count(execute_sql, count_sql)
    print(f"[clear] table={table}")
    print(f"[clear] matched_rows={before}")
    print(f"[clear] sql={delete_sql}")

    if before == 0:
        print("[clear] nothing to delete")
        return

    execute_dml(delete_sql)

    after = fetch_count(execute_sql, f"SELECT COUNT(*) AS cnt FROM {table}{where_sql}")
    print(f"[clear] deleted_rows={before - after}")
    print(f"[clear] remaining_matched_rows={after}")


def build_where_sql(args: argparse.Namespace) -> str:
    if args.all:
        return ""
    if args.source_date:
        return f" WHERE source_date = DATE {sql_literal(args.source_date)}"
    if args.source_date_from:
        return f" WHERE source_date >= DATE {sql_literal(args.source_date_from)}"
    if args.run_id:
        return f" WHERE run_id = {sql_literal(args.run_id)}"
    if args.application_no:
        return f" WHERE application_no = {sql_literal(args.application_no)}"
    raise ValueError("No delete scope selected")


def fetch_count(execute_sql: Any, query: str) -> int:
    df = execute_sql(query)
    if df.empty:
        return 0
    return int(df.iloc[0, 0])


def validate_identifier(identifier: str) -> None:
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)?", identifier):
        raise ValueError(f"Unsafe table identifier: {identifier}")


def sql_literal(value: str) -> str:
    return "'" + str(value).replace("'", "''") + "'"


if __name__ == "__main__":
    main()
