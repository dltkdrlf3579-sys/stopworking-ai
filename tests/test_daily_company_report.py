from __future__ import annotations

import configparser
import io
import json
import tempfile
import unittest
from datetime import date, datetime
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from stopright_ai.daily_company_report import (
    FIELDS, KST, aggregate, analyse, case_batches, prepare_rows, render_html, resolve_period,
    run_report, validate_notes,
)


def settings() -> configparser.ConfigParser:
    config = configparser.ConfigParser()
    config.read_dict({"columns": FIELDS, "report": {"max_batch_chars": "1800", "max_case_chars": "100"}})
    return config


def source() -> pd.DataFrame:
    return pd.DataFrame([
        {"출원번호": "A1", "업체명": "A업체", "등록일": "2026-09-08", "AI판정": "진성", "판정일시": "2026-09-08 10:00"},
        {"출원번호": "A1", "업체명": "A업체", "등록일": "2026-09-08", "AI판정": "가성", "판정일시": "2026-09-08 11:00"},
        {"출원번호": "A2", "업체명": "A업체", "등록일": "2026-09-08", "AI판정": "보류"},
        {"출원번호": "B1", "업체명": "B업체", "등록일": "2026-09-08", "AI판정": "진성", "에러여부": "false"},
        {"출원번호": "B2", "업체명": "B업체", "등록일": "2026-09-08", "AI판정": "가성", "에러여부": "true"},
        {"출원번호": "OLD", "업체명": "B업체", "등록일": "2026-09-07", "AI판정": "진성"},
    ])


class DailyReportTest(unittest.TestCase):
    def test_period_arguments_and_korean_yesterday_default(self):
        self.assertEqual(resolve_period("2026-09-08"), (date(2026, 9, 8), date(2026, 9, 8)))
        self.assertEqual(resolve_period(start_date="2026-08-01", end_date="2026-08-31"),
                         (date(2026, 8, 1), date(2026, 8, 31)))
        with patch("stopright_ai.daily_company_report.datetime") as clock:
            clock.now.return_value = datetime(2026, 9, 8, 0, 30, tzinfo=KST)
            self.assertEqual(resolve_period(), (date(2026, 9, 7), date(2026, 9, 7)))
            clock.now.assert_called_once_with(KST)
        for args in [{"start_date": "2026-09-01"}, {"end_date": "2026-09-07"},
                     {"target_date": "2026-09-01", "end_date": "2026-09-07"},
                     {"target_date": "2026-09-01", "start_date": "2026-09-01", "end_date": "2026-09-07"},
                     {"start_date": "2026-09-07", "end_date": "2026-09-01"},
                     {"start_date": "2026-02-30", "end_date": "2026-09-07"}]:
            with self.subTest(args=args), self.assertRaises(ValueError):
                resolve_period(**args)

    def test_cli_accepts_period_and_rejects_incomplete_or_conflicting_dates(self):
        from run_daily_company_report import main
        with patch("sys.argv", ["run_daily_company_report.py", "--start-date", "2026-09-01", "--end-date", "2026-09-07"]), \
                patch("run_daily_company_report.run_report") as run:
            main()
        run.assert_called_once_with("daily_report.ini", start_date="2026-09-01", end_date="2026-09-07", demo=False)
        for args in [["--start-date", "2026-09-01"], ["--end-date", "2026-09-07"],
                     ["--start-date", "2026-09-07", "--end-date", "2026-09-01"],
                     ["--target-date", "2026-09-01", "--end-date", "2026-09-07"],
                     ["--target-date", "2026-09-01", "--start-date", "2026-09-01"]]:
            with self.subTest(args=args), patch("sys.argv", ["run_daily_company_report.py", *args]), \
                    patch("sys.stderr", new_callable=io.StringIO), patch("run_daily_company_report.run_report") as run:
                with self.assertRaises(SystemExit) as exc:
                    main()
                self.assertEqual(exc.exception.code, 2)
                run.assert_not_called()

    def test_date_dedup_and_pending_denominator(self):
        data, meta = prepare_rows(source(), settings(), date(2026, 9, 8))
        stats = aggregate(data, date(2026, 9, 8), meta)
        self.assertEqual(stats["overall"], {"total": 4, "true": 1, "false": 1, "pending": 2, "false_pct": 50.0})
        self.assertEqual(meta["outside_date_rows"], 1)
        self.assertEqual(meta["duplicate_rows_removed"], 1)
        self.assertEqual(stats["companies"][0]["false_pct"], 100.0)
        same_day, same_meta = prepare_rows(source(), settings(), date(2026, 9, 8), date(2026, 9, 8))
        pd.testing.assert_frame_equal(data, same_day)
        self.assertEqual(meta, same_meta)

    def test_period_includes_both_days_in_korea_and_deduplicates(self):
        df = pd.DataFrame([
            {"id": "BEFORE", "date": "2026-08-31T14:59:59Z", "order": "2026-09-08 10:00", "prediction": "진성"},
            {"id": "START", "date": "2026-08-31T15:00:00Z", "order": "2026-09-08 10:00", "prediction": "진성"},
            {"id": "END", "date": "2026-09-07T14:59:59Z", "order": "2026-09-08 10:00", "prediction": "진성"},
            {"id": "AFTER", "date": "2026-09-07T15:00:00Z", "order": "2026-09-08 10:00", "prediction": "진성"},
            {"id": "START", "date": "2026-08-31T15:00:00Z", "order": "2026-09-08 11:00", "prediction": "가성"},
        ])
        df["company"] = "A업체"
        df["error"] = False
        start, end = date(2026, 9, 1), date(2026, 9, 7)
        data, meta = prepare_rows(df.rename(columns=FIELDS), settings(), start, end)
        self.assertEqual(set(data["id"]), {"START", "END"})
        self.assertEqual(dict(zip(data["id"], data["date"])), {"START": "2026-09-01", "END": "2026-09-07"})
        self.assertEqual(meta, {"queried_rows": 5, "outside_date_rows": 2, "duplicate_rows_removed": 1})
        stats = aggregate(data, start, meta, end)
        self.assertEqual(stats["overall"], {"total": 2, "true": 1, "false": 1, "pending": 0, "false_pct": 50.0})
        self.assertEqual(stats["period_label"], "2026-09-01 ~ 2026-09-07")
        self.assertEqual(stats["report_type"], "기간")

    def test_ambiguous_duplicates_fail(self):
        df = source().drop(columns="판정일시")
        with self.assertRaisesRegex(ValueError, "중복 출원번호"):
            prepare_rows(df, settings(), date(2026, 9, 8))

    def test_no_data_and_query_error_distinguished(self):
        empty = source().iloc[:0]
        data, meta = prepare_rows(empty, settings(), date(2026, 9, 8))
        self.assertEqual(aggregate(data, date(2026, 9, 8), meta)["overall"]["total"], 0)
        with self.assertRaisesRegex(ValueError, "필수 컬럼 누락"):
            prepare_rows(pd.DataFrame(), settings(), date(2026, 9, 8))

    def test_empty_period_with_numeric_or_boolean_db_columns(self):
        for dtype, value in [("bool", False), ("int64", 0), ("float64", float("nan")), ("boolean", pd.NA)]:
            for empty_query in (False, True):
                with self.subTest(dtype=dtype, empty_query=empty_query):
                    df = source().iloc[[0]].copy()
                    df["에러여부"] = pd.Series([value], dtype=dtype, index=df.index)
                    df["등록일"] = pd.to_datetime(df["등록일"])
                    if empty_query:
                        df = df.iloc[:0]
                    data, meta = prepare_rows(df, settings(), date(2026, 8, 1), date(2026, 8, 31))
                    self.assertTrue(data.empty)
                    self.assertEqual(meta["outside_date_rows"], len(df))

    def test_boolean_and_numeric_error_values_keep_pending_semantics(self):
        df = source().iloc[[0, 2, 3, 4]].copy()
        df["AI판정"] = "진성"
        df["에러여부"] = [True, False, 1, 0]
        data, _ = prepare_rows(df, settings(), date(2026, 9, 1), date(2026, 9, 8))
        self.assertEqual(data["prediction"].tolist(), ["보류/미판정", "진성", "보류/미판정", "진성"])

    def test_batches_cover_every_case_and_strip_base64(self):
        data, _ = prepare_rows(source(), settings(), date(2026, 9, 8))
        data["phenomenon"] = '<p>원문 현상</p><img src="data:image/png;base64,' + "A" * 4000 + '">'
        batches = case_batches(data, 1000, 100)
        cases = [case for batch in batches for case in batch]
        self.assertEqual([c["id"] for c in cases], data["id"].tolist())
        self.assertNotIn("base64", json.dumps(cases))
        self.assertTrue(all(len(json.dumps(b, ensure_ascii=False)) <= 1000 for b in batches))

    def test_evidence_from_wrong_company_rejected(self):
        with self.assertRaisesRegex(ValueError, "다른 업체"):
            validate_notes([{"company": "A", "detail": "관찰", "ids": ["B1"]}], {"A1", "B1"}, company_ids={"A": {"A1"}})

    def test_html_escapes_source_and_model_text(self):
        data, meta = prepare_rows(source(), settings(), date(2026, 9, 8))
        stats = aggregate(data, date(2026, 9, 8), meta)
        html = render_html(stats, {"overview": "<script>alert(1)</script>", "issues": [], "companies": [], "actions": []})
        self.assertNotIn("<script>", html)
        self.assertIn("&lt;script&gt;", html)
        self.assertIn("100.0%", html)

    def test_ai_receives_each_company_separately_and_full_statistics(self):
        config = settings()
        data, meta = prepare_rows(source(), config, date(2026, 9, 7), date(2026, 9, 8))
        stats = aggregate(data, date(2026, 9, 7), meta, date(2026, 9, 8))
        seen = []
        def fake_invoke(llm, system, user):
            payload = json.loads(user)
            if "cases" in payload:
                self.assertEqual(payload["start_date"], "2026-09-07")
                self.assertEqual(payload["end_date"], "2026-09-08")
                rows = payload["cases"]
                self.assertTrue(all(row["date"] in {"2026-09-07", "2026-09-08"} for row in rows))
                self.assertEqual(len({row["company"] for row in rows}), 1)
                seen.extend(row["id"] for row in rows)
                return {"observations": [{"title": "등록 사유", "detail": "기록 확인", "ids": [rows[0]["id"]]}]}
            self.assertEqual(payload["statistics"]["overall"]["total"], 5)
            self.assertEqual(payload["statistics"]["end_date"], "2026-09-08")
            return {"overview": "보고서 요약", "issues": [], "companies": [], "actions": []}
        with tempfile.TemporaryDirectory() as temp, patch("stopright_ai.daily_company_report.invoke_json", fake_invoke):
            result, partials = analyse(data, stats, "지침", object(), config, Path(temp) / "partial.json")
        self.assertCountEqual(seen, data["id"].tolist())
        self.assertEqual(sum(p["analysed_rows"] for p in partials), 5)
        self.assertEqual(result["overview"], "보고서 요약")

    def test_end_to_end_query_ai_and_mail_body_without_sending(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            config = settings()
            config["report"].update({"api_config": str(root / "config.ini"), "query_file": str(root / "queries/daily_company_report.sql"),
                                      "prompt_file": str(root / "policies/daily_company_report.md"), "output_dir": str(folder)})
            query = folder / "query.sql"
            query.write_text("SELECT '{target_date}' AS requested_date", encoding="utf-8")
            config["report"]["query_file"] = str(query)
            config_file = folder / "report.ini"
            with config_file.open("w", encoding="utf-8") as f:
                config.write(f)
            def query_fn(sql):
                self.assertIn("2026-09-08", sql)
                self.assertNotIn("{target_date}", sql)
                return source()
            def fake_invoke(llm, system, user):
                payload = json.loads(user)
                if "cases" in payload:
                    return {"observations": [{"title": "현상", "detail": "기록 확인", "ids": [payload["cases"][0]["id"]]}]}
                return {"overview": "테스트 요약", "issues": [], "companies": [], "actions": ["기준 확인"]}
            with patch("stopright_ai.daily_company_report.load_execute_sql", return_value=query_fn), patch("stopright_ai.daily_company_report.invoke_json", fake_invoke):
                report = run_report(str(config_file), "2026-09-08", llm=object())
            mail = json.loads((report.parent / "mail_content.json").read_text(encoding="utf-8"))
            self.assertEqual(mail["html_body"], report.read_text(encoding="utf-8"))
            self.assertFalse(mail["demo"])
            self.assertEqual(json.loads((report.parent / "summary.json").read_text(encoding="utf-8"))["overall"]["total"], 4)
            self.assertIn("2026-09-08 업체별 일일 보고서", mail["subject"])

            query.write_text("SELECT '{target_date}', '{start_date}', '{end_date}', '{next_date}'", encoding="utf-8")
            def period_query(sql):
                self.assertEqual(sql, "SELECT '2026-09-07', '2026-09-07', '2026-09-08', '2026-09-09'")
                return source()
            with patch("stopright_ai.daily_company_report.load_execute_sql", return_value=period_query), \
                    patch("stopright_ai.daily_company_report.invoke_json", fake_invoke):
                report = run_report(str(config_file), start_date="2026-09-07", end_date="2026-09-08", llm=object())
            mail = json.loads((report.parent / "mail_content.json").read_text(encoding="utf-8"))
            self.assertIn("2026-09-07 ~ 2026-09-08 업체별 기간 보고서", mail["subject"])
            self.assertIn("기간 요약", mail["html_body"])
            self.assertNotIn("오늘의 요약", mail["html_body"])
            self.assertEqual(json.loads((report.parent / "summary.json").read_text(encoding="utf-8"))["overall"]["total"], 5)

            with patch("stopright_ai.daily_company_report.load_execute_sql", return_value=lambda sql: source().assign(**{"에러여부": False})), \
                    patch("stopright_ai.daily_company_report.invoke_json") as invoke, \
                    patch("stopright_ai.daily_company_report.create_llm") as create:
                report = run_report(str(config_file), start_date="2026-08-01", end_date="2026-08-31")
            invoke.assert_not_called()
            create.assert_not_called()
            stats = json.loads((report.parent / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(stats["overall"]["total"], 0)
            self.assertEqual(stats["data_checks"]["outside_date_rows"], len(source()))
            self.assertIn("대상 기간에 조회된 작업중지권 사례가 없습니다", report.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
