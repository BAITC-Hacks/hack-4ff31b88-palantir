import csv
from datetime import datetime, timedelta, timezone
import io
import json
import os
from pathlib import Path
import tempfile
import types
import unittest
from contextlib import redirect_stdout
from unittest.mock import MagicMock, patch

from agent_analysis import analyze, summarize
from agent_backtest import run_backtest, stub_forecast
from agent_skeleton import run


class AnalysisTests(unittest.TestCase):
    def setUp(self):
        self.issue = datetime(2026, 2, 1, tzinfo=timezone.utc)
        self.times = [self.issue + timedelta(hours=h) for h in range(48)]
        self.rows = stub_forecast(self.issue)

    def test_complete_valid_forecast(self):
        report = analyze(self.rows, self.times)
        self.assertTrue(report["accepted"])
        self.assertEqual(report["valid_points"], 48)
        self.assertIsNone(report["mean_abs_revision"])

    def test_zero_one_and_jump(self):
        self.rows[0]["power_pred"], self.rows[1]["power_pred"] = 0, 1
        report = analyze(self.rows, self.times)
        self.assertTrue(report["accepted"])
        self.assertTrue(any(w["rule"] == "hourly_jump" for w in report["warnings"]))

    def test_missing_nonfinite_and_outside_values(self):
        for value in [None, "", float("nan"), float("inf"), -0.01, 1.01, True]:
            with self.subTest(value=value):
                self.rows[0]["power_pred"] = value
                report = analyze(self.rows, self.times)
                self.assertFalse(report["accepted"])
                json.dumps(report, allow_nan=False)

    def test_duplicate_missing_and_unexpected_hours(self):
        report = analyze(self.rows[:-1] + [self.rows[0]], self.times)
        self.assertEqual({e["rule"] for e in report["errors"]}, {"duplicate_hour", "missing_hour"})
        self.rows[0]["target_time"] = (self.issue - timedelta(hours=1)).isoformat()
        self.assertFalse(analyze(self.rows, self.times)["accepted"])

    def test_comparison_by_timestamp_not_position(self):
        previous = {t.isoformat(): 0.2 for t in self.times[:24]}
        report = analyze(list(reversed(self.rows)), self.times, previous)
        self.assertEqual(report["overlap_points"], 24)
        self.assertEqual(report["changed_points"], 24)
        self.assertAlmostEqual(report["mean_abs_revision"], 0.3)

    def test_utc_normalization_and_zero_previous(self):
        row = {"target_time": "2026-02-01T06:00:00+06:00", "power_pred": 0}
        report = analyze([row], [self.issue], {"2026-02-01T00:00:00": 0})
        self.assertTrue(report["accepted"])
        self.assertEqual(report["overlap_points"], 1)
        self.assertEqual(report["changed_points"], 0)

    def test_threshold_validation(self):
        with self.assertRaises(ValueError):
            analyze(self.rows, self.times, jump=float("nan"))

    def test_llm_no_key_and_failure_fallback(self):
        report = analyze(self.rows, self.times)
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(summarize(report, "test-model")["source"], "rules")
        with patch.dict(os.environ, {"OPENAI_API_KEY": "dummy"}), patch.dict("sys.modules", {"openai": None}):
            result = summarize(report, "test-model")
            self.assertEqual(result["source"], "rules")
            self.assertIn("llm_error", result)

    def test_llm_success_and_empty_response(self):
        report = analyze(self.rows, self.times)
        client = MagicMock()
        client.__enter__.return_value = client
        client.responses.create.return_value.output_text = "Тестовое резюме."
        module = types.SimpleNamespace(OpenAI=MagicMock(return_value=client))
        with patch.dict(os.environ, {"OPENAI_API_KEY": "dummy"}), patch.dict("sys.modules", {"openai": module}):
            self.assertEqual(summarize(report, "test-model")["source"], "llm")
            facts = json.loads(client.responses.create.call_args.kwargs["input"])
            self.assertNotIn("valid_forecast", facts)
            client.responses.create.return_value.output_text = ""
            self.assertEqual(summarize(report, "test-model")["source"], "rules")


class BacktestTests(unittest.TestCase):
    def test_full_calendar_and_exports(self):
        with tempfile.TemporaryDirectory() as folder, redirect_stdout(io.StringIO()):
            result = run_backtest(folder)
            self.assertEqual(result["runs"], 29)
            self.assertEqual(result["accepted_runs"], 29)
            self.assertEqual(result["replaced_points"], 672)
            self.assertEqual(result["changed_points"], 0)
            self.assertEqual(result["february_hours"], 672)
            self.assertEqual(result["scada_offset_h"], 6)
            self.assertTrue(result["complete"])
            with (Path(folder) / "february_latest.csv").open(encoding="utf-8-sig") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(rows[0]["target_time_local"], "2026-02-01T00:00:00+06:00")
            self.assertEqual(rows[-1]["target_time_local"], "2026-02-28T23:00:00+06:00")
            self.assertEqual(len({r["target_time"] for r in rows}), 672)
            with (Path(folder) / "forecast_history.csv").open(encoding="utf-8-sig") as f:
                history = list(csv.DictReader(f))
            self.assertEqual(len(history), 1392)
            self.assertEqual({int(r["lead_hour"]) for r in history}, set(range(1, 49)))
            with self.assertRaises(FileExistsError):
                run_backtest(folder)

    def test_revisions_are_logged_and_applied(self):
        def forecast(issue):
            rows = stub_forecast(issue)
            for row in rows:
                row["power_pred"] = 0.2 if issue.day == 31 else 0.4
            return rows
        with tempfile.TemporaryDirectory() as folder, redirect_stdout(io.StringIO()):
            result = run_backtest(folder, forecast_fn=forecast, end="2026-02-01")
            self.assertEqual(result["changed_points"], 24)
            records = [json.loads(s) for s in (Path(folder) / "runs.jsonl").read_text(encoding="utf-8").splitlines()]
            change = records[1]["analysis"]["changes"][0]
            self.assertEqual(change["old"], 0.2)
            self.assertEqual(change["new"], 0.4)
            self.assertAlmostEqual(change["delta"], 0.2)

    def test_rejection_retains_old_forecast_and_next_day_runs(self):
        calls = []
        def forecast(issue):
            calls.append(issue)
            rows = stub_forecast(issue)
            if issue.day == 1:
                rows[0]["power_pred"] = None
            return rows
        with tempfile.TemporaryDirectory() as folder, redirect_stdout(io.StringIO()):
            result = run_backtest(folder, forecast_fn=forecast, end="2026-02-02")
            self.assertEqual(len(calls), 3)
            self.assertEqual(result["accepted_runs"], 2)
            with (Path(folder) / "february_latest.csv").open(encoding="utf-8-sig") as f:
                row = next(r for r in csv.DictReader(f) if r["target_time"] == "2026-02-01T00:00:00+00:00")
            self.assertEqual(row["issue_time"], "2026-01-31T00:00:00+00:00")

    def test_provider_failure(self):
        def broken(issue):
            raise RuntimeError("upstream error")
        with tempfile.TemporaryDirectory() as folder, redirect_stdout(io.StringIO()):
            result = run_backtest(folder, forecast_fn=broken, end="2026-01-31")
            self.assertEqual(result["accepted_runs"], 0)
            self.assertFalse(result["complete"])

    def test_single_run_has_real_analysis(self):
        with tempfile.TemporaryDirectory() as folder, redirect_stdout(io.StringIO()):
            result = run(datetime(2026, 1, 31), Path(folder) / "single.json")
            self.assertEqual(result["mode"], "stub")
            self.assertTrue(result["analysis"]["accepted"])
            self.assertEqual(result["analysis"]["summary"]["source"], "rules")


if __name__ == "__main__":
    unittest.main()
