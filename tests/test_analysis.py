import json
import os
from pathlib import Path
import tempfile
import unittest
from datetime import date, datetime, timedelta, timezone
from unittest.mock import patch

from agent import analyze, demo_provider, run_backtest, summarize, validate_provenance


class AnalysisTests(unittest.TestCase):
    def setUp(self):
        self.t = datetime(2026, 2, 1, tzinfo=timezone(timedelta(hours=5)))

    def rows(self, values):
        return [{"turbine_id": "1", "valid_time": self.t + timedelta(hours=i), "power": p}
                for i, p in enumerate(values)]

    def check_values(self, values):
        rows = self.rows(values)
        return analyze(rows, [("1", r["valid_time"]) for r in rows])[0]

    def test_boundaries_and_warnings(self):
        report = self.check_values([0, 1])
        self.assertTrue(report["accepted"])
        self.assertEqual(report["warnings"][0]["rule"], "hourly_jump")

    def test_invalid_values(self):
        for value in [None, "", float("nan"), float("inf"), -0.01, 1.01, True]:
            with self.subTest(value=value):
                self.assertFalse(self.check_values([value])["accepted"])

    def test_missing_duplicate_and_naive(self):
        expected = [("1", self.t), ("1", self.t + timedelta(hours=1))]
        report, _ = analyze(self.rows([0.2]) * 2, expected)
        self.assertEqual({e["rule"] for e in report["errors"]}, {"duplicate", "missing_hour"})
        rows = self.rows([0.2])
        rows[0]["valid_time"] = "2026-02-01T00:00:00"
        self.assertFalse(analyze(rows, expected)[0]["accepted"])

    def test_comparison_matches_timestamp_and_turbine(self):
        key = self.t.astimezone(timezone.utc).isoformat()
        previous = {("1", key): 0.1, ("2", key): 0.9}
        report, _ = analyze(self.rows([0.4]), [("1", self.t)], previous)
        self.assertEqual(report["overlap_points"], 1)
        self.assertAlmostEqual(report["max_abs_revision"], 0.3)

    def test_zero_previous_and_identical_forecast(self):
        key = self.t.astimezone(timezone.utc).isoformat()
        report, _ = analyze(self.rows([0]), [("1", self.t)], {("1", key): 0})
        self.assertEqual(report["overlap_points"], 1)
        self.assertEqual(report["changed_points"], 0)

    def test_missing_key_fallback(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(summarize(self.check_values([0.2]), "model")["source"], "rules")

    def test_llm_failure_fallback(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test"}), patch.dict("sys.modules", {"openai": None}):
            result = summarize(self.check_values([0.2]), "model")
            self.assertEqual(result["source"], "rules")
            self.assertIn("llm_error", result)

    def test_future_weather_and_observations_rejected(self):
        batch = demo_provider(self.t, [self.t + timedelta(hours=1)], ["1"])
        batch["kind"] = "archived_forecast"
        validate_provenance(batch, self.t)
        for field in ["weather_issued_at", "weather_available_at", "trained_until"]:
            altered = dict(batch, **{field: (self.t + timedelta(hours=1)).isoformat()})
            with self.assertRaises(ValueError):
                validate_provenance(altered, self.t)
        for kind in ["observations", "reanalysis", "synthetic_demo"]:
            with self.assertRaises(ValueError):
                validate_provenance(dict(batch, kind=kind), self.t)


class BacktestTests(unittest.TestCase):
    def test_full_february_and_revision_history(self):
        with tempfile.TemporaryDirectory() as directory:
            result = run_backtest(demo_provider, directory, allow_demo=True)
            self.assertEqual(result["runs"], 28)
            self.assertEqual(result["accepted_runs"], 28)
            self.assertEqual(result["february_points"], 1344)
            self.assertEqual(result["replaced_points"], 1296)
            records = [json.loads(s) for s in (Path(directory) / "runs.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual(records[0]["replaced_points"], 0)
            self.assertTrue(all(r["replaced_points"] == 48 for r in records[1:]))
            latest = json.loads((Path(directory) / "february_latest.json").read_text(encoding="utf-8"))
            self.assertEqual(len({(r["turbine_id"], r["valid_time"]) for r in latest}), 1344)
            feb2 = next(r for r in latest if r["valid_time"].startswith("2026-02-02T00") and r["turbine_id"] == "1")
            self.assertTrue(feb2["as_of"].startswith("2026-02-01"))
            feb28 = next(r for r in latest if r["valid_time"].startswith("2026-02-28T23"))
            self.assertTrue(feb28["as_of"].startswith("2026-02-27"))
            with self.assertRaises(FileExistsError):
                run_backtest(demo_provider, directory, allow_demo=True)

    def test_rejected_batch_keeps_old_values_and_continues(self):
        calls = []
        def provider(as_of, times, turbines):
            calls.append(as_of.date())
            batch = demo_provider(as_of, times, turbines)
            if as_of.date() == date(2026, 2, 1):
                batch["rows"][0]["power"] = None
            return batch
        with tempfile.TemporaryDirectory() as directory:
            result = run_backtest(provider, directory, allow_demo=True, end=date(2026, 2, 2))
            self.assertEqual(len(calls), 3)
            self.assertEqual(result["accepted_runs"], 2)
            latest = json.loads((Path(directory) / "february_latest.json").read_text(encoding="utf-8"))
            self.assertTrue(all(r["as_of"].startswith("2026-01-31") for r in latest
                                if r["valid_time"].startswith("2026-02-02")))

    def test_provider_exception_is_logged(self):
        def broken(*args):
            raise RuntimeError("upstream unavailable")
        with tempfile.TemporaryDirectory() as directory:
            result = run_backtest(broken, directory, end=date(2026, 1, 31))
            self.assertEqual(result["accepted_runs"], 0)
            self.assertFalse(result["complete"])
            record = json.loads((Path(directory) / "runs.jsonl").read_text(encoding="utf-8"))
            self.assertEqual(record["analysis"]["errors"][0]["error_type"], "RuntimeError")


if __name__ == "__main__":
    unittest.main()
