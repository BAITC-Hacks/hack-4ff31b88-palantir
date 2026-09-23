"""Contract checks for the read-only data boundary used by the web agent."""
import csv
import json
from pathlib import Path
import tempfile
import unittest

from repository_data import RepositoryData, RepositoryDataError


class RepositoryDataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.run_id = "artifacts/agent_lgbm_llm"
        self.run = self.root / self.run_id
        self.run.mkdir(parents=True)
        (self.run / "summary.json").write_text(json.dumps({
            "mode": "lightgbm", "scada_offset_h": 6, "runs": 2, "accepted_runs": 1,
        }), encoding="utf-8")
        self.issue = "2026-01-31T00:00:00+00:00"
        self.other_issue = "2026-02-01T00:00:00+00:00"
        records = []
        for issue, accepted, source in [(self.issue, True, "llm"), (self.other_issue, False, "rules")]:
            records.append({"issue_time": issue, "analysis": {
                "accepted": accepted, "errors": [] if accepted else [{"rule": "missing"}],
                "warnings": [{"rule": "hourly_jump"}], "overlap_points": 1,
                "changed_points": 1, "mean_abs_revision": 0.2, "max_abs_revision": 0.2,
                "summary": {"source": source, "text": "Проверено."},
            }})
        (self.run / "runs.jsonl").write_text("\n".join(json.dumps(record) for record in records), encoding="utf-8")
        self.latest = [
            {"target_time": "2026-01-31T17:00:00+00:00", "target_time_local": "2026-01-31T23:00:00+06:00", "power_pred": 0.1, "issue_time": self.issue},
            {"target_time": "2026-01-31T18:00:00+00:00", "target_time_local": "2026-02-01T00:00:00+06:00", "power_pred": 0.2, "issue_time": self.issue},
            {"target_time": "2026-02-01T17:00:00+00:00", "target_time_local": "2026-02-01T23:00:00+06:00", "power_pred": 0.8, "issue_time": self.issue},
            {"target_time": "2026-02-01T18:00:00+00:00", "target_time_local": "2026-02-02T00:00:00+06:00", "power_pred": 0.9, "issue_time": self.issue},
        ]
        self.write_csv(self.run / "february_latest.csv", self.latest)
        self.history = [
            {"target_time": "2026-01-31T18:00:00+00:00", "power_pred": 0.2, "issue_time": self.issue, "lead_hour": 19, "accepted": "True"},
            {"target_time": "2026-02-01T18:00:00+00:00", "power_pred": 0.9, "issue_time": self.other_issue, "lead_hour": 19, "accepted": "False"},
        ]
        self.write_csv(self.run / "forecast_history.csv", self.history)
        self.data = RepositoryData(self.root)

    @staticmethod
    def write_csv(path, rows):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    def test_local_calendar_filter_is_inclusive_and_stats_precede_limit(self):
        result = self.data.forecast(self.run_id, start="2026-02-01", end="2026-02-01", limit=1)
        self.assertEqual(result["total_rows"], 2)
        self.assertTrue(result["truncated"])
        self.assertEqual(result["rows"][0]["target_time"], "2026-01-31T18:00:00+00:00")
        self.assertEqual(result["statistics"], {"mean_power": 0.5, "min_power": 0.2, "max_power": 0.8,
                         "min_power_time": "2026-02-01T00:00:00+06:00", "max_power_time": "2026-02-01T23:00:00+06:00"})
        self.assertEqual(result["timezone"], "UTC+6")

    def test_explicit_offset_boundaries_select_same_instant(self):
        result = self.data.forecast(self.run_id, start="2026-02-01T00:00:00+06:00", end="2026-01-31T18:00:00Z")
        self.assertEqual(result["total_rows"], 1)
        self.assertEqual(result["rows"][0]["power_pred"], 0.2)

    def test_issue_history_excludes_rejected_forecasts(self):
        accepted = self.data.forecast(self.run_id, issue_time="2026-01-31T06:00:00+06:00")
        rejected = self.data.forecast(self.run_id, issue_time=self.other_issue)
        self.assertEqual(accepted["total_rows"], 1)
        self.assertEqual(accepted["rows"][0]["lead_hour"], 19)
        self.assertEqual(accepted["sources"], [f"{self.run_id}/forecast_history.csv"])
        self.assertEqual(rejected["total_rows"], 0)
        self.assertIsNone(rejected["statistics"]["mean_power"])

    def test_counts_come_from_full_log_and_checks_filter_equivalent_offset(self):
        summary = self.data.run_summary(self.run_id)
        self.assertEqual(summary["checks"], {"total": 2, "accepted": 1, "rejected": 1, "errors": 1, "warnings": 2, "llm": 1, "rules": 1})
        result = self.data.checks(self.run_id, issue_time="2026-02-01T06:00:00+06:00")
        self.assertEqual(result["total_rows"], 1)
        self.assertFalse(result["rows"][0]["accepted"])
        self.assertEqual(result["rows"][0]["error_count"], 1)

    def test_discovery_prefers_llm_and_exposes_malformed_summary(self):
        other = self.root / "artifacts" / "aaa"
        other.mkdir()
        (other / "summary.json").write_text('{"mode":"stub"}', encoding="utf-8")
        broken = self.root / "artifacts" / "broken"
        broken.mkdir()
        (broken / "summary.json").write_text('{bad', encoding="utf-8")
        runs = self.data.list_runs()
        self.assertEqual([row["id"] for row in runs], [self.run_id, "artifacts/aaa"])
        self.assertTrue(any("broken/summary.json" in issue for issue in self.data.issues))

    def test_allowlist_rejects_traversal_absolute_and_normalized_aliases(self):
        for run_id in ("../outside", str(self.run), "artifacts/../artifacts/agent_lgbm_llm", ".env"):
            with self.subTest(run_id=run_id), self.assertRaises(RepositoryDataError):
                self.data.forecast(run_id)

    def test_escaping_symlink_cannot_be_read(self):
        with tempfile.TemporaryDirectory() as outside:
            outside_path = Path(outside) / "forecast.csv"
            outside_path.write_text("sensitive", encoding="utf-8")
            path = self.run / "february_latest.csv"
            path.unlink()
            try:
                path.symlink_to(outside_path)
            except OSError:
                self.skipTest("Создание ссылок недоступно в этой среде.")
            with self.assertRaises(RepositoryDataError):
                self.data.forecast(self.run_id)

    def test_empty_metrics_are_null_and_zero_samples_are_unavailable(self):
        self.write_csv(self.root / "model" / "metrics_february.csv", [
            {"model": "lightgbm", "horizon": "all", "nMAE_%": "", "nRMSE_%": "", "bias_%": "", "n": 0},
        ])
        self.write_csv(self.root / "model" / "metrics_lgbm.csv", [
            {"model": "lightgbm", "horizon": "all", "nMAE_%": 16.43, "nRMSE_%": 23.83, "bias_%": 5.9, "n": 1452},
        ])
        result = self.data.metrics()
        feb = next(row for row in result["rows"] if row["period"] == "2026-02")
        january = next(row for row in result["rows"] if row["period"] == "2026-01")
        self.assertFalse(feb["available"])
        self.assertIsNone(feb["nMAE_%"])
        self.assertTrue(january["available"])
        self.assertEqual(january["nMAE_%"], 16.43)
        json.dumps(result, allow_nan=False)

    def test_bad_rows_are_reported_even_outside_requested_period(self):
        self.latest[0]["power_pred"] = "NaN"
        self.write_csv(self.run / "february_latest.csv", self.latest)
        with self.assertRaises(RepositoryDataError):
            self.data.forecast(self.run_id, start="2026-02-01", end="2026-02-01")

    def test_summary_cannot_present_counts_that_disagree_with_log(self):
        path = self.run / "summary.json"
        summary = json.loads(path.read_text(encoding="utf-8"))
        summary["accepted_runs"] = 2
        path.write_text(json.dumps(summary), encoding="utf-8")
        with self.assertRaisesRegex(RepositoryDataError, "accepted_runs"):
            self.data.run_summary(self.run_id)

    def test_duplicate_timestamps_and_inconsistent_local_time_are_errors(self):
        self.write_csv(self.run / "february_latest.csv", self.latest + [self.latest[0]])
        with self.assertRaises(RepositoryDataError):
            self.data.forecast(self.run_id)
        self.latest[0]["target_time_local"] = "2026-02-01T23:00:00+06:00"
        self.write_csv(self.run / "february_latest.csv", self.latest)
        with self.assertRaises(RepositoryDataError):
            self.data.forecast(self.run_id)

    def test_invalid_dates_naive_datetimes_and_limits_are_errors(self):
        for arguments in ({"start": "2026-02-30"}, {"start": "2026-02-01T12:00:00"},
                          {"start": "2026-02-03", "end": "2026-02-01"}, {"limit": 0}, {"limit": 2001}):
            with self.subTest(arguments=arguments), self.assertRaises(RepositoryDataError):
                self.data.forecast(self.run_id, **arguments)


if __name__ == "__main__":
    unittest.main()
