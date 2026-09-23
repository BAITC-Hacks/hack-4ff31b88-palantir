"""UI contracts: read-only browsing, date filters, and explicit API submission."""
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from streamlit.testing.v1 import AppTest
except ImportError:
    AppTest = None

import repository_agent
import repository_data


class PreviewRepository:
    """A small artifact fixture with two runs and two local calendar days."""

    def __init__(self, root):
        self.issues = []

    def list_runs(self):
        return [{"id": "artifacts/demo_a", "summary": {}},
                {"id": "artifacts/demo_b", "summary": {}}]

    def run_summary(self, run_id):
        return {"summary": {"mode": "lightgbm", "accepted_runs": 2, "runs": 2,
                            "february_hours": 48, "expected_february_hours": 48},
                "checks": {"total": 2, "accepted": 2, "errors": 0, "warnings": 1, "llm": 2},
                "sources": [f"{run_id}/summary.json"]}

    def forecast(self, run_id, start=None, end=None, issue_time=None, limit=2000):
        begin = datetime(2026, 2, 1, tzinfo=timezone(timedelta(hours=6)))
        rows = []
        for hour in range(48):
            local = begin + timedelta(hours=hour)
            if start and local.date().isoformat() < start:
                continue
            if end and local.date().isoformat() > end:
                continue
            rows.append({"target_time_local": local.isoformat(),
                         "target_time": local.astimezone(timezone.utc).isoformat(),
                         "issue_time": "2026-01-31T00:00:00+00:00", "power_pred": .5})
        return {"rows": rows[:limit], "sources": [f"{run_id}/february_latest.csv"],
                "total_rows": len(rows), "truncated": len(rows) > limit,
                "statistics": {"mean_power": .5, "min_power": .5, "max_power": .5},
                "timezone": "UTC+6", "unit": "fraction_of_installed_capacity"}

    def checks(self, run_id, issue_time=None, limit=50):
        if limit > 50:
            raise repository_data.RepositoryDataError("Invalid limit")
        return {"rows": [{"issue_time": "2026-01-31T00:00:00+00:00", "accepted": True,
                           "error_count": 0, "warning_count": 1, "overlap_points": 24,
                           "changed_points": 12, "mean_abs_revision": .05, "max_abs_revision": .1,
                           "errors": [], "warnings": ["Большое расхождение погодных моделей"],
                           "summary": {"source": "llm", "text": "Сохранённое резюме"}}],
                "sources": [f"{run_id}/runs.jsonl"], "total_rows": 1, "truncated": False}

    def metrics(self):
        return {"rows": [{"model": "LightGBM", "horizon": "all", "nMAE_%": 16.4,
                           "nRMSE_%": 21.5, "bias_%": 1.2, "n": 100, "period": "2026-01"},
                          {"model": "LightGBM", "horizon": "all", "nMAE_%": None,
                           "nRMSE_%": None, "bias_%": None, "n": 0, "period": "2026-02"}],
                "sources": ["model/metrics_lgbm.csv", "model/metrics_february.csv"]}


@unittest.skipIf(AppTest is None, "Streamlit is not installed")
class RepositoryAgentUITest(unittest.TestCase):
    def setUp(self):
        self.data_patch = patch.object(repository_data, "RepositoryData", PreviewRepository)
        self.data_patch.start()
        self.addCleanup(self.data_patch.stop)
        self.api = Mock(return_value={
            "text": "В прогоне приняты два выпуска.",
            "sources": ["artifacts/demo_a/summary.json"],
            "tool_calls": [{"name": "get_run_summary", "arguments": {},
                            "sources": ["artifacts/demo_a/summary.json"]}],
            "model": "test-model", "usage": {},
        })
        self.api_patch = patch.object(repository_agent, "ask", self.api)
        self.api_patch.start()
        self.addCleanup(self.api_patch.stop)

    def app(self):
        result = AppTest.from_file(str(ROOT / "app" / "repository_agent.py"), default_timeout=20).run()
        self.assertEqual(len(result.exception), 0)
        return result

    def test_without_key_data_visible_chat_disabled_and_no_api_call(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "", "OPENAI_MODEL": "test-model"}):
            app = self.app()
            self.assertEqual(app.title[0].value, "Palantir · Агент данных")
            self.assertTrue(app.chat_input[0].disabled)
            self.assertIn("48 / 48", [metric.value for metric in app.metric])
            self.assertTrue(any("валидации" in item.value for item in app.subheader))
            self.api.assert_not_called()

    def test_date_filter_reads_inclusive_local_day(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": ""}):
            app = self.app()
            app.date_input[0].set_value((date(2026, 2, 1), date(2026, 2, 1))).run()
            self.assertEqual(len(app.exception), 0)
            self.assertEqual(next(metric.value for metric in app.metric
                                  if metric.label == "Часов"), "24")
            self.api.assert_not_called()

    def test_chat_submits_once_and_run_changes_isolate_history(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key-placeholder", "OPENAI_MODEL": "test-model"}):
            app = self.app()
            self.api.assert_not_called()
            app.chat_input[0].set_value("Подведи итоги").run()
            self.assertEqual(len(app.exception), 0)
            self.api.assert_called_once()
            self.assertEqual(self.api.call_args.kwargs["run_id"], "artifacts/demo_a")
            self.assertEqual(self.api.call_args.kwargs["history"], [])
            self.assertTrue(any(item.value == "В прогоне приняты два выпуска." for item in app.markdown))
            self.assertIn("Источники ответа", [item.label for item in app.expander])
            app.run()
            self.api.assert_called_once()
            app.sidebar.selectbox[0].set_value("artifacts/demo_b").run()
            self.assertEqual(len(app.exception), 0)
            self.assertFalse(any(item.value == "В прогоне приняты два выпуска." for item in app.markdown))
            self.api.assert_called_once()

    def test_api_failure_is_safe_and_not_retried_by_rerender(self):
        self.api.side_effect = RuntimeError("private provider details")
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key-placeholder"}):
            app = self.app()
            app.chat_input[0].set_value("Подведи итоги").run()
            self.assertEqual(len(app.exception), 0)
            self.assertTrue(any("Не удалось получить ответ" in item.value for item in app.error))
            self.assertFalse(any("private provider details" in item.value for item in app.error))
            app.run()
            self.api.assert_called_once()


if __name__ == "__main__":
    unittest.main()
