import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from agent import demo_provider, run_backtest
from forecast_pipeline import ForecastPipeline, PowerCurveBaseline


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.issue = datetime(2026, 1, 31, 17, tzinfo=timezone.utc)
        self.times = tuple(self.issue + timedelta(hours=i) for i in range(48))
        index = pd.date_range("2026-01-01", "2026-02-02", freq="h")
        self.history = pd.DataFrame({"ws_obs": np.arange(len(index)) % 20 + 1,
                                     "power": (np.arange(len(index)) % 20) / 20}, index=index)
        index = pd.date_range("2026-01-30", "2026-03-02", freq="h")
        self.archives = {"gfs": pd.DataFrame({f"ws100_d{n}": np.full(len(index), 8.0 + n)
                                             for n in (1, 2, 3)}, index=index)}

    def test_real_weather_module_contract(self):
        provider = ForecastPipeline(observations=self.history, weather="archive", archives=self.archives, scada_offset=7)
        result = provider(self.issue, self.times, ("station",))
        self.assertEqual(len(result["rows"]), 48)
        self.assertEqual(result["rows"][0]["valid_time"], self.issue.isoformat())
        self.assertLessEqual(pd.Timestamp(result["trained_until"]), pd.Timestamp(self.issue))
        self.assertLessEqual(pd.Timestamp(result["weather_available_at"]), pd.Timestamp(self.issue))

    def test_future_observations_do_not_change_baseline(self):
        features = pd.DataFrame({"ens_ws100": [7.0, 8.0]})
        one, _ = PowerCurveBaseline(self.history).predict(features, self.issue)
        altered = self.history.copy()
        altered.loc[altered.index + pd.Timedelta(hours=1) > pd.Timestamp(self.issue).tz_localize(None), "power"] = 0.99
        two, _ = PowerCurveBaseline(altered).predict(features, self.issue)
        np.testing.assert_array_equal(one, two)

    def test_missing_wind_rejects(self):
        archives = {"gfs": self.archives["gfs"] * np.nan}
        provider = ForecastPipeline(observations=self.history, weather="archive", archives=archives)
        with self.assertRaisesRegex(ValueError, "wind"):
            provider(self.issue, self.times, ("station",))

    def test_future_model_rejected_before_predict(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metadata.json"
            path.write_text(json.dumps({"training_data_available_at": "2026-02-01T00:00:00Z"}))
            with patch("forecast_pipeline.importlib.import_module") as loader:
                provider = ForecastPipeline(model_entry="fake:predict", model_metadata=path,
                                            weather="archive", archives=self.archives)
                with self.assertRaisesRegex(ValueError, "unavailable"):
                    provider(self.issue, self.times, ("station",))
                loader.return_value.predict.assert_not_called()

    def test_full_local_february(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            result = run_backtest(demo_provider, directory, turbines=("station",), offset_hours=0,
                                  issue_hour=17, target_start="issue", evaluation_offset_hours=7, allow_demo=True)
            self.assertTrue(result["complete"])
            self.assertEqual(result["february_points"], 672)
            self.assertEqual(result["replaced_points"], 648)
            latest = pd.read_csv(Path(directory) / "february_latest.csv")
            self.assertEqual(latest.valid_time.iloc[0], "2026-02-01T00:00:00+07:00")
            self.assertEqual(latest.valid_time.iloc[-1], "2026-02-28T23:00:00+07:00")
            history = pd.read_csv(Path(directory) / "forecast_history.csv")
            self.assertEqual(history.lead_end_hours.min(), 1)
            self.assertEqual(history.lead_end_hours.max(), 48)

    def test_pipeline_and_runner_together(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            provider = ForecastPipeline(observations=self.history, weather="archive", archives=self.archives)
            result = run_backtest(provider, directory, turbines=("station",), offset_hours=0,
                                  issue_hour=17, target_start="issue", evaluation_offset_hours=7)
            self.assertEqual(result["accepted_runs"], 28)
            self.assertTrue(result["complete"])

    def test_team_power_pred_contract(self):
        path = Path(__file__).resolve().parents[1] / "model/baseline_metadata.json"
        provider = ForecastPipeline(model_entry="model.baseline_nwp:predict", model_metadata=path,
                                    weather="archive", archives=self.archives)
        batch = provider(self.issue, self.times, ("station",))
        self.assertEqual(len(batch["rows"]), 48)
        self.assertTrue(all(0 <= row["power"] <= 1 for row in batch["rows"]))


class PlotTests(unittest.TestCase):
    def test_fact_alignment_and_unavailable_actuals(self):
        from plot_results import plot
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            pd.DataFrame({"turbine_id": ["station"] * 2,
                          "valid_time": ["2026-02-01T00:00:00Z", "2026-02-01T01:00:00Z"],
                          "power": [0.1, 0.2]}).to_csv(path / "forecast.csv", index=False)
            pd.DataFrame({"turbine_id": ["station"], "valid_time": ["2026-02-01T07:00:00+07:00"],
                          "actual_power": [0.3]}).to_csv(path / "facts.csv", index=False)
            matched = plot(path / "forecast.csv", path / "facts.csv", path / "chart")
            self.assertEqual(matched["matched_actual_points"], 1)
            self.assertTrue((path / "chart/turbine_station.png").exists())
            missing = plot(path / "forecast.csv", None, path / "only_forecast")
            self.assertEqual(missing["matched_actual_points"], 0)
