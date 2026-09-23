"""Integration of the team's data module with a model and the backtest runner."""
from __future__ import annotations

import importlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from data import config as C
from data.features import add_features
from data.weather_client import assemble, check_no_leakage, download_archive, fetch_forecast


def utc_naive(value):
    ts = pd.Timestamp(value)
    return ts.tz_convert("UTC").tz_localize(None) if ts.tzinfo else ts


def iso_utc(value):
    return utc_naive(value).tz_localize("UTC").isoformat()


class PowerCurveBaseline:
    """Simple empirical SCADA curve, explicitly a baseline rather than team's ML."""

    def __init__(self, observations):
        self.observations = observations.copy()
        self.observations.index = pd.DatetimeIndex([utc_naive(t) for t in self.observations.index])

    def predict(self, frame, as_of):
        # An observation describes a whole hour and becomes usable at its end.
        history = self.observations[self.observations.index + pd.Timedelta(hours=1) <= utc_naive(as_of)]
        history = history.dropna(subset=["ws_obs", "power"])
        history = history[history.power.between(0, 1) & (history.ws_obs >= 0)]
        if len(history) < 24:
            raise ValueError("Not enough past observations for the power-curve baseline")
        bins = (history.ws_obs / 0.5).round() * 0.5
        curve = history.groupby(bins).power.median().sort_index()
        if len(curve) < 2:
            raise ValueError("Not enough wind bins for the baseline")
        # Reference observed wind is at the turbine sensor; this mismatch with NWP
        # wind is an acknowledged baseline limitation, not a tuned physical curve.
        values = np.interp(frame.ens_ws100.to_numpy(), curve.index, curve.to_numpy()).clip(0, 1)
        return values, history.index.max() + pd.Timedelta(hours=1)


class ForecastPipeline:
    def __init__(self, *, observations=None, model_entry=None, model_metadata=None,
                 weather="fetch", archives=None, scada_offset=None):
        self.weather = weather
        self.archives = archives
        self.offset = C.utc_offset_h() if scada_offset is None else scada_offset
        self.baseline = PowerCurveBaseline(observations) if observations is not None else None
        self.model = None
        self.metadata = None
        if model_entry:
            module, function = model_entry.split(":", 1)
            self.model = getattr(importlib.import_module(module), function)
            if model_metadata is None:
                raise ValueError("--model-metadata is required with --model")
            self.metadata = json.loads(Path(model_metadata).read_text(encoding="utf-8"))
            # Includes preprocessing/feature selection as well as model fitting.
            pd.Timestamp(self.metadata["training_data_available_at"])
        if self.model is None and self.baseline is None:
            raise ValueError("Provide a model or explicitly select the baseline")

    def __call__(self, as_of, valid_times, turbine_ids):
        if tuple(turbine_ids) != ("station",):
            raise ValueError("Current team data module forecasts one station series")
        issue = utc_naive(as_of)
        if self.weather == "archive":
            if not self.archives:
                raise ValueError("No archived weather loaded")
            wx = assemble(self.archives, [issue], horizon=48)
        elif self.weather == "fetch":
            wx = fetch_forecast(issue, horizon=48)
        else:
            raise ValueError("Unknown weather source")
        if len(wx) != 48 or wx.target_time.duplicated().any():
            raise ValueError("Weather must contain 48 unique target hours")
        if set(wx.target_time.map(iso_utc)) != {iso_utc(t) for t in valid_times}:
            raise ValueError("Weather horizon differs from virtual clock")
        if wx[["issue_time", "target_time", "day_offset"]].isna().any().any():
            raise ValueError("Weather timing metadata contains missing values")
        if not wx.issue_time.eq(issue).all() or not wx.day_offset.isin(C.DAY_OFFSETS).all():
            raise ValueError("Weather issue time or day offset mismatch")
        check_no_leakage(wx)
        prepared = add_features(wx, self.offset)
        if "ens_ws100" not in prepared or not np.isfinite(prepared.ens_ws100).all():
            raise ValueError("Missing or nonfinite forecast wind speed")
        if self.model:
            available = utc_naive(self.metadata["training_data_available_at"])
            if available > issue:
                raise ValueError("Model training/preprocessing uses data unavailable at virtual issue time")
            # Never pass actuals/targets to the model. Frame contains weather only.
            prediction = self.model(prepared.copy())
            if not isinstance(prediction, pd.DataFrame):
                raise ValueError("Model must return DataFrame with target_time and power_pred (or power)")
            column = "power_pred" if "power_pred" in prediction else "power"
            if column not in prediction:
                raise ValueError("Model prediction column power_pred/power is missing")
            if prediction.target_time.duplicated().any() or len(prediction) != 48:
                raise ValueError("Model must preserve exactly 48 unique target hours")
            rows = [{"turbine_id": "station", "valid_time": iso_utc(t), "power": p}
                    for t, p in zip(prediction.target_time, prediction[column])]
            model_name = self.metadata.get("model_name", "team_model")
        else:
            values, available = self.baseline.predict(prepared, issue)
            rows = [{"turbine_id": "station", "valid_time": iso_utc(t), "power": float(p)}
                    for t, p in zip(prepared.target_time, values)]
            model_name = "power_curve_baseline"
        # These are conservative bounds under the data module's day-offset rule,
        # not exact NWP issue timestamps returned by Open-Meteo.
        run_bound = (wx.target_time + pd.Timedelta(hours=1) - pd.to_timedelta(24 * wx.day_offset, unit="h")).max()
        return {"kind": "archived_forecast", "source": "Open-Meteo Previous Runs",
                "weather_run_id": {"models": C.MODELS, "day_offsets": sorted(map(int, wx.day_offset.unique())),
                                   "provenance_type": "conservative_asof_bounds", "model": model_name},
                "weather_issued_at": iso_utc(run_bound),
                "weather_available_at": iso_utc(run_bound + pd.Timedelta(hours=C.PUBLISH_DELAY_H)),
                "trained_until": iso_utc(available), "rows": rows}


def load_archives(first, last):
    """Download each model once, then select as-of inputs on every virtual day."""
    start = utc_naive(first).normalize() - pd.Timedelta(days=1)
    end = utc_naive(last).normalize() + pd.Timedelta(days=3)
    return {prefix: download_archive(model, start, end) for prefix, model in C.MODELS.items()}
