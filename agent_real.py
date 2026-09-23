"""Adapters for the team's weather, features and prediction modules."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from data import config as C
from data.features import add_features
from data.weather_client import assemble, check_no_leakage, download_archive, fetch_forecast


class RealForecast:
    def __init__(self, mode, first_issue, *, archives=None, training_path=None,
                 trained_through=None):
        self.mode = mode
        self.archives = archives
        first = pd.Timestamp(first_issue)
        first = first.tz_localize('UTC') if first.tzinfo is None else first.tz_convert('UTC')
        self.first_issue = first
        self.provenance = {'weather': 'Open-Meteo Previous Runs', 'model': mode,
                           'utc_offset_h': C.utc_offset_h()}
        if mode == 'baseline':
            from model.baseline_nwp import fit_curves, predict
            path = Path(training_path or C.PROCESSED_DIR / 'train.parquet')
            train = pd.read_parquet(path)
            available = pd.to_datetime(train['target_time'], utc=True) + pd.Timedelta(hours=1)
            train = train.loc[available <= first].copy()
            if train.empty:
                raise ValueError('No training observations available at first issue')
            curves = fit_curves(train)
            if any(not curve['ws'] for curve in curves.values()):
                raise ValueError('Insufficient training observations for power curves')
            self.predict = lambda frame: predict(frame, curves)
            self.features = ['ens_ws100']
            self.provenance.update(training_rows=len(train),
                training_available_through=available.loc[train.index].max().isoformat(),
                training_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        elif mode == 'lightgbm':
            from model import train as module
            if not module.MODEL_PATH.exists() or not module.FEATURES_PATH.exists():
                raise FileNotFoundError('Need model/lightgbm_model.txt and model/lightgbm_features.json from Person 2')
            if trained_through is None:
                raise ValueError('Specify --trained-through: UTC availability of the last training label, confirmed by Person 2')
            cutoff = pd.Timestamp(trained_through)
            if cutoff.tzinfo is None:
                raise ValueError('--trained-through must contain a UTC offset')
            if cutoff > first:
                raise ValueError('Model training includes observations after the first issue')
            import lightgbm as lgb
            model = lgb.Booster(model_file=str(module.MODEL_PATH))
            self.features = json.loads(module.FEATURES_PATH.read_text(encoding='utf-8'))
            if model.feature_name() != self.features:
                raise ValueError('Saved model and feature list do not match')
            self.predict = lambda frame: module.predict(frame, model)
            self.provenance.update(training_available_through=cutoff.isoformat(),
                model_sha256=hashlib.sha256(module.MODEL_PATH.read_bytes()).hexdigest())
        else:
            raise ValueError('Unknown real model mode')

    def __call__(self, issue_time):
        issue = pd.Timestamp(issue_time)
        issue = issue.tz_localize('UTC') if issue.tzinfo is None else issue.tz_convert('UTC')
        if issue < self.first_issue:
            raise ValueError('Issue precedes training cutoff')
        naive = issue.tz_localize(None)
        weather = (fetch_forecast(naive) if self.archives is None
                   else assemble(self.archives, [naive]))
        check_no_leakage(weather)
        frame = add_features(weather)
        missing = set(self.features) - set(frame.columns)
        if missing:
            raise ValueError('Missing model features: ' + ', '.join(sorted(missing)))
        if not np.isfinite(frame[self.features].to_numpy(dtype=float)).all():
            raise ValueError('Missing or nonfinite weather features; refusing implicit imputation')
        prediction = self.predict(frame)
        if len(prediction) != len(frame) or not prediction['target_time'].equals(frame['target_time']):
            raise ValueError('Prediction changed target timestamps')
        return prediction[['target_time', 'power_pred']].to_dict('records')


def build_forecaster(mode, start, end, *, trained_through=None):
    first = pd.Timestamp(start) + pd.Timedelta(hours=C.ISSUE_HOUR_UTC)
    # Validate model before any network call.
    provider = RealForecast(mode, first, trained_through=trained_through)
    begin = (first - pd.Timedelta(days=1)).normalize()
    stop = (pd.Timestamp(end) + pd.Timedelta(hours=C.ISSUE_HOUR_UTC + C.HORIZON_H + 1)).normalize()
    provider.archives = {p: download_archive(m, begin, stop) for p, m in C.MODELS.items()}
    return provider
