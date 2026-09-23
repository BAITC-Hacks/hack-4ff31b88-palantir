"""Признаки из погодного прогноза. Одна функция для train, теста и агента,
чтобы признаки везде считались одинаково."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C

CALENDAR = ["lead_hour", "day_offset", "hour_local", "hour_sin", "hour_cos", "doy_sin", "doy_cos", "month"]


def add_features(df: pd.DataFrame, offset_h: int | None = None) -> pd.DataFrame:
    """Добавляет ансамблевые, физические и календарные признаки.

    На входе — выход weather_client.assemble / fetch_forecast.
    """
    df = df.copy()
    offset_h = C.utc_offset_h() if offset_h is None else offset_h
    prefixes = [p for p in C.MODELS if f"{p}_ws100" in df]

    for p in prefixes:
        ws100, ws10 = df[f"{p}_ws100"], df.get(f"{p}_ws10")
        if ws10 is not None:
            ratio = (ws100.clip(lower=0.1) / ws10.clip(lower=0.1))
            df[f"{p}_shear"] = np.log(ratio).clip(-1, 3) / np.log(10)   # показатель степени профиля
        if f"{p}_t2m" in df:
            rho = 288.15 / (df[f"{p}_t2m"] + 273.15)                    # плотность относительно стандартной
            df[f"{p}_rho_ws3"] = rho * ws100.clip(upper=25) ** 3 / 1000  # удельная мощность ветра

    # Ансамбль моделей: среднее и разброс (разброс = неопределённость прогноза)
    for v in ["ws100", "ws10", "gust10", "t2m", "wd100_sin", "wd100_cos"]:
        cols = [f"{p}_{v}" for p in prefixes if f"{p}_{v}" in df]
        if cols:
            df[f"ens_{v}"] = df[cols].mean(axis=1)
            if len(cols) > 1 and v == "ws100":
                df["ens_ws100_spread"] = df[cols].max(axis=1) - df[cols].min(axis=1)

    local = pd.to_datetime(df["target_time"]) + pd.Timedelta(hours=offset_h)
    df["target_time_local"] = local
    df["hour_local"] = local.dt.hour
    df["hour_sin"] = np.sin(2 * np.pi * df["hour_local"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour_local"] / 24)
    doy = local.dt.dayofyear
    df["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    df["month"] = local.dt.month
    return df


def feature_columns(df: pd.DataFrame) -> list[str]:
    """Список признаков для модели (всё, кроме ключей и целевых колонок)."""
    skip = {"issue_time", "target_time", "target_time_local", "power", "power_clean",
            "p_t1", "p_t2", "ws_obs", "temp_obs", "n_turbines", "is_anomaly"}
    return [c for c in df.columns if c not in skip and pd.api.types.is_numeric_dtype(df[c])]
