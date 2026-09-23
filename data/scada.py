"""Загрузка и очистка SCADA двух турбин, агрегация в часы (UTC)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C

COLS = ["id", "t_local", "ws", "p", "temp"]
MIN_RECORDS_PER_HOUR = 4   # из 6 десятиминуток


def find_raw_files(raw_dir: Path = C.RAW_DIR) -> dict[str, Path]:
    """Ищет CSV организаторов в data/raw по подстроке 'turbine 1' / 'turbine_1'."""
    out = {}
    for f in sorted(raw_dir.glob("*.csv")):
        name = f.name.lower().replace("_", " ")
        for k in ("1", "2"):
            if f"turbine {k}" in name:
                out[f"t{k}"] = f
    missing = {"t1", "t2"} - out.keys()
    if missing:
        raise FileNotFoundError(f"Не нашёл CSV для {missing} в {raw_dir}. Положи файлы организаторов туда.")
    return out


def load_turbine(path: Path) -> pd.DataFrame:
    """10-минутные данные: t_local, ws (м/с), p (0..1), temp (°C)."""
    df = pd.read_csv(path)
    df.columns = COLS
    df["t_local"] = pd.to_datetime(df["t_local"])
    df = df.drop(columns="id").drop_duplicates("t_local").sort_values("t_local")
    df["p"] = df["p"].clip(0, 1)
    return df.reset_index(drop=True)


def to_hourly_local(df: pd.DataFrame) -> pd.DataFrame:
    """Средние по часу [h, h+1) в местном времени SCADA."""
    g = df.set_index("t_local").resample("1h")
    out = g[["ws", "p", "temp"]].mean()
    out["n"] = g["p"].count()
    out.loc[out["n"] < MIN_RECORDS_PER_HOUR, ["ws", "p", "temp"]] = np.nan
    return out


def power_curve(hourly: pd.DataFrame, step: float = 0.5) -> pd.Series:
    """Эмпирическая кривая мощности: медиана p по бинам скорости ветра."""
    ok = hourly.dropna(subset=["ws", "p"])
    bins = (ok["ws"] / step).round() * step
    curve = ok.groupby(bins)["p"].median()
    return curve[ok.groupby(bins).size() >= 20]


def flag_anomalies(hourly: pd.DataFrame, curve: pd.Series) -> pd.Series:
    """Простои и ограничения: мощность намного ниже кривой при заметном ветре."""
    expected = np.interp(hourly["ws"].fillna(0), curve.index.to_numpy(), curve.to_numpy())
    shortfall = expected - hourly["p"]
    return (hourly["ws"] >= 4.5) & (shortfall > np.maximum(0.15, 0.5 * expected))


def station_hourly(offset_h: int, raw_dir: Path = C.RAW_DIR) -> pd.DataFrame:
    """Почасовая таблица станции, индекс target_time — начало часа в UTC.

    power — средняя нормализованная мощность турбин с данными за час.
    power_clean — то же, но без часов-аномалий (простои/ограничения):
    для обучения лучше брать её.
    """
    files = find_raw_files(raw_dir)
    parts = {}
    for k, f in files.items():
        h = to_hourly_local(load_turbine(f))
        h["anom"] = flag_anomalies(h, power_curve(h))
        parts[k] = h
    df = pd.concat(parts, axis=1)
    df.columns = [f"{c}_{k}" for k, c in df.columns]

    p = df[["p_t1", "p_t2"]]
    anom = df[["anom_t1", "anom_t2"]].fillna(False).astype(bool).to_numpy()
    p_clean = p.where(~anom)
    out = pd.DataFrame(index=df.index)
    out["p_t1"], out["p_t2"] = df["p_t1"], df["p_t2"]
    out["power"] = p.mean(axis=1)
    out["power_clean"] = p_clean.mean(axis=1)
    out["ws_obs"] = df[["ws_t1", "ws_t2"]].mean(axis=1)
    out["temp_obs"] = df[["temp_t1", "temp_t2"]].mean(axis=1)
    out["n_turbines"] = p.notna().sum(axis=1)
    out["is_anomaly"] = anom.any(axis=1)

    out.index = out.index - pd.Timedelta(hours=offset_h)  # местное -> UTC
    out.index.name = "target_time"
    out.insert(0, "target_time_local", out.index + pd.Timedelta(hours=offset_h))
    return out.dropna(subset=["power"])


def local_hourly_for_tz_check(raw_dir: Path = C.RAW_DIR) -> pd.DataFrame:
    """Средний по станции ws/temp в местном времени (для определения сдвига)."""
    files = find_raw_files(raw_dir)
    hs = [to_hourly_local(load_turbine(f))[["ws", "temp"]] for f in files.values()]
    return pd.concat(hs).groupby(level=0).mean()
