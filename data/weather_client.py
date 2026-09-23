"""Архивные прогнозы погоды Open-Meteo без утечки будущего.

Источник: Open-Meteo Previous Runs API. Переменная `<var>_previous_dayN`
хранит значение, которое модель NWP предсказала за N*24 ч до момента
(из прогона, стартовавшего ≥ N*24 ч назад).

Правило «as-of»: для прогноза, выпущенного в момент T, цель с лидом
L часов берётся из previous_dayN, где N = ceil((L + PUBLISH_DELAY_H) / 24).
Такой прогон стартовал не позже T - PUBLISH_DELAY_H, то есть был
опубликован к моменту T. Фактическая погода и свежие прогоны не
используются.

Главные функции:
    fetch_forecast(issue_time)            -> прогноз погоды на 48 ч для агента
    download_archive(model, start, end)   -> сырой архив day1..day3
    assemble(archives, issue_times)       -> признаки для многих выпусков сразу
"""
from __future__ import annotations

import hashlib
import math
import time
import warnings
from typing import Iterable

import numpy as np
import pandas as pd
import requests

from . import config as C

API_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"
CHUNK_DAYS = 60
TIMEOUT_S = 60


# ---------------------------------------------------------------- lead -> dayN
def day_offset_for_lead(lead_h: int, delay_h: int = C.PUBLISH_DELAY_H) -> int:
    """Какой previous_dayN можно использовать для лида lead_h без утечки."""
    n = math.ceil((lead_h + delay_h) / 24)
    if n > max(C.DAY_OFFSETS):
        raise ValueError(f"lead {lead_h} ч требует previous_day{n}, а скачиваем до day{max(C.DAY_OFFSETS)}")
    return n


# ------------------------------------------------------------------ скачивание
def _request(params: dict, retries: int = 5) -> dict:
    for attempt in range(retries):
        try:
            r = requests.get(API_URL, params=params, timeout=TIMEOUT_S)
        except requests.RequestException as e:  # сеть
            wait = 5 * (attempt + 1)
            print(f"  сеть: {e}; повтор через {wait} с")
            time.sleep(wait)
            continue
        if r.status_code == 429:
            print("  лимит Open-Meteo (429), ждём 60 с")
            time.sleep(60)
            continue
        if r.status_code >= 500:
            time.sleep(5 * (attempt + 1))
            continue
        data = r.json()
        if r.status_code != 200 or data.get("error"):
            raise RuntimeError(f"Open-Meteo {r.status_code}: {data.get('reason', data)}\nURL: {r.url}")
        return data
    raise RuntimeError(f"Open-Meteo не ответил после {retries} попыток")


def _fetch_chunk(model: str, start: pd.Timestamp, end: pd.Timestamp,
                 lat: float, lon: float) -> pd.DataFrame:
    api_vars = [f"{v}_previous_day{n}" for v in C.VARIABLES for n in C.DAY_OFFSETS]
    key = f"{model}|{lat:.4f}|{lon:.4f}|{start:%Y%m%d}|{end:%Y%m%d}|{','.join(api_vars)}"
    cache = C.CACHE_DIR / f"{model}_{start:%Y%m%d}_{end:%Y%m%d}_{hashlib.md5(key.encode()).hexdigest()[:8]}.csv"
    if cache.exists():
        return pd.read_csv(cache, index_col=0, parse_dates=True)

    data = _request({
        "latitude": lat,
        "longitude": lon,
        "hourly": ",".join(api_vars),
        "models": model,
        "start_date": f"{start:%Y-%m-%d}",
        "end_date": f"{end:%Y-%m-%d}",
        "wind_speed_unit": "ms",
        "timezone": "GMT",
    })
    h = data["hourly"]
    df = pd.DataFrame({k: v for k, v in h.items() if k != "time"},
                      index=pd.to_datetime(h["time"]))
    df.index.name = "valid_time"
    rename = {f"{v}_previous_day{n}": f"{s}_d{n}"
              for v, s in C.VARIABLES.items() for n in C.DAY_OFFSETS}
    df = df.rename(columns=rename).astype(float)

    # Кэшируем только полностью прошедшие периоды: свежие могут дополниться.
    if end < pd.Timestamp.now(tz="UTC").tz_localize(None).normalize() - pd.Timedelta(days=4):
        C.CACHE_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(cache)
    return df


def download_archive(model: str, start, end, lat: float = C.SITE_LAT,
                     lon: float = C.SITE_LON, verbose: bool = True) -> pd.DataFrame:
    """Архив previous_day1..3 одной модели за [start, end] (UTC, почасово)."""
    start, end = pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize()
    parts = []
    cur = start
    while cur <= end:
        stop = min(cur + pd.Timedelta(days=CHUNK_DAYS - 1), end)
        if verbose:
            print(f"  {model}: {cur:%Y-%m-%d} … {stop:%Y-%m-%d}")
        parts.append(_fetch_chunk(model, cur, stop, lat, lon))
        cur = stop + pd.Timedelta(days=1)
    df = pd.concat(parts).sort_index()
    return df[~df.index.duplicated(keep="first")]


# ------------------------------------------------------------ сборка признаков
def _interval_values(arch: pd.DataFrame, t0: pd.DatetimeIndex, col: str) -> np.ndarray:
    """Среднее мгновенных значений на границах часа [t0, t0+1h]."""
    a = arch[col].reindex(t0).to_numpy()
    b = arch[col].reindex(t0 + pd.Timedelta(hours=1)).to_numpy()
    return np.nanmean(np.vstack([a, b]), axis=0) if len(t0) else a


def assemble(archives: dict[str, pd.DataFrame], issue_times: Iterable,
             horizon: int = C.HORIZON_H, delay_h: int = C.PUBLISH_DELAY_H) -> pd.DataFrame:
    """Строки (issue_time, target_time, lead_hour) + погодные признаки as-of.

    target_time — начало целевого часа в UTC; цель — интервал
    [issue_time + (lead-1) ч, issue_time + lead ч).
    archives: {"ecmwf": df, "gfs": df} из download_archive.
    """
    issues = pd.DatetimeIndex(pd.to_datetime(list(issue_times)))
    leads = np.arange(1, horizon + 1)
    grid = pd.DataFrame({
        "issue_time": np.repeat(issues.values, len(leads)),
        "lead_hour": np.tile(leads, len(issues)),
    })
    grid["target_time"] = grid["issue_time"] + pd.to_timedelta(grid["lead_hour"] - 1, unit="h")
    grid["day_offset"] = [day_offset_for_lead(int(l), delay_h) for l in grid["lead_hour"]]

    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Mean of empty slice")
        for prefix, arch in archives.items():
            for n in sorted(grid["day_offset"].unique()):
                m = grid["day_offset"].to_numpy() == n
                t0 = pd.DatetimeIndex(grid.loc[m, "target_time"])
                for short in C.VARIABLES.values():
                    col = f"{short}_d{n}"
                    if col not in arch:
                        continue
                    if short == "wd100":  # направление: усредняем как вектор
                        rad = np.deg2rad(arch[col])
                        tmp = pd.DataFrame({"s": np.sin(rad), "c": np.cos(rad)}, index=arch.index)
                        grid.loc[m, f"{prefix}_wd100_sin"] = _interval_values(tmp, t0, "s")
                        grid.loc[m, f"{prefix}_wd100_cos"] = _interval_values(tmp, t0, "c")
                    else:
                        grid.loc[m, f"{prefix}_{short}"] = _interval_values(arch, t0, col)
    return grid


# Фактическое время выхода прогонов после старта (среднее по наблюдениям
# wethr.net/model-schedule): ECMWF IFS open data ≈ 7 ч 55 мин, GFS ≈ 5 ч.
# Плюс час на обработку в Open-Meteo. Прогоны стартуют каждые 6 ч (00/06/12/18 UTC).
RUN_INTERVAL_H = 6
REAL_PUBLICATION_H = {"ecmwf": 7 + 55 / 60, "gfs": 5.0}
INGEST_MARGIN_H = 1.0


def check_no_leakage(df: pd.DataFrame, delay_h: int = C.PUBLISH_DELAY_H) -> dict[str, float]:
    """Проверяет, что каждый использованный прогон реально вышел до issue_time.

    Для цели v с day_offset=N Open-Meteo берёт прогон, стартовавший не позже
    v − 24·N, то есть самый поздний возможный старт — это v − 24·N, округлённое
    вниз до сетки прогонов (00/06/12/18 UTC). К нему прибавляем фактическую
    задержку публикации модели и час на обработку. Возвращает минимальный запас
    в часах по каждой модели; если запас отрицательный — AssertionError.
    """
    last_instant = df["target_time"] + pd.Timedelta(hours=1)  # правая граница часа
    run_start = (last_instant - pd.to_timedelta(24 * df["day_offset"], unit="h")).dt.floor(f"{RUN_INTERVAL_H}h")
    margins = {}
    for model, pub_h in REAL_PUBLICATION_H.items():
        available = run_start + pd.Timedelta(hours=pub_h + INGEST_MARGIN_H)
        margin_h = (df["issue_time"] - available) / pd.Timedelta(hours=1)
        margins[model] = round(float(margin_h.min()), 2)
        bad = margin_h < 0
        if bad.any():
            raise AssertionError(f"Утечка ({model}): {int(bad.sum())} строк используют прогон, "
                                 f"опубликованный после issue_time")
    # Правило выбора day_offset само по себе тоже не должно нарушаться
    rule_start = last_instant - pd.to_timedelta(24 * df["day_offset"], unit="h")
    if (rule_start + pd.Timedelta(hours=delay_h) > df["issue_time"]).any():
        raise AssertionError("Нарушено правило N = ceil((L + delay) / 24)")
    return margins


# ------------------------------------------------------------- API для агента
def fetch_forecast(issue_time, models: dict[str, str] | None = None,
                   horizon: int = C.HORIZON_H, lat: float = C.SITE_LAT,
                   lon: float = C.SITE_LON) -> pd.DataFrame:
    """Прогноз погоды на horizon часов в том виде, как он был известен в issue_time.

    issue_time — UTC (naive или tz-aware). Возвращает 48 строк:
    issue_time, target_time, lead_hour, day_offset, <model>_<var>...
    """
    models = models or C.MODELS
    t = pd.Timestamp(issue_time)
    if t.tzinfo is not None:
        t = t.tz_convert("UTC").tz_localize(None)
    start = (t - pd.Timedelta(days=1)).normalize()
    end = (t + pd.Timedelta(hours=horizon + 1)).normalize()
    archives = {p: download_archive(m, start, end, lat, lon, verbose=False) for p, m in models.items()}
    df = assemble(archives, [t], horizon)
    check_no_leakage(df)
    return df


if __name__ == "__main__":
    # Быстрая проверка: python -m data.weather_client 2026-02-01T00:00
    import sys
    ts = sys.argv[1] if len(sys.argv) > 1 else "2026-02-01T00:00"
    out = fetch_forecast(ts)
    pd.set_option("display.width", 200)
    print(out.head(6).round(2).to_string())
    print("…", len(out), "строк; NaN по колонкам:")
    print(out.isna().mean().round(3)[lambda s: s > 0].to_string() or "нет")
