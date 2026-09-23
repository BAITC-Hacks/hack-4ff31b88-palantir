"""Бейзлайны и метрики: планка, которую должна побить основная модель.

Запуск из корня репозитория (после python -m data.build_dataset):
    python -m model.baseline_nwp

Бейзлайны:
    climatology  — средняя мощность по (месяц, час суток) на истории
    persistence  — последняя известная мощность до момента выпуска прогноза
    power_curve  — эмпирическая кривая мощности по прогнозному ветру ens_ws100,
                   отдельно для previous_day1/2/3 (чем дальше лид, тем кривая «мягче»)

Валидация: январь 2026 (по местному времени), обучение — всё, что раньше.
Метрики в % от установленной мощности (мощность уже нормирована на 0..1):
    nMAE = mean|y - ŷ| * 100,  nRMSE = sqrt(mean (y - ŷ)^2) * 100.

Результаты:
    model/metrics_baseline.csv        таблица метрик по горизонтам 1–24 / 25–48 ч
    model/power_curve.json            кривая мощности (для predict)
    model/forecast_baseline_feb.csv   прогноз кривой мощности на февраль (запасной вариант)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from data import config as C

MODEL_DIR = Path(__file__).resolve().parent
CURVE_PATH = MODEL_DIR / "power_curve.json"
VALID_START_LOCAL = pd.Timestamp("2026-01-01")
WIND_COL = "ens_ws100"


# ------------------------------------------------------------------ метрики
def metrics(y: pd.Series, p: pd.Series) -> dict:
    m = y.notna() & p.notna()
    e = p[m] - y[m]
    return {"nMAE_%": 100 * e.abs().mean(), "nRMSE_%": 100 * np.sqrt((e ** 2).mean()),
            "bias_%": 100 * e.mean(), "n": int(m.sum())}


def score_table(df: pd.DataFrame, pred_cols: list[str], y: str = "power") -> pd.DataFrame:
    """Метрики по каждому прогнозу для горизонтов 1–24, 25–48 и всех вместе."""
    buckets = {"1-24h": df["lead_hour"] <= 24, "25-48h": df["lead_hour"] > 24, "all": df["lead_hour"] > 0}
    rows = [{"model": c, "horizon": b, **metrics(df.loc[m, y], df.loc[m, c])}
            for c in pred_cols for b, m in buckets.items()]
    return pd.DataFrame(rows).round(2)


# ------------------------------------------------------------ кривая мощности
def fit_power_curve(ws: pd.Series, p: pd.Series, step: float = 0.5, min_n: int = 15) -> dict:
    ok = ws.notna() & p.notna()
    bins = (ws[ok] / step).round() * step
    g = p[ok].groupby(bins).agg(["median", "size"])
    g = g[g["size"] >= min_n]
    y = np.maximum.accumulate(g["median"].to_numpy())  # кривая не убывает с ростом ветра
    return {"ws": g.index.tolist(), "p": y.tolist()}


def apply_curve(curve: dict, ws: pd.Series) -> np.ndarray:
    return np.clip(np.interp(ws.fillna(0), curve["ws"], curve["p"]), 0, 1)


def fit_curves(train: pd.DataFrame) -> dict:
    """Отдельная кривая для каждого day_offset + общая на случай нового лида."""
    target = train["power_clean"]
    curves = {"all": fit_power_curve(train[WIND_COL], target)}
    for n, part in train.groupby("day_offset"):
        curves[str(int(n))] = fit_power_curve(part[WIND_COL], part["power_clean"])
    return curves


def predict(df: pd.DataFrame, curves: dict | None = None) -> pd.DataFrame:
    """Интерфейс как у основной модели: добавляет колонку power_pred (0..1)."""
    if curves is None:
        curves = json.loads(CURVE_PATH.read_text(encoding="utf-8"))
    out = df.copy()
    out["power_pred"] = np.nan
    for n, part in out.groupby("day_offset"):
        curve = curves.get(str(int(n)), curves["all"])
        out.loc[part.index, "power_pred"] = apply_curve(curve, part[WIND_COL])
    return out


# ------------------------------------------------------------ другие бейзлайны
def climatology(train: pd.DataFrame, df: pd.DataFrame) -> pd.Series:
    table = train.groupby(["month", "hour_local"])["power_clean"].mean()
    key = pd.MultiIndex.from_arrays([df["month"], df["hour_local"]])
    return pd.Series(table.reindex(key).to_numpy(), index=df.index)


def persistence(df: pd.DataFrame, scada: pd.DataFrame) -> pd.Series:
    """Мощность за последний полный час до выпуска прогноза, на весь горизонт."""
    last = scada["power"].reindex(pd.to_datetime(df["issue_time"]) - pd.Timedelta(hours=1))
    return pd.Series(last.to_numpy(), index=df.index)


# -------------------------------------------------------------------- main
def main() -> None:
    train_all = pd.read_parquet(C.PROCESSED_DIR / "train.parquet")
    scada = pd.read_parquet(C.PROCESSED_DIR / "scada_hourly.parquet")
    local = pd.to_datetime(train_all["target_time_local"])
    issue_local = pd.to_datetime(train_all["issue_time"]) + (local - pd.to_datetime(train_all["target_time"]))
    # в обучение — только выпуски, у которых весь горизонт до начала валидации
    tr = train_all[issue_local + pd.Timedelta(hours=C.HORIZON_H) <= VALID_START_LOCAL]
    # валидация — только выпуски после окончания обучающих данных (как в model/train.py)
    va = train_all[(local >= VALID_START_LOCAL) & (issue_local >= VALID_START_LOCAL)].copy()
    print(f"train: {len(tr)} строк ({tr['target_time_local'].min():%Y-%m-%d} … {tr['target_time_local'].max():%Y-%m-%d}), "
          f"valid: {len(va)} строк (январь 2026)")

    curves = fit_curves(tr)
    va["power_curve"] = predict(va, curves)["power_pred"]
    va["climatology"] = climatology(tr, va)
    va["persistence"] = persistence(va, scada)

    table = score_table(va, ["persistence", "climatology", "power_curve"])
    print("\nМетрики на январе 2026 (y = power, % от установленной мощности):")
    print(table.pivot(index="model", columns="horizon", values="nMAE_%")
          [["1-24h", "25-48h", "all"]].add_prefix("nMAE ").to_string())
    table.to_csv(MODEL_DIR / "metrics_baseline.csv", index=False)

    # Финальные кривые — на всей истории; прогноз на февраль как запасной вариант
    # Итоговая кривая для февраля: только наблюдения, известные к первому выпуску
    # (час закончился не позже 31.01 00:00 UTC), — иначе первый выпуск видел бы будущее.
    first_issue = pd.Timestamp(C.TEST_FIRST_ISSUE) + pd.Timedelta(hours=C.ISSUE_HOUR_UTC)
    known = pd.to_datetime(train_all["target_time"]) + pd.Timedelta(hours=1) <= first_issue
    curves = fit_curves(train_all[known])
    CURVE_PATH.write_text(json.dumps(curves, indent=1), encoding="utf-8")
    test = pd.read_parquet(C.PROCESSED_DIR / "test_features.parquet")
    fc = predict(test, curves)[["issue_time", "target_time", "target_time_local", "lead_hour", "power_pred"]]
    fc.to_csv(MODEL_DIR / "forecast_baseline_feb.csv", index=False)
    print(f"\nСохранено: metrics_baseline.csv, power_curve.json, forecast_baseline_feb.csv ({len(fc)} строк)")


if __name__ == "__main__":
    main()
