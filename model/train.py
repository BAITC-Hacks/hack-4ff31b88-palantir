"""Обучение LightGBM и единый интерфейс прогноза."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from data import config as C
from data.features import feature_columns

MODEL_DIR = Path(__file__).resolve().parent
MODEL_PATH = MODEL_DIR / "lightgbm_model.txt"
FEATURES_PATH = MODEL_DIR / "lightgbm_features.json"
METRICS_PATH = MODEL_DIR / "metrics_lgbm.csv"
TARGET = "power_clean"
VALID_START = pd.Timestamp("2026-01-01")
VALID_END = pd.Timestamp("2026-02-01")


def _numeric_frame(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = df.reindex(columns=columns).copy()
    for col in columns:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out.replace([np.inf, -np.inf], np.nan)


def metric_rows(df: pd.DataFrame, pred_col: str = "power_pred",
                model_name: str = "lightgbm", target_col: str = TARGET) -> pd.DataFrame:
    buckets = {"1-24h": df["lead_hour"].between(1, 24),
               "25-48h": df["lead_hour"].between(25, 48),
               "all": df["lead_hour"].between(1, 48)}
    rows = []
    for horizon, mask in buckets.items():
        part = df.loc[mask]
        y, p = part[target_col].to_numpy(), part[pred_col].to_numpy()
        ok = np.isfinite(y) & np.isfinite(p)
        err = p[ok] - y[ok]
        rows.append({"model": model_name, "horizon": horizon,
                     "nMAE_%": 100 * np.abs(err).mean() if ok.any() else np.nan,
                     "nRMSE_%": 100 * np.sqrt(np.mean(err ** 2)) if ok.any() else np.nan,
                     "bias_%": 100 * err.mean() if ok.any() else np.nan,
                     "n": int(ok.sum())})
    return pd.DataFrame(rows).round(2)


def train_model(train: pd.DataFrame, features: list[str]):
    try:
        import lightgbm as lgb
    except ImportError as exc:
        raise RuntimeError("LightGBM не установлен. Выполните `pip install -r requirements.txt`.") from exc
    fit = train.loc[pd.to_datetime(train["target_time_local"]) < VALID_START]
    model = lgb.LGBMRegressor(objective="regression_l1", n_estimators=700,
                               learning_rate=0.03, num_leaves=31,
                               subsample=0.85, colsample_bytree=0.9,
                               reg_lambda=1.0, random_state=42, verbosity=-1)
    model.fit(_numeric_frame(fit, features), fit[TARGET].clip(0, 1))
    return model, len(fit)


def predict(df: pd.DataFrame, model=None) -> pd.DataFrame:
    """Добавляет power_pred, гарантированно ограниченный диапазоном [0, 1]."""
    if model is None:
        try:
            import lightgbm as lgb
        except ImportError as exc:
            raise RuntimeError("Для predict нужен пакет lightgbm") from exc
        features = json.loads(FEATURES_PATH.read_text(encoding="utf-8"))
        model = lgb.Booster(model_file=str(MODEL_PATH))
    else:
        features = list(getattr(model, "feature_name_", [])) or json.loads(
            FEATURES_PATH.read_text(encoding="utf-8"))
    out = df.copy()
    out["power_pred"] = np.clip(model.predict(_numeric_frame(out, features)), 0, 1)
    return out


def main() -> None:
    train = pd.read_parquet(C.PROCESSED_DIR / "train.parquet")
    test = pd.read_parquet(C.PROCESSED_DIR / "test_features.parquet")
    scada = pd.read_parquet(C.PROCESSED_DIR / "scada_hourly.parquet")
    features = [c for c in feature_columns(train) if c != "day_offset"]
    model, fit_rows = train_model(train, features)
    model.booster_.save_model(str(MODEL_PATH))
    FEATURES_PATH.write_text(json.dumps(features, ensure_ascii=False, indent=2), encoding="utf-8")
    valid = train.loc[pd.to_datetime(train["target_time_local"]).between(
        VALID_START, VALID_END, inclusive="left")].copy()
    metrics = metric_rows(predict(valid, model))
    metrics.to_csv(METRICS_PATH, index=False)
    feb = predict(test, model)
    feb[["issue_time", "target_time", "target_time_local", "lead_hour", "power_pred"]].to_csv(
        MODEL_DIR / "forecast_lgbm_feb.csv", index=False)
    # В test_features нет target по контракту датасета. Подтягиваем фактическую
    # мощность из почасовой SCADA и считаем сравнение с тремя бейзлайнами.
    from model.baseline_nwp import climatology, fit_curves, persistence, predict as curve_predict
    feb["power"] = scada["power"].reindex(pd.to_datetime(feb["target_time"])).to_numpy()
    curves = fit_curves(train.loc[pd.to_datetime(train["target_time_local"]) < VALID_START])
    feb["power_curve"] = curve_predict(feb, curves)["power_pred"].to_numpy()
    fit = train.loc[pd.to_datetime(train["target_time_local"]) < VALID_START]
    feb["climatology"] = climatology(fit, feb).to_numpy()
    feb["persistence"] = persistence(feb, scada).to_numpy()
    feb_metrics = pd.concat([
        metric_rows(feb, "power_pred", "lightgbm", "power"),
        metric_rows(feb, "power_curve", "power_curve", "power"),
        metric_rows(feb, "climatology", "climatology", "power"),
        metric_rows(feb, "persistence", "persistence", "power"),
    ], ignore_index=True)
    feb_metrics.to_csv(MODEL_DIR / "metrics_february.csv", index=False)
    print(f"Обучение: {fit_rows} строк, признаков: {len(features)}")
    print(metrics[["horizon", "nMAE_%", "n"]].to_string(index=False))
    print("Февраль 2026, сравнение с бейзлайнами (nMAE, %):")
    print(feb_metrics.pivot(index="model", columns="horizon", values="nMAE_%").to_string())


if __name__ == "__main__":
    main()

