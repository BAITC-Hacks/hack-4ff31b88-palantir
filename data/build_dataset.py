"""Собирает датасеты для модели.

Запуск из корня репозитория:
    python -m data.build_dataset            # сдвиг времени SCADA определяется сам
    python -m data.build_dataset --tz 5     # задать сдвиг вручную

Результат в data/processed/:
    train.parquet          история: признаки прогноза погоды + фактическая мощность
    test_features.parquet  февраль 2026: только признаки (для бэктеста)
    scada_hourly.parquet   почасовая SCADA станции (UTC), для бейзлайнов
    meta.json              сдвиг времени, параметры сборки
"""
from __future__ import annotations

import argparse
import json
import warnings

import pandas as pd

from . import config as C
from . import scada
from .features import add_features, feature_columns
from .weather_client import assemble, check_no_leakage, download_archive


def nwp_hourly(arch: pd.DataFrame, short: str, n: int = 1) -> pd.Series:
    """Прогноз previous_dayN, усреднённый на интервал часа [t, t+1)."""
    s = arch[f"{short}_d{n}"]
    return (s + s.shift(-1)) / 2


def detect_utc_offset(local: pd.DataFrame, arch: pd.DataFrame, candidates=range(3, 10)) -> tuple[int, pd.DataFrame]:
    """Подбирает сдвиг SCADA-времени к UTC по корреляции с прогнозом NWP.

    Решает корреляция ветра: скорость на гондоле и прогноз на 100 м — одна высота.
    Температура только для справки: датчик на гондоле (~100 м), там суточный ход
    отстаёт от прогноза на 2 м, и по температуре сдвиг выходит на час больше.
    """
    ws = nwp_hourly(arch, "ws100")
    t2 = nwp_hourly(arch, "t2m")
    t2_anom = t2 - t2.rolling(24, center=True, min_periods=12).mean()
    rows = []
    for k in candidates:
        s = local.copy()
        s.index = s.index - pd.Timedelta(hours=k)
        temp_anom = s["temp"] - s["temp"].rolling(24, center=True, min_periods=12).mean()
        rows.append({
            "utc_offset_h": k,
            "corr_temp_diurnal": temp_anom.corr(t2_anom.reindex(s.index)),
            "corr_wind": s["ws"].corr(ws.reindex(s.index)),
        })
    table = pd.DataFrame(rows).set_index("utc_offset_h")
    return int(table["corr_wind"].idxmax()), table


def daily_issues(first, last, hour: int = C.ISSUE_HOUR_UTC) -> pd.DatetimeIndex:
    return pd.date_range(pd.Timestamp(first) + pd.Timedelta(hours=hour),
                         pd.Timestamp(last) + pd.Timedelta(hours=hour), freq="D")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tz", default="auto", help="сдвиг SCADA к UTC в часах или 'auto'")
    ap.add_argument("--start", default=C.ARCHIVE_START, help="начало архива погоды (UTC)")
    args = ap.parse_args()
    warnings.filterwarnings("ignore", category=FutureWarning)
    C.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Погода: архив прогнозов по каждой модели
    end = pd.Timestamp(C.TEST_LAST_ISSUE) + pd.Timedelta(days=3)
    print(f"[1/5] Скачиваю архив прогнозов Open-Meteo {args.start} … {end:%Y-%m-%d}")
    archives = {}
    for prefix, model in C.MODELS.items():
        try:
            a = download_archive(model, args.start, end)
        except Exception as e:  # одна модель упала — работаем с остальными
            print(f"  ! {model} пропущена: {e}")
            continue
        coverage = a.notna().mean()
        good = coverage[coverage > 0.5].index
        print(f"  {model}: {len(a)} ч, колонок с данными {len(good)}/{a.shape[1]}")
        if len(good):
            archives[prefix] = a[good]
    if not archives:
        raise SystemExit("Нет погодных данных ни по одной модели")

    # 2. Сдвиг времени SCADA
    print("[2/5] Определяю часовой пояс «Статистического времени»")
    ref = next(iter(archives.values()))
    auto_k, table = detect_utc_offset(scada.local_hourly_for_tz_check(), ref)
    print(table.round(3).to_string())
    offset = auto_k if args.tz == "auto" else int(args.tz)
    print(f"  по данным: UTC+{auto_k}; используем UTC+{offset}")
    if offset != auto_k:
        print("  ! ручной сдвиг расходится с найденным по данным, проверь")

    # 3. SCADA по часам в UTC
    print("[3/5] Готовлю почасовую SCADA")
    station = scada.station_hourly(offset)
    station.to_parquet(C.PROCESSED_DIR / "scada_hourly.parquet")
    print(f"  {len(station)} ч, аномалий (простои/ограничения): {station['is_anomaly'].mean():.1%}")

    # 4. Train: выпуски раз в сутки на всей истории
    print("[4/5] Собираю train")
    last_obs = station.index.max()
    issues = daily_issues(args.start, (last_obs - pd.Timedelta(hours=1)).normalize())
    X = assemble(archives, issues)
    check_no_leakage(X)
    X = add_features(X, offset)
    train = X.merge(station.drop(columns="target_time_local"), left_on="target_time",
                    right_index=True, how="inner")
    weather_cols = [c for c in train.columns if c.startswith(tuple(archives))]
    train = train.dropna(subset=weather_cols, how="all")
    train = train[train["target_time"] <= last_obs]
    train.to_parquet(C.PROCESSED_DIR / "train.parquet", index=False)

    # 5. Test: февраль 2026, только признаки
    print("[5/5] Собираю test_features (февраль 2026)")
    T = assemble(archives, daily_issues(C.TEST_FIRST_ISSUE, C.TEST_LAST_ISSUE))
    margins = check_no_leakage(T)
    print("  запас между выходом последнего использованного прогона и выпуском прогноза: "
          + ", ".join(f"{m.upper()} ≥ {h:g} ч" for m, h in margins.items()))
    T = add_features(T, offset)
    T.to_parquet(C.PROCESSED_DIR / "test_features.parquet", index=False)

    meta = {
        "utc_offset_h": offset,
        "utc_offset_detected_h": auto_k,
        "tz_check": table.round(4).reset_index().to_dict("records"),
        "models": {p: C.MODELS[p] for p in archives},
        "issue_hour_utc": C.ISSUE_HOUR_UTC,
        "publish_delay_h": C.PUBLISH_DELAY_H,
        "leakage_margin_h": margins,
        "horizon_h": C.HORIZON_H,
        "train_rows": len(train),
        "train_period_utc": [str(train["target_time"].min()), str(train["target_time"].max())],
        "test_rows": len(T),
        "features": feature_columns(train),
        "built_at": pd.Timestamp.now(tz="UTC").isoformat(),
    }
    C.META_PATH.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    # Сводка
    print("\nГотово.")
    print(f"  train.parquet: {len(train)} строк, {train['issue_time'].nunique()} выпусков, "
          f"{meta['train_period_utc'][0]} … {meta['train_period_utc'][1]}")
    print(f"  test_features.parquet: {len(T)} строк, выпуски {T['issue_time'].min()} … {T['issue_time'].max()}")
    ok = train[~train["is_anomaly"]]
    for n in sorted(ok["day_offset"].unique()):
        s = ok[ok["day_offset"] == n]
        print(f"  corr(ens_ws100, power) для previous_day{n}: {s['ens_ws100'].corr(s['power']):.3f}")
    nan = train[meta["features"]].isna().mean()
    if (nan > 0.05).any():
        print("  признаки с >5% пропусков:", nan[nan > 0.05].round(3).to_dict())
    print(f"  признаков: {len(meta['features'])}; целевая колонка: power (или power_clean без простоев)")


if __name__ == "__main__":
    main()
