"""Дашборд прогноза выработки ВЭС.

Запуск из корня репозитория:
    streamlit run app/dashboard.py

Показывает то, что уже посчитали модули data/ и model/ (и agent/, если он
сохранил журнал), а на вкладке «Запуск» прогоняет весь цикл агента вживую:
погода Open-Meteo → признаки → модель → проверки.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import altair as alt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from data import config as C  # noqa: E402

MODEL_DIR = ROOT / "model"
FORECAST_FILES = {  # имя на графике -> файл с прогнозом на февраль
    "LightGBM": MODEL_DIR / "forecast_lgbm_feb.csv",
    "Кривая мощности": MODEL_DIR / "forecast_baseline_feb.csv",
}
AGENT_RUN_ROOTS = [ROOT / "artifacts", ROOT / "outputs"]  # куда agent_backtest.py пишет --out
SPREAD_ALERT_MS = 3.0     # расхождение ECMWF и GFS, при котором прогноз считаем неуверенным
JUMP_ALERT = 0.4          # скачок мощности за час (доля от установленной)

st.set_page_config(page_title="Прогноз ВЭС", page_icon="🌬️", layout="wide")


# --------------------------------------------------------------- оформление
def theme() -> str:
    t = getattr(getattr(st, "context", None), "theme", None)
    return getattr(t, "type", None) or "light"


def palette() -> dict:
    """Категориальная палитра (проверена на различимость при дальтонизме)."""
    if theme() == "dark":
        return {"LightGBM": "#3987e5", "Кривая мощности": "#d95926", "Живой запуск": "#3987e5", "Итоговый прогноз агента": "#3987e5",
                "Сохранённый выпуск 00:00 UTC": "#d95926", "ECMWF": "#199e70", "GFS": "#9085e9", "Факт": "#c3c2b7"}
    return {"LightGBM": "#2a78d6", "Кривая мощности": "#eb6834", "Живой запуск": "#2a78d6", "Итоговый прогноз агента": "#2a78d6",
            "Сохранённый выпуск 00:00 UTC": "#eb6834", "ECMWF": "#1baf7a", "GFS": "#4a3aa7", "Факт": "#52514e"}


def line_chart(long: pd.DataFrame, y_title: str, value_fmt: str, height: int = 300,
               y_domain: list | None = None) -> alt.Chart:
    """Линии series(time) с перекрестием и подсказкой. long: time, series, value."""
    colors = palette()
    series = [s for s in colors if s in set(long["series"])]
    color = alt.Color("series:N", title=None, legend=alt.Legend(orient="top", symbolType="stroke", symbolStrokeWidth=3),
                      scale=alt.Scale(domain=series, range=[colors[s] for s in series]))
    dash = alt.StrokeDash("series:N", legend=None, scale=alt.Scale(
        domain=series, range=[[4, 3] if s == "Факт" else [1, 0] for s in series]))
    x = alt.X("time:T", title=None, axis=alt.Axis(format="%d.%m %H:%M", labelAngle=0, tickCount=8))
    y = alt.Y("value:Q", title=y_title, scale=alt.Scale(domain=y_domain) if y_domain else alt.Undefined)
    base = alt.Chart(long).encode(x=x)
    hover = alt.selection_point(nearest=True, on="pointerover", fields=["time"], empty=False, clear="pointerout")
    lines = base.mark_line(strokeWidth=2).encode(y=y, color=color, strokeDash=dash)
    points = base.mark_point(size=70, filled=True).encode(
        y=y, color=color,
        opacity=alt.condition(hover, alt.value(1), alt.value(0)),
        tooltip=[alt.Tooltip("time:T", title="Время", format="%d.%m %H:%M"),
                 alt.Tooltip("series:N", title="Ряд"),
                 alt.Tooltip("value:Q", title=y_title, format=value_fmt)])
    rule = base.mark_rule(color="#8f8e89").encode(
        opacity=alt.condition(hover, alt.value(0.5), alt.value(0))).add_params(hover)
    return (lines + points + rule).properties(height=height)


def show_chart(chart) -> None:
    try:
        st.altair_chart(chart, width="stretch")
    except TypeError:  # старые версии Streamlit
        st.altair_chart(chart, use_container_width=True)


# ------------------------------------------------------------------- данные
def mtime(path: Path) -> float:
    """Время изменения файла — часть ключа кэша: новый файл сразу перечитывается.
    (Имена параметров кэшируемых функций без «_»: такие Streamlit исключает из ключа.)"""
    return path.stat().st_mtime if path.exists() else 0.0


@st.cache_data
def _load_parquet(name: str, file_mtime: float) -> pd.DataFrame | None:
    p = C.PROCESSED_DIR / name
    return pd.read_parquet(p) if p.exists() else None


def load_parquet(name: str) -> pd.DataFrame | None:
    return _load_parquet(name, mtime(C.PROCESSED_DIR / name))


@st.cache_data
def _load_forecasts(file_mtimes: tuple) -> dict[str, pd.DataFrame]:
    out = {}
    for name, path in FORECAST_FILES.items():
        if path.exists():
            df = pd.read_csv(path, parse_dates=["issue_time", "target_time", "target_time_local"])
            out[name] = df
    return out


def load_forecasts() -> dict[str, pd.DataFrame]:
    return _load_forecasts(tuple(mtime(p) for p in FORECAST_FILES.values()))


def load_metrics() -> pd.DataFrame | None:
    parts = []
    for f in ["metrics_baseline.csv", "metrics_lgbm.csv"]:
        p = MODEL_DIR / f
        if p.exists():
            parts.append(pd.read_csv(p))
    return pd.concat(parts, ignore_index=True) if parts else None


def load_meta() -> dict:
    return json.loads(C.META_PATH.read_text(encoding="utf-8")) if C.META_PATH.exists() else {}


def pct(x: float) -> str:
    return "—" if pd.isna(x) else f"{100 * x:.0f}%"


# ------------------------------------------------------------------ проверки
def run_checks(fc: pd.DataFrame, prev: pd.DataFrame | None) -> list[tuple[str, str, str]]:
    """Те же проверки, что делает агент на шаге «анализ результата».

    fc: target_time, power_pred, ens_ws100_spread (если есть). prev — прошлый выпуск.
    Возвращает (статус, проверка, пояснение); статус: ok / warn.
    """
    res = []
    p = fc["power_pred"] if "power_pred" in fc else pd.Series(dtype=float)
    times = pd.to_datetime(fc["target_time"]) if "target_time" in fc else pd.Series(dtype="datetime64[ns]")
    if len(fc) == 0 or p.notna().sum() == 0:
        res.append(("warn", f"Полный горизонт {C.HORIZON_H} ч", "прогноз пустой"))
        return res  # остальные проверки не имеют смысла
    # Ожидаемая сетка: момент выпуска + 0…47 ч, каждый час ровно один раз
    start = pd.Timestamp(fc["issue_time"].iloc[0]) if "issue_time" in fc else times.min()
    expected = set(pd.date_range(start, periods=C.HORIZON_H, freq="h"))
    got = list(times[p.notna().to_numpy()])
    dups = len(got) - len(set(got))
    missing = len(expected - set(got))
    extra = len(set(got) - expected)
    ok_grid = dups == 0 and missing == 0 and extra == 0
    note = "все 48 часов на месте, без дублей" if ok_grid else ", ".join(
        x for x in [f"нет {missing} ч" if missing else "", f"дублей: {dups}" if dups else "",
                    f"лишних часов: {extra}" if extra else ""] if x) + " — прогноз неполный"
    res.append(("ok" if ok_grid else "warn", f"Полный горизонт {C.HORIZON_H} ч", note))
    bad = int(((p < 0) | (p > 1) | p.isna()).sum())
    res.append(("ok" if bad == 0 else "warn", "Значения в диапазоне 0–100%",
                "все значения в диапазоне" if bad == 0 else f"{bad} ч вне диапазона или пустые"))
    jumps = int((p.diff().abs() > JUMP_ALERT).sum())
    res.append(("ok" if jumps == 0 else "warn", f"Нет скачков больше {JUMP_ALERT:.0%} за час",
                "ход прогноза плавный" if jumps == 0 else f"резких скачков: {jumps} — проверить погоду"))
    if "ens_ws100_spread" in fc:
        n = int((fc["ens_ws100_spread"] > SPREAD_ALERT_MS).sum())
        res.append(("ok" if n <= 6 else "warn", f"ECMWF и GFS согласны (разброс < {SPREAD_ALERT_MS:g} м/с)",
                    f"{n} ч с большим разбросом" + ("" if n <= 6 else " — прогноз неуверенный, стоит пересчитать на свежем прогоне")))
    if prev is not None:
        m = fc.merge(prev[["target_time", "power_pred"]], on="target_time", suffixes=("", "_prev"))
        if len(m):
            d = (m["power_pred"] - m["power_pred_prev"]).abs().mean()
            res.append(("ok" if d < 0.15 else "warn", "Согласован с прошлым выпуском",
                        f"среднее изменение на общих {len(m)} ч: {d:.0%}"))
    return res


def show_checks(checks) -> None:
    for status, name, note in checks:
        icon = "✅" if status == "ok" else "⚠️"
        st.markdown(f"{icon} **{name}** — {note}")


# ---------------------------------------------------------------- интерфейс
meta = load_meta()
offset = int(meta.get("utc_offset_h", C.utc_offset_h()))
test = load_parquet("test_features.parquet")
scada = load_parquet("scada_hourly.parquet")
forecasts = load_forecasts()

st.title("Прогноз выработки ветроэлектростанции")
st.caption(f"Почасовой прогноз на 24–48 ч · погода: Open-Meteo, архив прогнозов ECMWF и GFS · "
           f"время на графиках местное (UTC+{offset}, как в данных SCADA)")

if test is None or not forecasts:
    st.error("Нет готовых данных. Сначала выполните `python -m data.build_dataset` и "
             "`python -m model.baseline_nwp` (или `python -m model.train`).")
    st.stop()

issues = sorted(set().union(*[set(df["issue_time"]) for df in forecasts.values()]))
with st.sidebar:
    st.header("Выпуск прогноза")
    labels = {t: f"{(t + pd.Timedelta(hours=offset)):%d.%m.%Y %H:%M} (UTC {t:%H:%M})" for t in issues}
    issue = st.selectbox("Момент выпуска", issues, index=min(1, len(issues) - 1), format_func=labels.get)
    shown = st.multiselect("Модели", list(forecasts), default=list(forecasts))
    st.divider()
    st.markdown(f"**Станция:** 2 турбины, {C.SITE_LAT:.4f}, {C.SITE_LON:.4f}")
    st.markdown(f"**Погода:** {', '.join(meta.get('models', {}).values()) or 'ECMWF, GFS'}")
    st.markdown("**Без утечки:** каждый лид берётся из прогона, опубликованного до момента выпуска")

tab_fc, tab_check, tab_quality, tab_run = st.tabs(
    ["📈 Прогноз", "🔎 Проверки агента", "🎯 Качество модели", "▶️ Запуск агента"])

# ---- 1. прогноз на 48 ч
with tab_fc:
    wx = test[test["issue_time"] == issue].copy()
    rows = []
    for name in shown:
        f = forecasts[name]
        f = f[f["issue_time"] == issue]
        rows.append(pd.DataFrame({"time": f["target_time_local"], "series": name, "value": 100 * f["power_pred"]}))
    fact = None
    if scada is not None and len(wx):
        fact = scada["power"].reindex(pd.to_datetime(wx["target_time"])).to_numpy()
        if np.isfinite(fact).any():
            rows.append(pd.DataFrame({"time": wx["target_time_local"].to_numpy(), "series": "Факт", "value": 100 * fact}))
    power_long = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["time", "series", "value"])

    main_name = shown[0] if shown else next(iter(forecasts))
    main = forecasts[main_name]
    main = main[main["issue_time"] == issue].sort_values("lead_hour")
    d1, d2 = main[main["lead_hour"] <= 24], main[main["lead_hour"] > 24]
    peak = main.loc[main["power_pred"].idxmax()] if len(main) else None
    spread = wx["ens_ws100_spread"] if "ens_ws100_spread" in wx else pd.Series(0.0, index=wx.index)
    spread_hours = int((spread > SPREAD_ALERT_MS).sum()) if len(wx) else 0

    c = st.columns(4)
    c[0].metric("Средняя загрузка, 1–24 ч", pct(d1["power_pred"].mean()))
    c[1].metric("Средняя загрузка, 25–48 ч", pct(d2["power_pred"].mean()))
    c[2].metric(f"Пик, {peak['target_time_local']:%d.%m %H:%M}" if peak is not None else "Пик",
                pct(peak["power_pred"]) if peak is not None else "—")
    c[3].metric("Часов с неуверенной погодой", f"{spread_hours} из 48")
    st.caption(f"Карточки посчитаны по модели «{main_name}». Загрузка — средняя мощность в % от установленной.")

    st.subheader("Прогноз мощности, % от установленной")
    show_chart(line_chart(power_long, "Мощность, %", ".0f", y_domain=[0, 100]))

    st.subheader("Прогноз ветра на 100 м: ECMWF и GFS")
    wind_parts = [pd.DataFrame({"time": wx["target_time_local"], "series": name, "value": wx[col]})
                  for name, col in [("ECMWF", "ecmwf_ws100"), ("GFS", "gfs_ws100")] if col in wx]
    wind_long = pd.concat(wind_parts) if wind_parts else pd.DataFrame(columns=["time", "series", "value"])
    show_chart(line_chart(wind_long.dropna(), "Ветер, м/с", ".1f", height=220))
    st.caption("Когда модели сильно расходятся, прогноз менее надёжен — агент отмечает такие часы.")

    with st.expander("Таблица прогноза"):
        wx_cols = ["target_time_local"] + [c for c in ["ecmwf_ws100", "gfs_ws100", "ens_ws100_spread"] if c in wx]
        table = main[["target_time_local", "lead_hour", "power_pred"]].merge(
            wx[wx_cols], on="target_time_local", how="left")
        table = table.rename(columns={"target_time_local": "Время (местное)", "lead_hour": "Лид, ч",
                                      "power_pred": "Мощность, доля", "ecmwf_ws100": "ECMWF, м/с",
                                      "gfs_ws100": "GFS, м/с", "ens_ws100_spread": "Разброс, м/с"})
        st.dataframe(table.round({c: 3 for c in table.columns if c != "Время (местное)"}), hide_index=True)
        st.download_button("Скачать CSV", table.to_csv(index=False).encode("utf-8"),
                           file_name=f"forecast_{issue:%Y%m%d_%H}utc.csv", mime="text/csv")

# ---- 2. проверки
with tab_check:
    st.subheader(f"Анализ выпуска {labels[issue]}")
    fc = main.merge(wx[["target_time"] + (["ens_ws100_spread"] if "ens_ws100_spread" in wx else [])],
                    on="target_time", how="left")
    prev_issue = [t for t in issues if t < issue]
    prev = forecasts[main_name]
    prev = prev[prev["issue_time"] == prev_issue[-1]] if prev_issue else None
    show_checks(run_checks(fc, prev))

    st.divider()
    st.subheader("Журнал агента (agent_backtest.py)")
    runs_dirs = sorted({p.parent for r in AGENT_RUN_ROOTS if r.exists() for p in r.rglob("summary.json")},
                       key=lambda d: d.stat().st_mtime, reverse=True)
    if not runs_dirs:
        st.info("Журнал появится после бэктеста: `python agent_backtest.py --mode lightgbm "
                "--trained-through 2025-12-31T18:00:00+00:00 --out artifacts/agent_lgbm`.")
    else:
        run_dir = st.selectbox("Прогон агента", runs_dirs, format_func=lambda d: str(d.relative_to(ROOT)))
        summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
        k = st.columns(4)
        k[0].metric("Режим", summary.get("mode", "—"))
        k[1].metric("Принято выпусков", f"{summary.get('accepted_runs')} из {summary.get('runs')}")
        k[2].metric("Часов февраля", f"{summary.get('february_hours')} из {summary.get('expected_february_hours')}")
        k[3].metric("Пересчитано точек", summary.get("replaced_points", "—"))

        runs_path = run_dir / "runs.jsonl"
        if runs_path.exists():
            runs = [json.loads(line) for line in runs_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            table = pd.DataFrame([{
                "Выпуск (UTC)": r["issue_time"][:16].replace("T", " "),
                "Решение": "✅ принят" if r["analysis"]["accepted"] else "⛔ отклонён",
                "Ошибок": len(r["analysis"]["errors"]),
                "Предупреждений": len(r["analysis"]["warnings"]),
                "Пересчитано ч": r.get("replaced_points", 0),
                "Ср. изменение": r["analysis"].get("mean_abs_revision"),
                "Резюме": (r["analysis"].get("summary") or {}).get("text", ""),
            } for r in runs])
            st.dataframe(table.round({"Ср. изменение": 3}), hide_index=True)
            this = next((r for r in runs if pd.Timestamp(r["issue_time"]).tz_convert(None) == issue), None)
            if this is not None:
                with st.expander(f"Подробно: выпуск {labels[issue]}"):
                    s_ = this["analysis"].get("summary") or {}
                    st.markdown(f"**Резюме ({'LLM' if s_.get('source') == 'llm' else 'правила'}):** {s_.get('text', '—')}")
                    st.json({"errors": this["analysis"]["errors"], "warnings": this["analysis"]["warnings"]}, expanded=False)

        latest = run_dir / "february_latest.csv"
        if latest.exists():
            fl = pd.read_csv(latest)
            fl["time"] = pd.to_datetime(fl["target_time_local"].str.slice(0, 19))
            st.subheader("Итоговый прогноз на февраль: для каждого часа — самый свежий принятый выпуск")
            show_chart(line_chart(pd.DataFrame({"time": fl["time"], "series": "Итоговый прогноз агента",
                                                "value": 100 * fl["power_pred"]}),
                                  "Мощность, %", ".0f", height=260, y_domain=[0, 100]))
            st.download_button("Скачать итоговый прогноз (CSV)", latest.read_bytes(),
                               file_name="february_forecast.csv", mime="text/csv")


# ---- 3. качество
with tab_quality:
    metrics = load_metrics()
    st.subheader("Ошибка на январе 2026 (валидация), nMAE в % от установленной мощности")
    if metrics is not None:
        pivot = metrics.pivot_table(index="model", columns="horizon", values="nMAE_%").reindex(
            columns=["1-24h", "25-48h", "all"]).sort_values("all")
        pivot.columns = ["1–24 ч", "25–48 ч", "Все горизонты"]
        st.dataframe(pivot.round(2))
        st.caption("persistence — «как в последний час», climatology — среднее по часу и месяцу, "
                   "power_curve — кривая мощности по прогнозу ветра. Меньше — лучше.")
    train = load_parquet("train.parquet")
    if train is not None:
        try:
            from model.baseline_nwp import fit_curves, predict as curve_predict
            local = pd.to_datetime(train["target_time_local"])
            jan = train[(local >= "2026-01-01") & (train["lead_hour"] <= 24)].copy()
            hist = train[local < "2026-01-01"]
            jan["pred"] = curve_predict(jan, fit_curves(hist))["power_pred"]
            st.subheader("Январь 2026: факт и прогноз кривой мощности (лид 1–24 ч)")
            jl = pd.concat([
                pd.DataFrame({"time": jan["target_time_local"], "series": "Факт", "value": 100 * jan["power"]}),
                pd.DataFrame({"time": jan["target_time_local"], "series": "Кривая мощности", "value": 100 * jan["pred"]}),
            ])
            show_chart(line_chart(jl, "Мощность, %", ".0f", height=280, y_domain=[0, 100]))
        except Exception as e:  # noqa: BLE001
            st.warning(f"Не удалось построить график января: {e}")

# ---- 4. живой запуск
with tab_run:
    st.subheader("Полный цикл агента вживую")
    st.markdown("Погода скачивается из Open-Meteo **в том виде, в каком была известна** на момент выпуска, "
                "дальше — признаки, модель и проверки. Выпуск в 12:00 UTC вместо 00:00 показывает "
                "**пересчёт при обновлении прогноза погоды**.")
    col = st.columns(3)
    day = col[0].date_input("Дата выпуска", value=pd.Timestamp("2026-02-10").date(),
                            min_value=pd.Timestamp("2026-01-31").date(), max_value=pd.Timestamp("2026-02-28").date())
    hour = col[1].selectbox("Час выпуска, UTC", [0, 6, 12, 18], format_func=lambda h: f"{h:02d}:00")
    model_name = col[2].selectbox("Модель", ["LightGBM", "Кривая мощности"],
                                  index=0 if (MODEL_DIR / "lightgbm_model.txt").exists() else 1)
    if st.button("Запустить агента", type="primary"):
        t = pd.Timestamp(day) + pd.Timedelta(hours=hour)
        with st.status("Агент работает…", expanded=True) as status:
            try:
                from data.features import add_features
                from data.weather_client import fetch_forecast
                st.write("1. Получаю прогноз погоды ECMWF и GFS на момент выпуска…")
                wx_live = fetch_forecast(t)
                st.write(f"   получено {len(wx_live)} ч, прогоны не новее T − {C.PUBLISH_DELAY_H} ч")
                st.write("2. Готовлю признаки…")
                X = add_features(wx_live, offset)
                st.write(f"3. Запускаю модель: {model_name}…")
                if model_name == "LightGBM":
                    from model.train import predict as model_predict
                else:
                    from model.baseline_nwp import predict as model_predict
                out = model_predict(X)
                st.write("4. Проверяю результат…")
                base = forecasts.get(model_name)
                prev = base[base["issue_time"] == pd.Timestamp(day)] if base is not None and hour else None
                checks = run_checks(out, prev if prev is not None and len(prev) else None)
                status.update(label="Готово", state="complete")
            except Exception as e:  # noqa: BLE001
                status.update(label="Ошибка", state="error")
                st.exception(e)
                st.stop()
        show_checks(checks)
        live = [pd.DataFrame({"time": out["target_time_local"], "series": "Живой запуск", "value": 100 * out["power_pred"]})]
        if prev is not None and len(prev):
            live.append(pd.DataFrame({"time": prev["target_time_local"], "series": "Сохранённый выпуск 00:00 UTC",
                                      "value": 100 * prev["power_pred"]}))
        show_chart(line_chart(pd.concat(live), "Мощность, %", ".0f", y_domain=[0, 100]))
        if prev is not None and len(prev):
            st.caption(f"Оранжевая линия — сохранённый выпуск того же дня в 00:00 UTC той же модели: "
                       f"видно, как прогноз изменился после обновления погоды к {hour:02d}:00 UTC.")
