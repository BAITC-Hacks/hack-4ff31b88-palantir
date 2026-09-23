"""Отдельная страница агента: streamlit run app/repository_agent.py."""
from __future__ import annotations

import math
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# The entry page and API module share a filename; the project root must win.
if str(ROOT) in sys.path:
    sys.path.remove(str(ROOT))
sys.path.insert(0, str(ROOT))

import altair as alt
import pandas as pd
import streamlit as st
from streamlit.components.v1 import html as component_html

from repository_agent import AgentError, DEFAULT_MODEL, ask
from repository_data import RepositoryData, RepositoryDataError
from app.wind_scene import wind_scene_html


st.set_page_config(page_title="Palantir · Агент данных", page_icon="◈", layout="wide")
st.html("<style>" + (ROOT / "app" / "agent_theme.css").read_text(encoding="utf-8") + "</style>")


def value(number, *, percent: bool = False, digits: int = 1) -> str:
    """Format nullable values without turning missing data into zero."""
    try:
        number = float(number)
    except (TypeError, ValueError):
        return "—"
    if not math.isfinite(number):
        return "—"
    if percent:
        return f"{number * 100:.{digits}f}%".replace(".", ",")
    return f"{number:,.{digits}f}".replace(",", " ").replace(".", ",")


def sources(paths: list[str], *, label: str = "Файлы-источники") -> None:
    if paths:
        with st.expander(label):
            st.code("\n".join(dict.fromkeys(str(path) for path in paths)), language=None)


def local_times(frame: pd.DataFrame) -> pd.Series:
    """Keep the saved SCADA wall-clock time regardless of browser timezone."""
    return pd.to_datetime(frame["target_time_local"].astype(str).str[:19], errors="coerce")


def forecast_chart(frame: pd.DataFrame, timezone_label: str) -> alt.Chart:
    chart_data = frame.assign(
        local_time=local_times(frame),
        power_percent=pd.to_numeric(frame["power_pred"], errors="coerce") * 100,
    )
    return (
        alt.Chart(chart_data)
        .mark_area(
            line={"color": "#a2efd2", "strokeWidth": 2.2},
            color=alt.Gradient(
                gradient="linear", x1=0, y1=0, x2=0, y2=1,
                stops=[alt.GradientStop(color="#55caa4", offset=0),
                       alt.GradientStop(color="transparent", offset=1)],
            ),
            opacity=0.65,
        )
        .encode(
            x=alt.X("local_time:T", title=f"Местное время SCADA ({timezone_label})",
                    axis=alt.Axis(format="%d.%m %H:%M", labelAngle=0, tickCount=5)),
            y=alt.Y("power_percent:Q", title="Мощность, % от установленной",
                    scale=alt.Scale(domain=[0, 100])),
            tooltip=[alt.Tooltip("local_time:T", title=f"Время {timezone_label}", format="%d.%m.%Y %H:%M"),
                     alt.Tooltip("power_percent:Q", title="Мощность, %", format=".2f"),
                     alt.Tooltip("issue_time:N", title="Выпуск, UTC")],
        )
        .properties(height=290, background="transparent")
        .configure_view(strokeOpacity=0)
        .configure_axis(labelColor="#a7c0c7", titleColor="#b7d0d2", gridColor="#28434a",
                        gridOpacity=0.55, domainColor="#28434a", tickColor="#28434a", titleFontWeight=400)
    )


def render_chat(run_id: str, model: str, api_ready: bool) -> None:
    st.subheader("Спросить агента")
    st.caption("Ответы OpenAI по файлам выбранного прогона, с указанием источников.")
    if not api_ready:
        st.info("Данные доступны. Для диалога задайте OPENAI_API_KEY в терминале, "
                "из которого запущена эта страница, и перезапустите Streamlit.")
        with st.expander("Как подключить API в PowerShell"):
            st.code(
                "$env:OPENAI_API_KEY = [System.Net.NetworkCredential]::new('', "
                "(Read-Host 'API-ключ OpenAI' -AsSecureString)).Password\n"
                "py -m streamlit run app/repository_agent.py",
                language="powershell",
            )
            st.caption("Вставляйте ключ только в защищённый запрос терминала.")

    histories = st.session_state.setdefault("repository_agent_histories", {})
    errors = st.session_state.setdefault("repository_agent_errors", {})
    history_key = f"{run_id}\n{model}"
    messages = histories.setdefault(history_key, [])
    pending = None
    if not messages:
        for index, suggestion in enumerate([
            "Кратко подведи итоги этого бэктеста",
            "Какие проверки требуют внимания?",
            "Когда ожидается максимальная мощность?",
        ]):
            if st.button(suggestion, key=f"suggestion_{index}", disabled=not api_ready,
                         use_container_width=True):
                pending = suggestion

    with st.container(border=False, **({"height": 320} if messages else {})):
        if not messages:
            st.markdown('<div class="chat-empty"><div class="chat-empty-symbol">✦</div>'
                        '<strong>От данных — к пониманию</strong>'
                        '<p>Задайте вопрос о выработке, сравните модели '
                        'или разберите предупреждения агента.</p></div>', unsafe_allow_html=True)
        for message in messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if message.get("sources"):
                    sources(message["sources"], label="Источники ответа")
                if message.get("tool_calls"):
                    with st.expander("Что прочитал агент"):
                        for call in message["tool_calls"]:
                            st.code(str(call.get("name", "Чтение данных")), language=None)
                            st.json(call.get("arguments", {}), expanded=False)
                            for source in call.get("sources", []):
                                st.text(str(source))
                            if call.get("error"):
                                st.warning(str(call["error"]))
                if message.get("model"):
                    st.caption(f"Модель: {message['model']}")

    prompt = st.chat_input("Задайте вопрос о прогнозе или проверках…", disabled=not api_ready)
    pending = prompt or pending
    st.caption("Агент читает весь выбранный прогон. Фильтр дат слева применяется к графику.")
    if messages and st.button("Очистить диалог", key="clear_agent_chat"):
        histories[history_key] = []
        errors.pop(history_key, None)
        st.rerun()

    if pending:
        conversation = [{"role": item["role"], "content": item["content"]}
                        for item in messages[-12:]]
        messages.append({"role": "user", "content": pending})
        errors.pop(history_key, None)
        try:
            with st.spinner("Агент читает файлы и готовит ответ…"):
                result = ask(pending, root=ROOT, run_id=run_id, model=model, history=conversation)
            messages.append({"role": "assistant", "content": result["text"],
                             "sources": result.get("sources", []),
                             "tool_calls": result.get("tool_calls", []),
                             "model": result.get("model", model)})
        except AgentError as exc:
            errors[history_key] = str(exc)
        except Exception:
            # Provider details can contain identifiers or secrets; keep them off the page.
            errors[history_key] = (
                "Не удалось получить ответ. Проверьте подключение, настройки API и повторите вопрос."
            )
        st.rerun()
    if errors.get(history_key):
        st.error(errors[history_key])


def render_checks(payload: dict) -> None:
    rows = payload.get("rows", [])
    st.subheader("Проверки сохранённых выпусков")
    st.caption("Принятие выпуска означает прохождение проверок данных. "
               "Точность прогноза определяется сравнением с фактической выработкой.")
    if not rows:
        st.info("Для этого прогона журнал проверок отсутствует или пуст.")
        return
    table = pd.DataFrame([{
        "Выпуск (UTC)": row.get("issue_time"),
        "Результат": "Принят" if row.get("accepted") else "Отклонён",
        "Ошибок": row.get("error_count"),
        "Предупреждений": row.get("warning_count"),
        "Общих часов": row.get("overlap_points"),
        "Изменено часов": row.get("changed_points"),
        "Среднее изменение, п.п.": (
            float(row["mean_abs_revision"]) * 100
            if row.get("mean_abs_revision") is not None else None
        ),
        "Резюме": "OpenAI" if row.get("summary", {}).get("source") == "llm" else "Правила",
    } for row in rows])
    st.dataframe(table.round(2), hide_index=True, use_container_width=True)
    if payload.get("truncated"):
        st.caption(f"Показаны {len(rows)} из {payload.get('total_rows')} выпусков.")
    choice = st.selectbox("Подробности выпуска", list(range(len(rows))),
                          format_func=lambda index: str(rows[index].get("issue_time", index)))
    selected = rows[choice]
    if selected.get("details_truncated"):
        st.caption("Длинные списки замечаний или резюме сокращены. Полный текст находится в журнале прогона.")
    left, right = st.columns(2)
    with left:
        st.markdown("**Ошибки проверки**")
        errors = selected.get("errors", [])
        if errors:
            for error in errors:
                st.error(str(error))
        else:
            st.success("Ошибок проверки нет")
    with right:
        st.markdown("**Предупреждения**")
        warnings = selected.get("warnings", [])
        if warnings:
            for warning in warnings:
                st.warning(str(warning))
        else:
            st.info("Предупреждений нет")
    saved_summary = selected.get("summary") or {}
    if saved_summary.get("text"):
        with st.expander("Сохранённое резюме выпуска"):
            st.caption("Текст из журнала; новый запрос к API не выполняется.")
            st.write(saved_summary["text"])
    sources(payload.get("sources", []))


def render_metrics(payload: dict) -> None:
    st.subheader("Качество моделей на январской валидации")
    st.info("Метрики января 2026 относятся к валидации модели. "
            "Они не измеряют точность февральского бэктеста.")
    rows = payload.get("rows", [])
    if not rows:
        st.info("Файлы с метриками моделей не найдены.")
        return
    frame = pd.DataFrame(rows)
    # A file may include an empty February row. n=0 is never a measured error.
    available = pd.to_numeric(frame.get("n", pd.Series(index=frame.index, dtype=float)), errors="coerce") > 0
    january = frame[available & frame.get("period", pd.Series("", index=frame.index)).eq("2026-01")]
    if not january.empty:
        columns = [name for name in ["model", "horizon", "nMAE_%", "nRMSE_%", "bias_%", "n"]
                   if name in january]
        st.dataframe(january[columns].rename(columns={
            "model": "Модель", "horizon": "Горизонт", "nMAE_%": "nMAE, %",
            "nRMSE_%": "nRMSE, %", "bias_%": "Смещение, %", "n": "Наблюдений",
        }).round(2), hide_index=True, use_container_width=True)
        st.caption("nMAE и nRMSE нормированы на установленную мощность; меньше — лучше. "
                   "Смещение показывает направление средней ошибки.")
    else:
        st.warning("В файлах нет рассчитанных январских метрик с n > 0.")
    other = frame[available & ~frame.get("period", pd.Series("", index=frame.index)).eq("2026-01")]
    if not other.empty:
        with st.expander("Метрики других периодов"):
            st.dataframe(other, hide_index=True, use_container_width=True)
    if (~available).any():
        st.caption("Строки без наблюдений (n = 0) не показаны как измеренная точность.")
    sources(payload.get("sources", []))


def main() -> None:
    try:
        repository = RepositoryData(ROOT)
        runs = repository.list_runs()
    except (RepositoryDataError, OSError, ValueError):
        st.error("Не удалось прочитать данные репозитория. Проверьте сохранённые файлы прогонов.")
        st.stop()

    with st.sidebar:
        st.markdown("### ◈ Palantir")
        st.caption("АГЕНТ ДАННЫХ ВЭС")
        st.divider()
        if not runs:
            st.info("Сохранённых прогонов пока нет.")
            st.stop()
        ids = [run["id"] for run in runs]
        default_index = next((index for index, name in enumerate(ids) if "lgbm_llm" in name), 0)
        run_id = st.selectbox("Прогон репозитория", ids, index=default_index)
        st.caption("Данные читаются из локальных файлов этого репозитория.")
        if st.button("Обновить данные", use_container_width=True):
            st.rerun()
        st.divider()
        model = st.text_input("Модель OpenAI", value=os.getenv("OPENAI_MODEL") or DEFAULT_MODEL).strip()
        model = model or DEFAULT_MODEL
        api_ready = bool(os.getenv("OPENAI_API_KEY", "").strip())
        if api_ready:
            st.success("Ключ API задан")
            st.caption("Доступ к API проверяется при отправке вопроса.")
        else:
            st.warning("Ключ API не задан")
        st.caption("Запрос к OpenAI выполняется только при отправке вопроса. "
                   "Обычный просмотр данных не расходует API-баланс.")
        st.divider()
        animation_enabled = st.toggle("Анимация ветропарка", value=True, key="wind_animation")

    # Keep animated SVG in its own document: st.html sanitizes inline SVG away.
    component_html(wind_scene_html(animate=animation_enabled), height=0, scrolling=False, tab_index=-1)
    with st.container(key="hero"):
        st.markdown('<div class="eyebrow">Энергия ветра · Аналитика</div>', unsafe_allow_html=True)
        st.title("Palantir · Агент данных")
        st.markdown('<p class="hero-lead">Понимайте каждый час выработки. '
                    'Прогнозы, проверки и ответы агента — в одном пространстве.</p>', unsafe_allow_html=True)
        st.markdown('<div class="hero-badges"><span><i class="dot"></i>Прогноз на 48 часов</span>'
                    '<span>ECMWF + GFS</span><span>Исторические данные</span></div>', unsafe_allow_html=True)
    st.caption(f"Выбранный прогон: {run_id}")

    try:
        overview = repository.run_summary(run_id)
        complete_forecast = repository.forecast(run_id, limit=2000)
        check_payload = repository.checks(run_id, limit=50)
        metric_payload = repository.metrics()
    except (RepositoryDataError, OSError, ValueError):
        st.error("Данные выбранного прогона не удалось прочитать. "
                 "Выберите другой прогон или проверьте его файлы.")
        st.stop()
    summary = overview.get("summary", {})
    check_totals = overview.get("checks", {})
    with st.container(key="overview_metrics"):
        cards = st.columns(4)
    cards[0].metric("Принято выпусков", f"{summary.get('accepted_runs', '—')} / {summary.get('runs', '—')}")
    cards[1].metric("Часов февраля", f"{summary.get('february_hours', '—')} / {summary.get('expected_february_hours', '—')}")
    cards[2].metric("Ошибок проверки", value(check_totals.get("errors"), digits=0))
    cards[3].metric("Резюме OpenAI", value(check_totals.get("llm"), digits=0))
    st.caption("Покрытие и проверки рассчитаны по сохранённому прогону. "
               "Мощность показана в долях установленной мощности станции.")
    issues = getattr(repository, "issues", [])
    if issues:
        with st.expander(f"Замечания к файлам: {len(issues)}"):
            for issue in issues:
                st.warning(str(issue))

    st.divider()
    with st.container(key="workspace_panels"):
        forecast_col, chat_col = st.columns([1.35, 1], gap="large")
    with forecast_col, st.container(border=True, key="forecast_panel"):
        st.subheader("Итоговый прогноз")
        timezone_label = complete_forecast.get("timezone", "UTC+6")
        st.caption(f"Для каждого часа — самый свежий принятый выпуск. Время SCADA: {timezone_label}.")
        forecast_frame = pd.DataFrame(complete_forecast.get("rows", []))
        if forecast_frame.empty:
            st.info("Итоговый прогноз для этого прогона отсутствует или пуст.")
        else:
            timestamps = local_times(forecast_frame).dropna()
            if timestamps.empty:
                st.warning("В прогнозе нет корректных временных отметок.")
            else:
                first, last = timestamps.min().date(), timestamps.max().date()
                selected_dates = st.date_input(
                    "Период на графике", value=(first, last), min_value=first, max_value=last,
                    key=f"forecast_dates_{run_id}", format="DD.MM.YYYY",
                )
                if isinstance(selected_dates, (list, tuple)) and len(selected_dates) == 2:
                    start_date, end_date = selected_dates
                    payload = repository.forecast(run_id, start=start_date.isoformat(),
                                                  end=end_date.isoformat(), limit=2000)
                else:
                    payload = complete_forecast
                    st.caption("Выберите обе даты для ограничения периода.")
                shown = pd.DataFrame(payload.get("rows", []))
                if shown.empty:
                    st.info("На выбранные даты прогноз отсутствует.")
                else:
                    statistics = payload.get("statistics", {})
                    mini = st.columns(3)
                    mini[0].metric("Средняя", value(statistics.get("mean_power"), percent=True))
                    mini[1].metric("Пик", value(statistics.get("max_power"), percent=True))
                    mini[2].metric("Часов", str(payload.get("total_rows", len(shown))))
                    st.altair_chart(forecast_chart(shown, timezone_label), use_container_width=True, theme=None)
                    with st.expander("Почасовая таблица и CSV"):
                        table = shown.copy()
                        table["power_percent"] = pd.to_numeric(table["power_pred"], errors="coerce") * 100
                        table = table[["target_time_local", "power_percent", "issue_time"]].rename(columns={
                            "target_time_local": f"Время {timezone_label}", "power_percent": "Мощность, %",
                            "issue_time": "Выпуск, UTC",
                        })
                        st.dataframe(table.round(3), hide_index=True, use_container_width=True)
                        st.download_button("Скачать выбранный прогноз CSV",
                                           shown.to_csv(index=False).encode("utf-8-sig"),
                                           file_name="palantir_forecast.csv", mime="text/csv")
                    if payload.get("truncated"):
                        st.warning(f"Показаны первые {len(shown)} из {payload.get('total_rows')} строк. "
                                   "Сузьте период, чтобы увидеть и скачать все его данные.")
                    sources(payload.get("sources", []))
    with chat_col:
        with st.container(border=True, key="chat_panel"):
            render_chat(run_id, model, api_ready)

    st.divider()
    checks_tab, quality_tab, run_tab = st.tabs(["Проверки выпусков", "Качество моделей", "О прогоне"])
    with checks_tab:
        render_checks(check_payload)
    with quality_tab:
        render_metrics(metric_payload)
    with run_tab:
        st.subheader("Результаты бэктеста")
        st.caption("Сохранённые результаты выбранного прогона. Время выпуска — UTC.")
        st.json(summary, expanded=True)
        sources(overview.get("sources", []))
    st.markdown('<div class="page-footer"><span>Palantir / Wind intelligence</span>'
                '<span>Прогноз · Анализ · Решение</span></div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
