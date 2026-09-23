"""HTTP API агента прогноза выработки ВЭС.

Запуск из корня репозитория:
    python -m uvicorn api.main:app --port 8000
    -> Swagger (все эндпоинты можно вызвать из браузера): http://localhost:8000/docs

Агент тот же, что в agent_backtest.py: погода Open-Meteo на момент выпуска ->
признаки -> модель -> проверки (agent_analysis.analyze) -> резюме (LLM или правила)
-> решение: принять выпуск (заменить пересечение с прошлым прогнозом) или отклонить.
Состояние (журнал и актуальный прогноз) хранится в api/state.sqlite.

Настройки через переменные окружения:
    AGENT_MODEL=lightgbm|baseline         модель (по умолчанию lightgbm)
    TRAINED_THROUGH=2025-12-31T18:00:00+00:00   граница обучения LightGBM
    LLM_MODEL=gpt-4o-mini + OPENAI_API_KEY   резюме от LLM (без них — по правилам)
    AGENT_SCHEDULE_HOURS=6                агент сам запускается каждые N часов (0 — выключено)
"""
from __future__ import annotations

import json
import os
import sqlite3
import sys
import threading
import time
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402
from fastapi import FastAPI, HTTPException, Query  # noqa: E402
from fastapi.responses import RedirectResponse  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from agent_analysis import analyze, summarize, utc  # noqa: E402
from data import config as C  # noqa: E402

MODEL_MODE = os.getenv("AGENT_MODEL", "lightgbm")
TRAINED_THROUGH = os.getenv("TRAINED_THROUGH", "2025-12-31T18:00:00+00:00")
LLM_MODEL = os.getenv("LLM_MODEL") or None
SCHEDULE_HOURS = int(os.getenv("AGENT_SCHEDULE_HOURS", "0") or 0)
DB_PATH = Path(os.getenv("AGENT_DB", ROOT / "api" / "state.sqlite"))
BACKTEST_DIRS = [ROOT / "artifacts" / "agent_lgbm_llm", ROOT / "artifacts" / "agent_lgbm"]

_lock = threading.Lock()
_scheduler_state: dict = {"enabled": SCHEDULE_HOURS > 0, "every_hours": SCHEDULE_HOURS, "last_error": None}


# ------------------------------------------------------------------ хранилище
def db() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS runs(issue_time TEXT PRIMARY KEY, created_at TEXT NOT NULL, record TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS latest(target_time TEXT PRIMARY KEY, power_pred REAL NOT NULL, issue_time TEXT NOT NULL);
    """)
    return conn


def local_iso(t: str) -> str:
    return (utc(t) + timedelta(hours=C.utc_offset_h())).replace(tzinfo=None).isoformat(timespec="minutes")


def floor_issue(now: datetime, step_h: int = 6) -> datetime:
    """Последний «час выпуска» (00/06/12/18 UTC), не позже now."""
    now = now.astimezone(timezone.utc).replace(minute=0, second=0, microsecond=0)
    return now - timedelta(hours=now.hour % step_h)


# ------------------------------------------------------------------ цикл агента
def run_agent(issue: datetime, mode: str = MODEL_MODE, llm_model: str | None = LLM_MODEL) -> dict:
    """Один полный цикл агента на момент выпуска issue (UTC). Результат — в журнал."""
    from agent_real import RealForecast  # тяжёлые импорты — только при запуске

    issue = issue.astimezone(timezone.utc)
    expected = [issue + timedelta(hours=h) for h in range(C.HORIZON_H)]
    with _lock:
        conn = db()
        try:
            previous = dict(conn.execute("SELECT target_time, power_pred FROM latest"))
            try:
                provider = RealForecast(mode, issue, trained_through=TRAINED_THROUGH if mode == "lightgbm" else None)
                forecast = provider(issue)
                report = analyze(forecast, expected, previous)
                provenance = provider.provenance
            except Exception as exc:  # ошибка конвейера -> выпуск отклонён, прошлый прогноз остаётся
                report = analyze([], expected, previous)
                report["errors"].insert(0, {"rule": "pipeline_error", "error_type": type(exc).__name__,
                                            "message": str(exc)[:300]})
                provenance = {"model": mode}
            report["summary"] = summarize(report, llm_model)
            record = {"mode": mode, "provenance": provenance, "issue_time": issue.isoformat(),
                      "replaced_points": report["overlap_points"] if report["accepted"] else 0,
                      "analysis": report, "source": "api"}
            with conn:
                conn.execute("INSERT OR REPLACE INTO runs VALUES (?, ?, ?)",
                             (issue.isoformat(), datetime.now(timezone.utc).isoformat(timespec="seconds"),
                              json.dumps(record, ensure_ascii=False, allow_nan=False, default=str)))
                if report["accepted"]:
                    conn.executemany("INSERT OR REPLACE INTO latest VALUES (?, ?, ?)",
                                     [(r["target_time"], r["power_pred"], issue.isoformat())
                                      for r in report["valid_forecast"]])
        finally:
            conn.close()
    return record


def backtest_dir() -> Path | None:
    return next((d for d in BACKTEST_DIRS if (d / "summary.json").exists()), None)


def backtest_runs() -> list[dict]:
    d = backtest_dir()
    if d is None or not (d / "runs.jsonl").exists():
        return []
    return [json.loads(line) | {"source": f"backtest:{d.name}"}
            for line in (d / "runs.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]


def api_runs() -> list[dict]:
    conn = db()
    try:
        return [json.loads(r[0]) for r in conn.execute("SELECT record FROM runs ORDER BY issue_time")]
    finally:
        conn.close()


def short(record: dict) -> dict:
    a = record["analysis"]
    return {"issue_time": record["issue_time"], "source": record.get("source", "api"), "mode": record.get("mode"),
            "accepted": a["accepted"], "errors": len(a["errors"]), "warnings": len(a["warnings"]),
            "replaced_points": record.get("replaced_points", 0), "mean_abs_revision": a.get("mean_abs_revision"),
            "summary": (a.get("summary") or {}).get("text"), "summary_source": (a.get("summary") or {}).get("source")}


def points(rows) -> list[dict]:
    return [{"target_time": t, "target_time_local": local_iso(t), "power_pred": round(float(p), 4),
             "power_pct": round(100 * float(p), 1), **({"issue_time": i} if i else {})} for t, p, i in rows]


# ------------------------------------------------------------------ расписание
def scheduler_loop() -> None:
    """Каждые SCHEDULE_HOURS часов агент сам делает выпуск на текущий момент."""
    while True:
        issue = floor_issue(datetime.now(timezone.utc), SCHEDULE_HOURS)
        done = {r["issue_time"] for r in api_runs()}
        if issue.isoformat() not in done:
            try:
                run_agent(issue)
                _scheduler_state["last_run"] = issue.isoformat()
                _scheduler_state["last_error"] = None
            except Exception as exc:  # noqa: BLE001
                _scheduler_state["last_error"] = f"{type(exc).__name__}: {exc}"[:300]
        _scheduler_state["next_check"] = (datetime.now(timezone.utc) + timedelta(minutes=5)).isoformat(timespec="seconds")
        time.sleep(300)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    db().close()
    if SCHEDULE_HOURS > 0:
        threading.Thread(target=scheduler_loop, daemon=True, name="agent-scheduler").start()
    yield


# ------------------------------------------------------------------ API
app = FastAPI(
    title="Агент прогноза выработки ВЭС",
    version="1.0",
    description=("Почасовой прогноз выработки ветроэлектростанции на 48 ч. Агент сам получает архивный прогноз "
                 "погоды Open-Meteo (ECMWF + GFS) на момент выпуска, считает мощность, проверяет результат, "
                 "пишет резюме (LLM или правила) и решает, принять ли выпуск. Мощность — доля от установленной "
                 "(0…1); время *_local — время SCADA станции (UTC+6)."),
    lifespan=lifespan,
)


class RunRequest(BaseModel):
    issue_time: str | None = Field(None, description="Момент выпуска, UTC, ISO. Пусто — ближайший прошедший "
                                                     "срок 00/06/12/18 UTC (реальный прогноз «на сейчас»).",
                                   examples=["2026-02-10T00:00:00+00:00"])
    mode: str = Field(MODEL_MODE, description="lightgbm или baseline")
    llm_model: str | None = Field(LLM_MODEL, description="модель OpenAI для резюме; пусто — резюме по правилам")


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse("/docs")


@app.get("/health", summary="Состояние сервиса")
def health():
    runs = api_runs()
    return {"status": "ok", "model": MODEL_MODE, "trained_through": TRAINED_THROUGH,
            "model_file": (ROOT / "model" / "lightgbm_model.txt").exists(),
            "llm": bool(LLM_MODEL and os.getenv("OPENAI_API_KEY")), "llm_model": LLM_MODEL,
            "api_runs": len(runs), "last_api_run": runs[-1]["issue_time"] if runs else None,
            "backtest": str(backtest_dir().relative_to(ROOT)) if backtest_dir() else None,
            "scheduler": _scheduler_state}


@app.post("/forecast/run", summary="Запустить полный цикл агента на момент выпуска")
def forecast_run(req: RunRequest):
    if req.mode not in {"lightgbm", "baseline"}:
        raise HTTPException(422, "mode: lightgbm или baseline")
    issue = utc(req.issue_time) if req.issue_time else floor_issue(datetime.now(timezone.utc))
    record = run_agent(issue, req.mode, req.llm_model)
    a = record["analysis"]
    return {**short(record), "errors_detail": a["errors"], "warnings_detail": a["warnings"],
            "forecast": points((p["target_time"], p["power_pred"], None) for p in a["valid_forecast"])}


@app.get("/forecast/latest", summary="Актуальный прогноз: для каждого часа — самый свежий принятый выпуск")
def forecast_latest(source: str = Query("api", description="api — выпуски через API; backtest — итог февраля"),
                    hours: int = Query(48, ge=1, le=2000, description="сколько ближайших часов вернуть")):
    if source == "backtest":
        d = backtest_dir()
        if d is None:
            raise HTTPException(404, "Нет результатов бэктеста в artifacts/")
        df = pd.read_csv(d / "february_latest.csv")
        rows = list(df[["target_time", "power_pred", "issue_time"]].itertuples(index=False, name=None))
        return {"source": f"backtest:{d.name}", "points": points(rows[:hours]), "total_hours": len(rows)}
    conn = db()
    try:
        rows = list(conn.execute("SELECT target_time, power_pred, issue_time FROM latest ORDER BY target_time"))
    finally:
        conn.close()
    if not rows:
        raise HTTPException(404, "Через API ещё не было принятых выпусков: вызовите POST /forecast/run "
                                 "или используйте source=backtest")
    newest_issue = max(r[2] for r in rows)
    ahead = [r for r in rows if utc(r[0]) >= utc(newest_issue)][:hours]  # горизонт от самого свежего выпуска
    return {"source": "api", "newest_issue": newest_issue, "points": points(ahead), "total_hours": len(rows)}


@app.get("/forecast/{issue_time}", summary="Выпуск целиком: прогноз на 48 ч, проверки и резюме")
def forecast_issue(issue_time: str):
    key = utc(issue_time).isoformat()
    for record in api_runs() + backtest_runs():
        if record["issue_time"] == key:
            a = record["analysis"]
            return {**short(record), "errors_detail": a["errors"], "warnings_detail": a["warnings"],
                    "forecast": points((p["target_time"], p["power_pred"], None) for p in a["valid_forecast"])}
    raise HTTPException(404, f"Выпуск {key} не найден ни в API, ни в бэктесте")


@app.get("/runs", summary="Журнал решений агента")
def runs(source: str = Query("all", description="all, api или backtest"), limit: int = Query(50, ge=1, le=500)):
    items = (api_runs() if source in {"all", "api"} else []) + (backtest_runs() if source in {"all", "backtest"} else [])
    items = sorted(items, key=lambda r: r["issue_time"], reverse=True)[:limit]
    return [short(r) for r in items]
