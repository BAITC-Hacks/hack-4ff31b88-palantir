"""Ежедневный бэктест агента по календарю data.config."""
from __future__ import annotations

import argparse
import csv
from datetime import date, datetime, time, timedelta, timezone
import json
from pathlib import Path
import sqlite3

from agent_analysis import analyze, summarize, utc
from agent_skeleton import fetch_weather, prepare, predict
from data import config as C


def stub_forecast(issue_time):
    return predict(prepare(fetch_weather(issue_time)))


def run_backtest(output, *, forecast_fn=stub_forecast, start=None, end=None,
                 issue_hour=None, scada_offset=None, jump=0.30, revision=0.15, llm_model=None,
                 mode='stub', provenance=None):
    """Один ряд станции. callback получает виртуальное UTC-время, возвращает список.

    По умолчанию проверяется механика на заглушках; CLI --mode подключает модель.
    Обновление актуального прогноза и журнал фиксируются одной транзакцией.
    """
    start = date.fromisoformat(start or C.TEST_FIRST_ISSUE)
    end = date.fromisoformat(end or C.TEST_LAST_ISSUE)
    hour = C.ISSUE_HOUR_UTC if issue_hour is None else issue_hour
    offset = C.utc_offset_h() if scada_offset is None else scada_offset
    if end < start:
        raise ValueError("End date must not precede start date")
    time(hour)
    local_tz = timezone(timedelta(hours=offset))
    # Validate rule thresholds before creating the run directory.
    analyze([], [], jump=jump, revision=revision)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    database = output / "backtest.sqlite"
    with database.open("xb"):
        pass  # Never overwrite an existing experiment.
    conn = sqlite3.connect(database)
    try:
        conn.executescript("""
            CREATE TABLE runs(issue_time TEXT PRIMARY KEY, record TEXT NOT NULL);
            CREATE TABLE latest(target_time TEXT PRIMARY KEY, power_pred REAL NOT NULL, issue_time TEXT NOT NULL);
        """)
        day = start
        while day <= end:
            issue = datetime.combine(day, time(hour), timezone.utc)
            expected = [issue + timedelta(hours=h) for h in range(C.HORIZON_H)]
            previous = dict(conn.execute("SELECT target_time, power_pred FROM latest"))
            try:
                forecast = forecast_fn(issue)
                report = analyze(forecast, expected, previous, jump=jump, revision=revision)
            except Exception as exc:
                report = analyze([], expected, previous, jump=jump, revision=revision)
                report["errors"].insert(0, {"rule": "pipeline_error", "error_type": type(exc).__name__})
            report["summary"] = summarize(report, llm_model)
            record = {"mode": mode, "provenance": provenance or {}, "issue_time": issue.isoformat(),
                      "replaced_points": report["overlap_points"] if report["accepted"] else 0,
                      "analysis": report}
            with conn:
                conn.execute("INSERT INTO runs VALUES (?, ?)",
                             (issue.isoformat(), json.dumps(record, ensure_ascii=False, allow_nan=False)))
                if report["accepted"]:
                    conn.executemany("INSERT OR REPLACE INTO latest VALUES (?, ?, ?)",
                                     [(r["target_time"], r["power_pred"], issue.isoformat())
                                      for r in report["valid_forecast"]])
            print(f"{issue.isoformat()} accepted={report['accepted']} replaced={record['replaced_points']}", flush=True)
            day += timedelta(days=1)
        records = [json.loads(row[0]) for row in conn.execute("SELECT record FROM runs ORDER BY issue_time")]
        begin = datetime(2026, 2, 1, tzinfo=local_tz)
        stop = datetime(2026, 3, 1, tzinfo=local_tz)
        rows = [{"target_time": t, "target_time_local": utc(t).astimezone(local_tz).isoformat(),
                 "power_pred": p, "issue_time": issue}
                for t, p, issue in conn.execute("SELECT * FROM latest ORDER BY target_time")
                if begin <= utc(t) < stop]
        with (output / "february_latest.csv").open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["target_time", "target_time_local", "power_pred", "issue_time"])
            writer.writeheader()
            writer.writerows(rows)
        with (output / "forecast_history.csv").open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["issue_time", "target_time", "lead_hour", "power_pred", "accepted"])
            writer.writeheader()
            for r in records:
                for point in r["analysis"]["valid_forecast"]:
                    writer.writerow(dict(point, issue_time=r["issue_time"], accepted=r["analysis"]["accepted"],
                                         lead_hour=int((utc(point["target_time"]) - utc(r["issue_time"])).total_seconds() / 3600) + 1))
        (output / "runs.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False, allow_nan=False) + "\n"
                                                   for r in records), encoding="utf-8")
        summary = {"mode": mode, "provenance": provenance or {}, "first_issue_date": start.isoformat(), "last_issue_date": end.isoformat(),
                   "issue_hour_utc": hour, "scada_offset_h": offset,
                   "runs": len(records), "accepted_runs": sum(r["analysis"]["accepted"] for r in records),
                   "replaced_points": sum(r["replaced_points"] for r in records),
                   "changed_points": sum(r["analysis"]["changed_points"] for r in records if r["analysis"]["accepted"]),
                   "february_hours": len(rows), "expected_february_hours": 672, "complete": len(rows) == 672}
        (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        return summary
    finally:
        conn.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--start")
    parser.add_argument("--end")
    parser.add_argument("--jump", type=float, default=0.30)
    parser.add_argument("--revision", type=float, default=0.15)
    parser.add_argument("--llm-model")
    parser.add_argument('--mode', choices=['stub', 'baseline', 'lightgbm'], default='stub')
    parser.add_argument('--trained-through', help='UTC availability of last training label, confirmed by model owner')
    args = parser.parse_args()
    provider = stub_forecast
    if args.mode != 'stub':
        from agent_real import build_forecaster
        provider = build_forecaster(args.mode, args.start or C.TEST_FIRST_ISSUE,
                                    args.end or C.TEST_LAST_ISSUE, trained_through=args.trained_through)
    result = run_backtest(args.out, start=args.start, end=args.end,
                          jump=args.jump, revision=args.revision, llm_model=args.llm_model,
                          forecast_fn=provider, mode=args.mode, provenance=getattr(provider, 'provenance', None))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["complete"] and result["accepted_runs"] == result["runs"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
