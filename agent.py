"""Analysis and virtual-clock backtest. Python 3.10+, standard library core."""
from __future__ import annotations

import argparse
import csv
import importlib
import json
import math
import os
import sqlite3
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path


def stamp(value):
    result = datetime.fromisoformat(value) if isinstance(value, str) else value
    if not isinstance(result, datetime) or result.utcoffset() is None:
        raise ValueError("Every timestamp must have an explicit UTC offset")
    return result


def canonical(value):
    return stamp(value).astimezone(timezone.utc).isoformat()


@dataclass(frozen=True)
class Rules:
    jump: float = 0.30
    revision: float = 0.15
    epsilon: float = 1e-9

    def __post_init__(self):
        if not all(math.isfinite(x) and 0 <= x <= 1 for x in
                   (self.jump, self.revision, self.epsilon)):
            raise ValueError("Thresholds must be finite and in [0, 1]")


def analyze(rows, expected, previous=None, rules=Rules()):
    """Keys are (turbine_id, aware datetime); values are normalized power.

    Missing/nonfinite/out-of-range values reject the whole batch. Large
    ramps/revisions are warnings, never evidence of an incorrect forecast.
    """
    expected = {(str(t), canonical(ts)) for t, ts in expected}
    previous = previous or {}
    errors, warnings, values = [], [], {}
    seen = set()
    for row in rows:
        try:
            key = (str(row["turbine_id"]), canonical(row["valid_time"]))
        except (KeyError, TypeError, ValueError):
            errors.append({"rule": "invalid_key"})
            continue
        detail = {"turbine_id": key[0], "valid_time": key[1]}
        if key in seen:
            errors.append({"rule": "duplicate", **detail})
            continue
        seen.add(key)
        if key not in expected:
            errors.append({"rule": "unexpected_hour_or_turbine", **detail})
            continue
        raw = row.get("power")
        try:
            if raw is None or isinstance(raw, bool):
                raise ValueError()
            number = float(raw)
            if not math.isfinite(number):
                raise ValueError()
        except (TypeError, ValueError, OverflowError):
            errors.append({"rule": "missing_or_nonfinite", **detail})
            continue
        if not 0 <= number <= 1:
            errors.append({"rule": "outside_0_1", "value": number, **detail})
            continue
        values[key] = number
    for key in sorted(expected - seen):
        errors.append({"rule": "missing_hour", "turbine_id": key[0], "valid_time": key[1]})
    for (turbine, ts), number in sorted(values.items()):
        prev_key = (turbine, canonical(stamp(ts) - timedelta(hours=1)))
        # Include the boundary with the last previously forecast hour.
        before = values.get(prev_key, previous.get(prev_key))
        if before is not None and abs(number - before) > rules.jump:
            warnings.append({"rule": "hourly_jump", "turbine_id": turbine,
                             "valid_time": ts, "delta": number - before})
    changes = []
    for key in sorted(values.keys() & previous.keys()):
        delta = values[key] - previous[key]
        changes.append({"turbine_id": key[0], "valid_time": key[1],
                        "old": previous[key], "new": values[key], "delta": delta,
                        "changed": abs(delta) > rules.epsilon})
        if abs(delta) > rules.revision:
            warnings.append({"rule": "large_revision", "turbine_id": key[0],
                             "valid_time": key[1], "delta": delta})
    report = {"accepted": not errors, "expected_points": len(expected),
              "valid_points": len(values), "errors": errors, "warnings": warnings,
              "overlap_points": len(changes),
              "changed_points": sum(c["changed"] for c in changes),
              "mean_abs_revision": (sum(abs(c["delta"]) for c in changes) / len(changes)
                                    if changes else None),
              "max_abs_revision": max((abs(c["delta"]) for c in changes), default=None),
              "changes": changes}
    return report, values


def summarize(report, model=None):
    """LLM only describes computed statistics; its output cannot alter power."""
    fallback = (f"{'Принят' if report['accepted'] else 'Отклонён'}: "
                f"{report['valid_points']}/{report['expected_points']} точек. "
                f"Ошибок: {len(report['errors'])}, предупреждений: {len(report['warnings'])}. "
                f"Изменено {report['changed_points']} из {report['overlap_points']} "
                "перекрывающихся точек. Оценка точности требует фактической выработки.")
    if not os.getenv("OPENAI_API_KEY") or not model:
        return {"text": fallback, "source": "rules"}
    try:
        from openai import OpenAI
        facts = {k: v for k, v in report.items() if k not in ("changes", "errors", "warnings")}
        facts.update(error_count=len(report["errors"]), warning_count=len(report["warnings"]))
        with OpenAI(timeout=20.0, max_retries=0) as client:
            response = client.responses.create(
                model=model, store=False, max_output_tokens=500,
                instructions=("Опиши результат проверки прогноза ВЭС на русском в 2–3 предложениях. "
                              "Используй только переданные числа. Не придумывай причины скачков, "
                              "точность, погоду или рекомендации менять значения."),
                input=json.dumps(facts, ensure_ascii=False, allow_nan=False))
        if not response.output_text.strip():
            raise ValueError("Empty model output")
        return {"text": response.output_text.strip(), "source": "llm"}
    except Exception as exc:
        # Do not log raw exceptions: SDK messages may contain sensitive request data.
        return {"text": fallback, "source": "rules", "llm_error": type(exc).__name__}


def validate_provenance(batch, as_of, allow_demo=False):
    if batch["kind"] != "archived_forecast" and not (allow_demo and batch["kind"] == "synthetic_demo"):
        raise ValueError("Expected archived_forecast, not observations/reanalysis")
    if not batch.get("source") or not batch.get("weather_run_id"):
        raise ValueError("Source and weather_run_id are required")
    issued, available, trained = (stamp(batch[k]) for k in
                                 ("weather_issued_at", "weather_available_at", "trained_until"))
    if not issued <= available <= as_of or trained > as_of:
        raise ValueError("Future data: weather issue/availability or training cutoff exceeds virtual clock")


def run_backtest(provider, output, *, start=date(2026, 1, 31), end=date(2026, 2, 27),
                 turbines=("1", "2"), offset_hours=5, issue_hour=23,
                 rules=Rules(), llm_model=None, allow_demo=False,
                 target_start="next_midnight", evaluation_offset_hours=None):
    """Call provider(as_of, valid_times, turbine_ids) once per virtual day.

    output must not contain an existing database, preventing accidental history loss.
    The provider must fetch as-of weather and enforce its own training cutoff.
    """
    if end < start or not turbines or len(set(turbines)) != len(turbines):
        raise ValueError("Invalid dates or turbine IDs")
    local_tz = timezone(timedelta(hours=offset_hours))
    evaluation_tz = timezone(timedelta(hours=offset_hours if evaluation_offset_hours is None else evaluation_offset_hours))
    if target_start not in ("issue", "next_midnight"):
        raise ValueError("Unknown target-start convention")
    time(issue_hour)  # validate hour before creating output
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    database = output / "backtest.sqlite"
    # Exclusive creation avoids silently mixing independent experiments.
    with database.open("xb"):
        pass
    conn = sqlite3.connect(database)
    conn.executescript("""
        CREATE TABLE runs (as_of TEXT PRIMARY KEY, payload TEXT NOT NULL);
        CREATE TABLE latest (turbine TEXT, valid_time TEXT, power REAL,
                             as_of TEXT, PRIMARY KEY(turbine, valid_time));
    """)
    day = start
    try:
        while day <= end:
            as_of = datetime.combine(day, time(issue_hour), local_tz)
            first = datetime.combine(day + timedelta(days=1), time(), local_tz)
            if target_start == "issue":
                first = as_of
            times = tuple(first + timedelta(hours=h) for h in range(48))
            expected = [(t, ts) for t in turbines for ts in times]
            previous = {(t, ts): p for t, ts, p in
                        conn.execute("SELECT turbine, valid_time, power FROM latest")}
            batch = None
            try:
                batch = provider(as_of, times, tuple(turbines))
                validate_provenance(batch, as_of, allow_demo)
                report, values = analyze(batch["rows"], expected, previous, rules)
            except Exception as exc:
                # The run is recorded and the next day still executes.
                report, values = analyze([], expected, previous, rules)
                report["errors"].insert(0, {"rule": "provider_or_provenance_error",
                                             "error_type": type(exc).__name__})
                # Locally raised validation errors are useful to the operator.
                if isinstance(exc, (ValueError, FileNotFoundError, NotImplementedError)):
                    report["errors"][0]["detail"] = str(exc)[:400]
            report["summary"] = summarize(report, llm_model)
            record = {"as_of": as_of.isoformat(), "forecast_start": first.isoformat(),
                      "forecast_end_exclusive": (first + timedelta(hours=48)).isoformat(),
                      "provenance": {k: batch.get(k) for k in
                                     ("kind", "source", "weather_run_id", "weather_issued_at",
                                      "weather_available_at", "trained_until")} if isinstance(batch, dict) else None,
                      "analysis": report,
                      "forecast": [{"turbine_id": t, "valid_time": ts, "power": p}
                                   for (t, ts), p in sorted(values.items())],
                      "replaced_points": report["overlap_points"] if report["accepted"] else 0}
            # Revision log and replacement are committed together.
            with conn:
                conn.execute("INSERT INTO runs VALUES (?, ?)",
                             (as_of.isoformat(), json.dumps(record, ensure_ascii=False, allow_nan=False)))
                if report["accepted"]:
                    conn.executemany("INSERT OR REPLACE INTO latest VALUES (?, ?, ?, ?)",
                                     [(t, ts, p, as_of.isoformat()) for (t, ts), p in values.items()])
            print(f"{as_of.isoformat()} accepted={report['accepted']} replaced={record['replaced_points']}", flush=True)
            day += timedelta(days=1)
        records = [json.loads(r[0]) for r in conn.execute("SELECT payload FROM runs ORDER BY as_of")]
        month_start = datetime(2026, 2, 1, tzinfo=evaluation_tz)
        month_end = datetime(2026, 3, 1, tzinfo=evaluation_tz)
        latest = [{"turbine_id": t, "valid_time": stamp(ts).astimezone(evaluation_tz).isoformat(),
                   "power": p, "as_of": issue}
                  for t, ts, p, issue in conn.execute("SELECT * FROM latest ORDER BY turbine, valid_time")
                  if month_start <= stamp(ts) < month_end]
        result = {"mode": "demo" if allow_demo else "backtest", "runs": len(records),
                  "accepted_runs": sum(r["analysis"]["accepted"] for r in records),
                  "replaced_points": sum(r["replaced_points"] for r in records),
                  "february_points": len(latest), "expected_february_points": 672 * len(turbines),
                  "complete": len(latest) == 672 * len(turbines),
                  "accuracy_metrics": None, "target_start": target_start,
                  "evaluation_offset_hours": month_start.utcoffset().total_seconds() / 3600,
                  "forecast_kind": sorted({r["provenance"]["kind"] for r in records if r["provenance"]})}
        (output / "runs.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False, allow_nan=False) + "\n"
                                                   for r in records), encoding="utf-8")
        for name, data in (("february_latest.json", latest), ("summary.json", result)):
            (output / name).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        with (output / "february_latest.csv").open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["turbine_id", "valid_time", "power", "as_of"])
            writer.writeheader()
            writer.writerows(latest)
        with (output / "forecast_history.csv").open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["turbine_id", "valid_time", "power", "as_of", "lead_hours", "lead_end_hours", "accepted"])
            writer.writeheader()
            for record in records:
                for row in record["forecast"]:
                    writer.writerow(dict(row, as_of=record["as_of"],
                                         lead_hours=(stamp(row["valid_time"]) - stamp(record["as_of"])).total_seconds() / 3600,
                                         lead_end_hours=(stamp(row["valid_time"]) - stamp(record["as_of"])).total_seconds() / 3600 + 1,
                                         accepted=record["analysis"]["accepted"]))
        return result
    finally:
        conn.close()


def demo_provider(as_of, valid_times, turbine_ids):
    """Synthetic values for exercising the runner. NOT a weather/ML model."""
    rows = [{"turbine_id": t, "valid_time": ts.isoformat(),
             "power": round(0.45 + 0.25 * math.sin(ts.timestamp() / 18000 + i)
                            + 0.04 * math.sin(as_of.toordinal()), 6)}
            for i, t in enumerate(turbine_ids) for ts in valid_times]
    return {"kind": "synthetic_demo", "source": "synthetic sine wave, no weather",
            "weather_run_id": f"demo-{as_of.date()}", "weather_issued_at": as_of.isoformat(),
            "weather_available_at": as_of.isoformat(), "trained_until": as_of.isoformat(), "rows": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--demo", action="store_true")
    mode.add_argument("--provider", help="Import path module:function")
    parser.add_argument("--out", required=True)
    parser.add_argument("--start", type=date.fromisoformat, default=date(2026, 1, 31))
    parser.add_argument("--end", type=date.fromisoformat, default=date(2026, 2, 27))
    parser.add_argument("--offset-hours", type=int, default=5)
    parser.add_argument("--issue-hour", type=int, default=23)
    parser.add_argument("--turbines", nargs="+", default=["1", "2"])
    parser.add_argument("--jump", type=float, default=0.30)
    parser.add_argument("--revision", type=float, default=0.15)
    parser.add_argument("--llm-model", default=os.getenv("OPENAI_MODEL"))
    args = parser.parse_args()
    if args.demo:
        provider = demo_provider
    else:
        module, function = args.provider.split(":", 1)
        provider = getattr(importlib.import_module(module), function)
    result = run_backtest(provider, args.out, start=args.start, end=args.end,
                          turbines=tuple(args.turbines), offset_hours=args.offset_hours,
                          issue_hour=args.issue_hour, rules=Rules(args.jump, args.revision),
                          llm_model=args.llm_model, allow_demo=args.demo)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["complete"] and result["accepted_runs"] == result["runs"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
