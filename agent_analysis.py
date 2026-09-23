"""Проверки прогноза станции и необязательное текстовое резюме."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
import math
import os


def utc(value) -> datetime:
    """В модуле данных команды naive-время означает UTC."""
    result = datetime.fromisoformat(value) if isinstance(value, str) else value
    if not isinstance(result, datetime):
        raise ValueError("Expected datetime or ISO timestamp")
    if result.tzinfo is None:
        result = result.replace(tzinfo=timezone.utc)
    return result.astimezone(timezone.utc)


def analyze(forecast, expected_times, previous=None, *, jump=0.30, revision=0.15):
    """forecast: список {target_time, power_pred}; previous: UTC ISO -> power.

    Ошибки отклоняют весь пакет. Скачки и пересмотры дают предупреждения.
    0 и 1 допустимы. Числа не обрезаются и пропуски не заполняются.
    """
    if not all(math.isfinite(x) and 0 <= x <= 1 for x in (jump, revision)):
        raise ValueError("Thresholds must be finite and in [0, 1]")
    expected = {utc(t).isoformat() for t in expected_times}
    previous = {utc(t).isoformat(): p for t, p in (previous or {}).items()}
    values, seen, errors, warnings = {}, set(), [], []
    for row in forecast:
        try:
            target = utc(row["target_time"]).isoformat()
        except (KeyError, TypeError, ValueError):
            errors.append({"rule": "invalid_timestamp"})
            continue
        if target in seen:
            errors.append({"rule": "duplicate_hour", "target_time": target})
            continue
        seen.add(target)
        if target not in expected:
            errors.append({"rule": "unexpected_hour", "target_time": target})
            continue
        raw = row.get("power_pred")
        try:
            if raw is None or isinstance(raw, bool):
                raise ValueError()
            number = float(raw)
            if not math.isfinite(number):
                raise ValueError()
        except (ValueError, TypeError, OverflowError):
            errors.append({"rule": "missing_or_nonfinite", "target_time": target})
            continue
        if not 0 <= number <= 1:
            errors.append({"rule": "outside_0_1", "target_time": target, "value": number})
            continue
        values[target] = number
    errors.extend({"rule": "missing_hour", "target_time": t} for t in sorted(expected - seen))
    for target, number in sorted(values.items()):
        before_key = (utc(target) - timedelta(hours=1)).isoformat()
        before = values.get(before_key, previous.get(before_key))
        if before is not None and abs(number - before) > jump:
            warnings.append({"rule": "hourly_jump", "target_time": target, "delta": number - before})
    changes = []
    for target in sorted(values.keys() & previous.keys()):
        delta = values[target] - previous[target]
        changes.append({"target_time": target, "old": previous[target], "new": values[target],
                        "delta": delta, "changed": abs(delta) > 1e-9})
        if abs(delta) > revision:
            warnings.append({"rule": "large_revision", "target_time": target, "delta": delta})
    return {"accepted": not errors, "expected_points": len(expected), "valid_points": len(values),
            "errors": errors, "warnings": warnings, "overlap_points": len(changes),
            "changed_points": sum(c["changed"] for c in changes),
            "mean_abs_revision": sum(abs(c["delta"]) for c in changes) / len(changes) if changes else None,
            "max_abs_revision": max((abs(c["delta"]) for c in changes), default=None),
            "changes": changes,
            "valid_forecast": [{"target_time": t, "power_pred": p} for t, p in sorted(values.items())]}


def summarize(report, llm_model=None):
    fallback = (f"{'Принят' if report['accepted'] else 'Отклонён'} прогноз: "
                f"{report['valid_points']}/{report['expected_points']} точек; "
                f"ошибок {len(report['errors'])}, предупреждений {len(report['warnings'])}. "
                f"Изменено {report['changed_points']} из {report['overlap_points']} перекрывающихся точек.")
    if not llm_model or not os.getenv("OPENAI_API_KEY"):
        return {"source": "rules", "text": fallback}
    try:
        from openai import OpenAI
        facts = {k: report[k] for k in ("accepted", "expected_points", "valid_points", "overlap_points",
                                        "changed_points", "mean_abs_revision", "max_abs_revision")}
        facts.update(error_count=len(report["errors"]), warning_count=len(report["warnings"]))
        with OpenAI(timeout=20.0, max_retries=0) as client:
            response = client.responses.create(
                model=llm_model, store=False, max_output_tokens=400,
                instructions="Опиши проверки прогноза ВЭС по переданным числам в 2–3 предложениях на русском. "
                             "Не придумывай причины, погоду или точность. Изменение версии не равно ошибке прогноза.",
                input=json.dumps(facts, ensure_ascii=False, allow_nan=False))
        text = response.output_text.strip()
        if not text:
            raise ValueError("Empty LLM response")
        return {"source": "llm", "text": text}
    except Exception as exc:
        return {"source": "rules", "text": fallback, "llm_error": type(exc).__name__}
