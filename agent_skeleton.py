"""Агент: погода и модель на заглушках, анализ работает по правилам.

Запуск:
    python agent_skeleton.py --issue-time 2026-01-31T00:00:00+00:00 --output artifacts/agent_stub.json

Все значения синтетические. Реальные погодный и модельный модули пока не подключены.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta
from pathlib import Path

from agent_analysis import analyze, summarize


def fetch_weather(issue_time: datetime) -> list[dict]:
    """Заглушка получения 48 почасовых прогнозов погоды."""
    return [
        {
            "target_time": (issue_time + timedelta(hours=hour)).isoformat(),
            "wind_speed": 6.0,
            "temperature": 10.0,
        }
        for hour in range(48)
    ]


def prepare(weather: list[dict]) -> list[dict]:
    """Заглушка подготовки признаков: пока только копирование строк."""
    return [dict(row) for row in weather]


def predict(features: list[dict]) -> list[dict]:
    """Заглушка модели: постоянная синтетическая мощность."""
    return [
        {"target_time": row["target_time"], "power_pred": 0.5}
        for row in features
    ]


def save(result: dict, output_path: str | Path) -> Path:
    """Сохранение результата демонстрационного запуска в JSON."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return path


def run(issue_time: datetime, output_path: str | Path, llm_model=None) -> dict:
    """Один проход по цепочке; без бэктеста и повторных расчётов."""
    print("fetch_weather")
    weather = fetch_weather(issue_time)
    print("prepare")
    features = prepare(weather)
    print("predict")
    forecast = predict(features)
    print("analyze")
    expected = [issue_time + timedelta(hours=h) for h in range(48)]
    analysis = analyze(forecast, expected)
    analysis["summary"] = summarize(analysis, llm_model)
    result = {
        "mode": "stub",
        "issue_time": issue_time.isoformat(),
        "forecast": forecast,
        "analysis": analysis,
    }
    print("save")
    save(result, output_path)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--issue-time", required=True, type=datetime.fromisoformat)
    parser.add_argument("--output", default="artifacts/agent_stub.json")
    parser.add_argument("--llm-model", help="Optional model name; requires OPENAI_API_KEY")
    args = parser.parse_args()
    result = run(args.issue_time, args.output, args.llm_model)
    print(f"Готово: {len(result['forecast'])} синтетических точек → {args.output}")


if __name__ == "__main__":
    main()
