"""Общие настройки модуля данных. Меняются здесь, а не в коде."""
from __future__ import annotations

import json
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent
RAW_DIR = DATA_DIR / "raw"            # сюда кладём CSV от организаторов
PROCESSED_DIR = DATA_DIR / "processed"
CACHE_DIR = DATA_DIR / "cache"        # кэш ответов Open-Meteo (можно удалять)

# Координаты из ссылок Google Maps в кейсе.
TURBINES = {
    "t1": (43.645150, 78.535604),
    "t2": (43.645150, 78.535604),  # TODO: вписать координаты турбины 2 (скорее всего в паре км)
}
# Погоду берём в одной точке: турбины рядом, сетка NWP 9–25 км.
SITE_LAT, SITE_LON = TURBINES["t1"]

# Погодный источник: Open-Meteo Previous Runs API (архив прогнозов, не факта).
# Префикс колонки -> имя модели в API.
MODELS = {
    "ecmwf": "ecmwf_ifs025",
    "gfs": "gfs_seamless",
}
# Имя переменной в API -> короткое имя колонки.
VARIABLES = {
    "wind_speed_10m": "ws10",
    "wind_speed_100m": "ws100",
    "wind_direction_100m": "wd100",
    "wind_gusts_10m": "gust10",
    "temperature_2m": "t2m",
}
DAY_OFFSETS = (1, 2, 3)   # _previous_day1.._previous_day3

# Консервативная задержка публикации прогона NWP (ECMWF open data выходит
# ~7 ч после старта прогона). Прогноз с лидом L берётся из прогона,
# стартовавшего не позже чем за L + PUBLISH_DELAY_H часов до цели.
PUBLISH_DELAY_H = 8
HORIZON_H = 48
ISSUE_HOUR_UTC = 0        # момент выпуска прогноза: 00:00 UTC (05:00 по Астане)

# Периоды
ARCHIVE_START = "2024-01-01"   # с этой даты у Open-Meteo есть Previous Runs
SCADA_END_LOCAL = "2026-01-31 23:59"
TEST_FIRST_ISSUE = "2026-01-31"
TEST_LAST_ISSUE = "2026-02-28"

DEFAULT_UTC_OFFSET_H = 5  # Казахстан (единый UTC+5 с 01.03.2024)
META_PATH = PROCESSED_DIR / "meta.json"


def utc_offset_h() -> int:
    """Сдвиг «Статистического времени» SCADA относительно UTC.

    build_dataset.py определяет его по данным и пишет в meta.json.
    """
    if META_PATH.exists():
        return int(json.loads(META_PATH.read_text(encoding="utf-8"))["utc_offset_h"])
    return DEFAULT_UTC_OFFSET_H
