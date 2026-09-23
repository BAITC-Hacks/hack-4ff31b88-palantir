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
    "t2": (43.643194, 78.538833),  # 43°38'35.5"N 78°32'19.8"E
}
# Погоду берём в одной точке: турбины в 339 м друг от друга, а сетка NWP 9–25 км,
# поэтому обе попадают в одну ячейку модели.
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
# Январь 2026 — валидация: всё, что учится на данных (модель, кривая мощности для
# поиска простоев), видит только часы до этой даты (время SCADA).
VALID_START_LOCAL = "2026-01-01"
ARCHIVE_START = "2024-01-01"   # с этой даты у Open-Meteo есть Previous Runs
SCADA_END_LOCAL = "2026-01-31 23:59"
TEST_FIRST_ISSUE = "2026-01-31"
TEST_LAST_ISSUE = "2026-02-28"

# SCADA живёт по UTC+6 (старое время Алматы): в марте 2024 Казахстан перешёл
# на UTC+5, но в данных нет скачка на 01.03.2024 — часы SCADA не переводили.
# Подтверждено корреляцией ветра SCADA с прогнозом ECMWF (максимум на +6).
DEFAULT_UTC_OFFSET_H = 6
META_PATH = PROCESSED_DIR / "meta.json"


def utc_offset_h() -> int:
    """Сдвиг «Статистического времени» SCADA относительно UTC.

    build_dataset.py определяет его по данным и пишет в meta.json.
    """
    if META_PATH.exists():
        return int(json.loads(META_PATH.read_text(encoding="utf-8"))["utc_offset_h"])
    return DEFAULT_UTC_OFFSET_H
