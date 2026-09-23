# hack-4ff31b88-palantir
Hackathon team repository for Palantir

## Модель и метрики

Запуск модели: `python -m model.train`. Обучение идёт на `train.parquet` до
01.01.2026, январь 2026 используется для валидации, а февраль — для итогового
прогноза. Используются ветер и его ансамбль, направление в sin/cos,
температура, час, месяц и календарные циклы. Предсказание обрезается в [0, 1].

`nMAE = 100 * mean(abs(y - pred))`; результаты LightGBM записываются в
`model/metrics_lgbm.csv`, прогноз февраля — в `model/forecast_lgbm_feb.csv`.
После появления фактической SCADA скрипт также записывает сравнение LightGBM,
persistence, climatology и power curve за февраль в `model/metrics_february.csv`
с разбиением на горизонты 1–24 и 25–48 часов.

Бейзлайны на январе 2026:

| Модель | 1–24 ч | 25–48 ч | Все горизонты |
|---|---:|---:|---:|
| Persistence | 30.53% | 33.76% | 32.14% |
| Climatology | 29.40% | 29.40% | 29.40% |
| Power curve | 21.81% | 21.92% | 21.86% |

