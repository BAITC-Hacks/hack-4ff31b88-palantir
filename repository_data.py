"""Read-only, bounded access to the repository's published forecast artifacts."""
from __future__ import annotations

import csv
from datetime import date, datetime, time, timedelta, timezone
import io
import json
import math
import os
from pathlib import Path
import re
from typing import Any


class RepositoryDataError(ValueError):
    """An unavailable or invalid artifact, safe to explain to the user."""


class RepositoryData:
    MAX_FILE_BYTES = 16 * 1024 * 1024
    MAX_ROWS = 100_000
    MAX_RUNS = 100
    MAX_LOG_ROWS = 5_000
    MAX_FORECAST_OUTPUT = 2_000
    MAX_CHECK_OUTPUT = 50

    def __init__(self, root: Path):
        self.root = Path(root).resolve()
        if not self.root.is_dir():
            raise RepositoryDataError("Папка репозитория не найдена.")
        self.issues: list[str] = []
        self._runs: dict[str, Path] = {}

    def _safe_path(self, relative: str | Path) -> Path:
        candidate = self.root / relative
        try:
            resolved = candidate.resolve()
            resolved.relative_to(self.root)
        except (OSError, RuntimeError, ValueError) as exc:
            raise RepositoryDataError("Путь выходит за пределы репозитория.") from exc
        return resolved

    def _read(self, relative: str | Path) -> str:
        path = self._safe_path(relative)
        try:
            with path.open("rb") as stream:
                raw = stream.read(self.MAX_FILE_BYTES + 1)
            if len(raw) > self.MAX_FILE_BYTES:
                raise RepositoryDataError(f"Файл {relative} превышает допустимый размер.")
            return raw.decode("utf-8-sig")
        except RepositoryDataError:
            raise
        except (OSError, UnicodeError) as exc:
            raise RepositoryDataError(f"Не удалось прочитать файл {relative}.") from exc

    @staticmethod
    def _json(text: str, source: str) -> Any:
        def reject_constant(value):
            raise ValueError(f"Недопустимое число: {value}")

        def unique_keys(pairs):
            result = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError("Повторяющийся ключ JSON")
                result[key] = value
            return result

        try:
            value = json.loads(text, parse_constant=reject_constant, object_pairs_hook=unique_keys)
            # JSON exponents can also overflow to infinity without parse_constant.
            json.dumps(value, allow_nan=False)
            return value
        except (ValueError, TypeError, RecursionError) as exc:
            raise RepositoryDataError(f"Некорректный JSON в {source}.") from exc

    def _summary(self, run_id: str) -> dict:
        source = f"{run_id}/summary.json"
        summary = self._json(self._read(source), source)
        if not isinstance(summary, dict) or not isinstance(summary.get("mode"), str):
            raise RepositoryDataError(f"Некорректная сводка в {source}.")
        if len(json.dumps(summary, ensure_ascii=False)) > 65_536:
            raise RepositoryDataError(f"Сводка {source} превышает допустимый размер.")
        self._local_timezone(summary)
        return summary

    def list_runs(self) -> list[dict]:
        self.issues = []
        self._runs = {}
        found = []
        for directory in ("artifacts", "outputs"):
            try:
                base = self._safe_path(directory)
            except RepositoryDataError as exc:
                self.issues.append(f"{directory}: {exc}")
                continue
            if not base.is_dir():
                continue
            for current, directories, files in os.walk(base, followlinks=False):
                # Also reject junctions and directory links before descending.
                safe_directories = []
                for name in sorted(directories):
                    candidate = Path(current) / name
                    try:
                        resolved = self._safe_path(candidate)
                        if not candidate.is_symlink() and resolved == candidate.absolute():
                            safe_directories.append(name)
                        else:
                            self.issues.append("Каталог-ссылка пропущен при поиске прогонов.")
                    except RepositoryDataError:
                        self.issues.append("Каталог за пределами репозитория пропущен.")
                directories[:] = safe_directories
                if "summary.json" not in files:
                    continue
                run_id = Path(current).relative_to(self.root).as_posix()
                try:
                    summary = self._summary(run_id)
                except RepositoryDataError as exc:
                    self.issues.append(str(exc))
                    continue
                found.append({"id": run_id, "summary": summary})
                self._runs[run_id] = Path(current)
                if len(found) >= self.MAX_RUNS:
                    self.issues.append("Поиск ограничен первыми 100 прогонами.")
                    break
            if len(found) >= self.MAX_RUNS:
                break
        return sorted(found, key=lambda item: (item["id"] != "artifacts/agent_lgbm_llm", item["id"]))

    def _run(self, run_id: str) -> str:
        if not isinstance(run_id, str):
            raise RepositoryDataError("Идентификатор прогона должен быть строкой.")
        # Rebuild the allowlist so newly added or removed artifacts are reflected.
        self.list_runs()
        if run_id not in self._runs:
            raise RepositoryDataError("Прогон не найден. Выберите его из списка доступных прогонов.")
        self._safe_path(run_id)
        return run_id

    @staticmethod
    def _local_timezone(summary: dict) -> timezone:
        provenance = summary.get("provenance", {})
        if not isinstance(provenance, dict):
            raise RepositoryDataError("Некорректные сведения о происхождении прогноза.")
        offset = summary.get("scada_offset_h", provenance.get("utc_offset_h", 6))
        if isinstance(offset, bool) or not isinstance(offset, (float, int)) or not math.isfinite(offset):
            raise RepositoryDataError("Некорректный часовой пояс прогона.")
        if not -24 < offset < 24:
            raise RepositoryDataError("Некорректный часовой пояс прогона.")
        return timezone(timedelta(hours=offset))

    @staticmethod
    def _datetime(value: Any, field: str) -> datetime:
        try:
            if not isinstance(value, str) or "T" not in value:
                raise ValueError
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            if parsed.tzinfo is None or parsed.utcoffset() is None:
                raise ValueError
            return parsed.astimezone(timezone.utc)
        except (ValueError, OverflowError) as exc:
            raise RepositoryDataError(f"Некорректное время {field}: укажите ISO-время с часовым поясом.") from exc

    @staticmethod
    def _number(value: Any, field: str, nullable: bool = False) -> float | None:
        if nullable and (value is None or value == ""):
            return None
        try:
            if isinstance(value, bool):
                raise ValueError
            number = float(value)
            if not math.isfinite(number):
                raise ValueError
            return number
        except (ValueError, TypeError) as exc:
            raise RepositoryDataError(f"Некорректное числовое значение {field}.") from exc

    @classmethod
    def _count(cls, value: Any, field: str) -> int:
        number = cls._number(value, field)
        if number < 0 or number > 1_000_000_000 or not number.is_integer():
            raise RepositoryDataError(f"Некорректный счётчик {field}.")
        return int(number)

    @staticmethod
    def _limit(value: int, maximum: int) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= maximum:
            raise RepositoryDataError(f"Лимит должен быть целым числом от 1 до {maximum}.")
        return value

    def _csv(self, source: str, required: set[str]) -> list[dict]:
        try:
            reader = csv.DictReader(io.StringIO(self._read(source)), strict=True)
            fields = reader.fieldnames
            if not fields or len(fields) != len(set(fields)) or not required.issubset(fields):
                raise RepositoryDataError(f"В {source} отсутствуют обязательные столбцы или есть дубликаты.")
            rows = []
            for row in reader:
                if None in row or any(value is None for value in row.values()):
                    raise RepositoryDataError(f"Нарушена структура строки CSV в {source}.")
                rows.append(row)
                if len(rows) > self.MAX_ROWS:
                    raise RepositoryDataError(f"Файл {source} содержит слишком много строк.")
            return rows
        except csv.Error as exc:
            raise RepositoryDataError(f"Некорректный CSV в {source}.") from exc

    def _log_rows(self, run_id: str) -> list[dict]:
        source = f"{run_id}/runs.jsonl"
        rows = []
        seen = set()
        for line_number, line in enumerate(self._read(source).splitlines(), 1):
            if not line.strip():
                continue
            record = self._json(line, f"{source}:{line_number}")
            if not isinstance(record, dict) or not isinstance(record.get("analysis"), dict):
                raise RepositoryDataError(f"Некорректная запись проверки в {source}:{line_number}.")
            issue = self._datetime(record.get("issue_time"), "issue_time").isoformat()
            if issue in seen:
                raise RepositoryDataError(f"Повторяющийся выпуск в {source}.")
            seen.add(issue)
            analysis = record["analysis"]
            if type(analysis.get("accepted")) is not bool:
                raise RepositoryDataError(f"Некорректный статус проверки в {source}.")
            errors, warnings = analysis.get("errors"), analysis.get("warnings")
            summary = analysis.get("summary", {})
            if not isinstance(errors, list) or not isinstance(warnings, list):
                raise RepositoryDataError(f"Некорректный список ошибок или предупреждений в {source}.")
            if not isinstance(summary, dict) or not isinstance(summary.get("source"), str) or not isinstance(summary.get("text"), str):
                raise RepositoryDataError(f"Некорректное резюме проверки в {source}.")
            row = {
                "issue_time": issue, "accepted": analysis["accepted"],
                "error_count": len(errors), "warning_count": len(warnings),
                "overlap_points": self._count(analysis.get("overlap_points"), "overlap_points"),
                "changed_points": self._count(analysis.get("changed_points"), "changed_points"),
                "mean_abs_revision": self._number(analysis.get("mean_abs_revision"), "mean_abs_revision", True),
                "max_abs_revision": self._number(analysis.get("max_abs_revision"), "max_abs_revision", True),
                "errors": errors, "warnings": warnings,
                "summary": {"source": summary["source"], "text": summary["text"]},
            }
            if "llm_error" in summary:
                row["summary"]["llm_error"] = str(summary["llm_error"])
            rows.append(row)
            if len(rows) > self.MAX_LOG_ROWS:
                raise RepositoryDataError(f"Слишком много выпусков в {source}.")
        return sorted(rows, key=lambda row: row["issue_time"])

    def run_summary(self, run_id: str) -> dict:
        run_id = self._run(run_id)
        summary = self._summary(run_id)
        rows = self._log_rows(run_id)
        counts = {
            "total": len(rows), "accepted": sum(row["accepted"] for row in rows),
            "rejected": sum(not row["accepted"] for row in rows),
            "errors": sum(row["error_count"] for row in rows),
            "warnings": sum(row["warning_count"] for row in rows),
            "llm": sum(row["summary"]["source"] == "llm" for row in rows),
            "rules": sum(row["summary"]["source"] == "rules" for row in rows),
        }
        for field, actual in (("runs", counts["total"]), ("accepted_runs", counts["accepted"])):
            if field in summary and self._count(summary[field], field) != actual:
                raise RepositoryDataError(f"Счётчик {field} в summary.json не совпадает с журналом выпусков.")
        return {"sources": [f"{run_id}/summary.json", f"{run_id}/runs.jsonl"], "summary": summary, "checks": counts}

    def _boundary(self, value: str | None, local_tz: timezone, end: bool = False) -> tuple[datetime | None, bool]:
        if value is None:
            return None, False
        if isinstance(value, str) and re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
            try:
                day = date.fromisoformat(value)
                if end:
                    day += timedelta(days=1)
                return datetime.combine(day, time(), local_tz).astimezone(timezone.utc), end
            except (ValueError, OverflowError) as exc:
                raise RepositoryDataError("Некорректная календарная дата фильтра.") from exc
        return self._datetime(value, "фильтра"), False

    def forecast(self, run_id: str, start: str | None = None, end: str | None = None,
                 issue_time: str | None = None, limit: int = 2000) -> dict:
        run_id = self._run(run_id)
        limit = self._limit(limit, self.MAX_FORECAST_OUTPUT)
        local_tz = self._local_timezone(self._summary(run_id))
        begin, _ = self._boundary(start, local_tz)
        stop, exclusive_stop = self._boundary(end, local_tz, end=True)
        if begin is not None and stop is not None and (begin > stop or (exclusive_stop and begin == stop)):
            raise RepositoryDataError("Начало периода должно предшествовать его концу.")
        issue_filter = self._datetime(issue_time, "issue_time") if issue_time is not None else None
        filename = "forecast_history.csv" if issue_filter is not None else "february_latest.csv"
        source = f"{run_id}/{filename}"
        required = {"target_time", "issue_time", "power_pred"}
        if issue_filter is not None:
            required |= {"accepted", "lead_hour"}
        rows = []
        seen = set()
        for raw in self._csv(source, required):
            target = self._datetime(raw["target_time"], "target_time")
            issue = self._datetime(raw["issue_time"], "issue_time")
            if target < issue:
                raise RepositoryDataError(f"Прогноз предшествует моменту выпуска в {source}.")
            key = (issue, target) if issue_filter is not None else target
            if key in seen:
                raise RepositoryDataError(f"Повторяющееся время прогноза в {source}.")
            seen.add(key)
            power = self._number(raw["power_pred"], "power_pred")
            if not 0 <= power <= 1:
                raise RepositoryDataError(f"Мощность в {source} должна быть долей от 0 до 1.")
            local_target = target.astimezone(local_tz)
            if "target_time_local" in raw:
                if self._datetime(raw["target_time_local"], "target_time_local") != target:
                    raise RepositoryDataError(f"Местное время не соответствует UTC в {source}.")
                original_local = datetime.fromisoformat(raw["target_time_local"].replace("Z", "+00:00"))
                if original_local.utcoffset() != local_tz.utcoffset(None):
                    raise RepositoryDataError(f"Местное время не соответствует часовому поясу прогона в {source}.")
            row = {"target_time": target.isoformat(), "target_time_local": local_target.isoformat(),
                   "power_pred": power, "issue_time": issue.isoformat()}
            if "lead_hour" in raw:
                lead = self._count(raw["lead_hour"], "lead_hour")
                if lead != (target - issue).total_seconds() / 3600 + 1:
                    raise RepositoryDataError(f"Горизонт lead_hour не соответствует времени в {source}.")
                row["lead_hour"] = lead
            accepted = True
            if "accepted" in raw:
                if raw["accepted"].lower() not in ("true", "false"):
                    raise RepositoryDataError(f"Некорректный статус accepted в {source}.")
                accepted = raw["accepted"].lower() == "true"
            if not accepted or (issue_filter is not None and issue != issue_filter):
                continue
            if begin is not None and target < begin:
                continue
            if stop is not None and (target >= stop if exclusive_stop else target > stop):
                continue
            rows.append(row)
        rows.sort(key=lambda row: (row["target_time"], row["issue_time"]))
        powers = [row["power_pred"] for row in rows]
        minimum = min(rows, key=lambda row: row["power_pred"]) if rows else None
        maximum = max(rows, key=lambda row: row["power_pred"]) if rows else None
        offset = local_tz.utcoffset(None).total_seconds() / 3600
        return {
            "sources": [source], "rows": rows[:limit], "total_rows": len(rows), "truncated": len(rows) > limit,
            "statistics": {"mean_power": math.fsum(powers) / len(powers) if powers else None,
                           "min_power": min(powers) if powers else None, "max_power": max(powers) if powers else None,
                           "min_power_time": minimum["target_time_local"] if minimum else None,
                           "max_power_time": maximum["target_time_local"] if maximum else None},
            "unit": "fraction_of_installed_capacity", "timezone": f"UTC{offset:+g}",
        }

    def checks(self, run_id: str, issue_time: str | None = None, limit: int = 50) -> dict:
        run_id = self._run(run_id)
        limit = self._limit(limit, self.MAX_CHECK_OUTPUT)
        rows = self._log_rows(run_id)
        if issue_time is not None:
            issue = self._datetime(issue_time, "issue_time").isoformat()
            rows = [row for row in rows if row["issue_time"] == issue]
        output = rows[:limit]
        for row in output:
            details_truncated = len(row["errors"]) > 100 or len(row["warnings"]) > 100 or len(row["summary"]["text"]) > 8000
            row["errors"] = row["errors"][:100]
            row["warnings"] = row["warnings"][:100]
            row["summary"]["text"] = row["summary"]["text"][:8000]
            row["details_truncated"] = details_truncated
        return {"sources": [f"{run_id}/runs.jsonl"], "rows": output, "total_rows": len(rows), "truncated": len(rows) > limit}

    def metrics(self) -> dict:
        sources, rows = [], []
        for filename, period in (("metrics_baseline.csv", "2026-01"), ("metrics_lgbm.csv", "2026-01"), ("metrics_february.csv", "2026-02")):
            source = f"model/{filename}"
            if not self._safe_path(source).exists():
                continue
            records = self._csv(source, {"model", "horizon", "nMAE_%", "nRMSE_%", "bias_%", "n"})
            for record in records:
                row = {"model": record["model"], "horizon": record["horizon"],
                       "n": self._count(record["n"], "n"), "period": period, "source": source}
                row["available"] = row["n"] > 0
                for field in ("nMAE_%", "nRMSE_%", "bias_%"):
                    value = self._number(record[field], field, nullable=True)
                    row[field] = value if row["available"] else None
                rows.append(row)
            sources.append(source)
        return {"sources": sources, "rows": rows}
