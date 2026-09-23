"""Team entry point: daily 48-hour station forecasts for February 2026."""
import argparse
from datetime import date, datetime, time, timezone
import json
from pathlib import Path

from agent import Rules, demo_provider, run_backtest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--demo", action="store_true")
    mode.add_argument("--baseline", action="store_true")
    mode.add_argument("--model", help="module:function, returning target_time,power DataFrame")
    parser.add_argument("--model-metadata")
    parser.add_argument("--weather", choices=("fetch", "archive"), default="archive")
    parser.add_argument("--scada", default="data/processed/scada_hourly.parquet")
    parser.add_argument("--scada-offset", type=int)
    parser.add_argument("--out", required=True)
    parser.add_argument("--start", type=date.fromisoformat, default=date(2026, 1, 31))
    parser.add_argument("--end", type=date.fromisoformat, default=date(2026, 2, 27))
    parser.add_argument("--jump", type=float, default=0.30)
    parser.add_argument("--revision", type=float, default=0.15)
    parser.add_argument("--llm-model")
    args = parser.parse_args()
    from data import config as C
    offset = args.scada_offset if args.scada_offset is not None else C.utc_offset_h()
    if not 1 <= offset <= 12:
        parser.error("This next-local-day schedule requires a SCADA offset between +1 and +12")
    hour = 24 - offset
    if (Path(args.out) / "backtest.sqlite").exists():
        parser.error("Output already contains a run; choose a new --out directory")
    if args.demo:
        provider = demo_provider
    else:
        import pandas as pd
        from forecast_pipeline import ForecastPipeline, load_archives
        first = datetime.combine(args.start, time(hour), timezone.utc)
        last = datetime.combine(args.end, time(hour), timezone.utc)
        observations = pd.read_parquet(args.scada) if args.baseline else None
        # Check the model contract before triggering any network requests.
        provider = ForecastPipeline(observations=observations, model_entry=args.model,
                                    model_metadata=args.model_metadata, weather=args.weather,
                                    scada_offset=offset)
        if args.weather == "archive":
            provider.archives = load_archives(first, last)
    result = run_backtest(provider, args.out, start=args.start, end=args.end,
                          turbines=("station",), offset_hours=0, issue_hour=hour,
                          target_start="issue", evaluation_offset_hours=offset,
                          rules=Rules(args.jump, args.revision), llm_model=args.llm_model,
                          allow_demo=args.demo)
    result["prediction_mode"] = "demo" if args.demo else "baseline" if args.baseline else "team_model"
    (Path(args.out) / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["complete"] and result["accepted_runs"] == result["runs"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
