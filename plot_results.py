"""Join latest predictions to hourly facts and render per-turbine plots."""
import argparse
from pathlib import Path


def plot(predictions, facts, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    from agent import canonical

    def read(path, value):
        frame = pd.read_csv(path, dtype={"turbine_id": str})
        frame["valid_time"] = frame["valid_time"].map(canonical)
        frame["valid_time"] = pd.to_datetime(frame["valid_time"], utc=True)
        frame[value] = pd.to_numeric(frame[value], errors="raise")
        if not frame[value].between(0, 1).all():
            raise ValueError(f"{value} must be finite and between 0 and 1")
        if frame.duplicated(["turbine_id", "valid_time"]).any():
            raise ValueError("Duplicate turbine/hour keys")
        return frame

    predicted = read(predictions, "power")
    if facts:
        actual = read(facts, "actual_power")
        joined = predicted.merge(actual[["turbine_id", "valid_time", "actual_power"]],
                                 on=["turbine_id", "valid_time"], how="left", validate="one_to_one")
    else:
        joined = predicted.copy()
        joined["actual_power"] = float("nan")
    if facts and not joined["actual_power"].notna().any():
        raise ValueError("No matching actual February observations")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for turbine, group in joined.groupby("turbine_id"):
        group = group.sort_values("valid_time").set_index("valid_time")
        # Reindex so missing forecast hours are visible as gaps in the chart.
        group = group.reindex(pd.date_range(group.index.min(), group.index.max(), freq="h"))
        fig, axis = plt.subplots(figsize=(14, 4))
        if facts:
            axis.plot(group.index, group["actual_power"], label="Actual", color="#222222", linewidth=1)
        axis.plot(group.index, group["power"], label="Latest accepted forecast", color="#2277bb", linewidth=1)
        title = f"{turbine}: forecast vs actual" if facts else f"{turbine}: forecast only (actuals unavailable)"
        axis.set(title=title, xlabel="Time (UTC)", ylabel="Normalized power", ylim=(-0.03, 1.03))
        axis.legend()
        axis.grid(alpha=0.2)
        fig.autofmt_xdate()
        fig.tight_layout()
        safe_name = "".join(c if c.isalnum() else "_" for c in turbine)
        fig.savefig(output / f"turbine_{safe_name}.png", dpi=160)
        plt.close(fig)
    coverage = {"forecast_points": len(joined), "matched_actual_points": int(joined.actual_power.notna().sum())}
    print(coverage)
    return coverage


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--facts", help="Optional CSV: turbine_id,valid_time,actual_power")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    plot(args.predictions, args.facts, args.out)
