"""Post-process precomputed forecasts across multiple horizons.

This script collects prediction CSV files for several forecast horizons, generates
IEEE-style figures, computes summary statistics, runs a simple demand response
optimization, and exports all intermediate artifacts for reproducibility.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional

import cvxpy as cp
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_absolute_error, r2_score

BLUE = "#1f77b4"  # Actual / baseline
ORANGE = "#ff7f0e"  # Predicted / DR-adjusted
GRAY = "#7f7f7f"


@dataclass
class HorizonArtifacts:
    horizon: int
    dataframe: pd.DataFrame
    adjusted: Optional[pd.Series]
    metrics_pred: Mapping[str, float]
    metrics_adjusted: Optional[Mapping[str, float]]
    dr_summary: Optional[Mapping[str, float]]
    prediction_path: Path
    dr_metadata: Optional[Mapping[str, float]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create IEEE-ready plots, metrics tables, and demand response outputs "
            "for precomputed prediction horizons without re-running training."
        )
    )
    parser.add_argument(
        "--predictions-dir",
        type=Path,
        default=Path("predictions"),
        help="Directory containing horizon-specific CSV prediction files.",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="{horizon}h_ahead.csv",
        help=(
            "Filename pattern for horizon CSV files. Use '{horizon}' or '{h}' placeholders. "
            "Example: 'microgrid_{horizon}h.csv'."
        ),
    )
    parser.add_argument(
        "--horizons",
        type=int,
        nargs="+",
        default=[1, 4, 8, 16, 24],
        help="Forecast horizons (in hours) to process.",
    )
    parser.add_argument(
        "--datetime-column",
        type=str,
        default="datetime",
        help="Name of the timestamp column, if present.",
    )
    parser.add_argument(
        "--actual-column",
        type=str,
        default="Actual",
        help="Column containing ground-truth values.",
    )
    parser.add_argument(
        "--prediction-column",
        type=str,
        default="Predicted",
        help="Column containing model predictions.",
    )
    parser.add_argument(
        "--price-column",
        type=str,
        default="Price",
        help=(
            "Column for energy price used in demand response optimization. "
            "If missing, unit price of 1.0 is assumed."
        ),
    )
    parser.add_argument(
        "--dr-capacity-column",
        type=str,
        default="DR_Capacity",
        help=(
            "Column describing per-interval DR shedding capacity. When not available, "
            "use --dr-capacity to provide a scalar limit."
        ),
    )
    parser.add_argument(
        "--dr-capacity",
        type=float,
        default=None,
        help="Scalar demand response capacity (kW) applied uniformly if column missing.",
    )
    parser.add_argument(
        "--dr-budget",
        type=float,
        default=None,
        help="Optional total kWh shedding budget enforced across the horizon.",
    )
    parser.add_argument(
        "--plots-dir",
        type=Path,
        default=Path("plots"),
        help="Directory for generated figures.",
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=Path("reports"),
        help="Directory for metrics tables.",
    )
    parser.add_argument(
        "--exports-dir",
        type=Path,
        default=Path("exports"),
        help="Directory exporting data for external visualization tools.",
    )
    parser.add_argument(
        "--no-pdf",
        action="store_true",
        help="Skip exporting PDF copies of figures (PNG is always generated).",
    )
    return parser.parse_args()


def set_ieee_style() -> None:
    mpl.rcParams.update(
        {
            "figure.figsize": (7.2, 3.6),
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "font.size": 9,
            "axes.titlesize": 9,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.grid": True,
            "grid.linestyle": ":",
            "grid.linewidth": 0.5,
            "lines.linewidth": 1.2,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    sns.set_style("whitegrid")


def ensure_directories(*dirs: Path) -> None:
    for directory in dirs:
        directory.mkdir(parents=True, exist_ok=True)


def find_prediction_file(base_dir: Path, pattern: str, horizon: int) -> Path:
    formatted = pattern.format(horizon=horizon, h=horizon)
    candidate = Path(formatted)
    if not candidate.is_absolute():
        candidate = base_dir / candidate
    if candidate.exists():
        return candidate
    raise FileNotFoundError(
        f"Could not locate prediction file for horizon {horizon} using pattern '{pattern}'. "
        f"Expected path: {candidate}"
    )


def load_predictions(
    path: Path,
    datetime_column: str,
    actual_column: str,
    prediction_column: str,
) -> pd.DataFrame:
    df = pd.read_csv(path)
    if datetime_column in df.columns:
        df[datetime_column] = pd.to_datetime(df[datetime_column])
        df = df.sort_values(datetime_column)
    if actual_column not in df.columns or prediction_column not in df.columns:
        raise ValueError(
            f"File {path} must contain '{actual_column}' and '{prediction_column}' columns."
        )
    df = df.reset_index(drop=True)
    return df


def compute_metrics(actual: np.ndarray, predicted: np.ndarray) -> Dict[str, float]:
    residuals = actual - predicted
    return {
        "MAE": float(mean_absolute_error(actual, predicted)),
        "RMSE": float(np.sqrt(np.mean(residuals**2))),
        "MAPE": float(np.mean(np.abs(residuals / np.clip(np.abs(actual), 1e-9, None))) * 100.0),
        "R2": float(r2_score(actual, predicted)),
        "Bias": float(np.mean(residuals)),
        "STD_Residual": float(np.std(residuals)),
    }


def run_demand_response(
    df: pd.DataFrame,
    prediction_column: str,
    price_column: str,
    dr_capacity_column: str,
    scalar_capacity: Optional[float],
    dr_budget: Optional[float],
) -> Dict[str, np.ndarray]:
    predicted = df[prediction_column].to_numpy(dtype=float)
    n = predicted.size
    if price_column in df.columns:
        price = df[price_column].fillna(0.0).to_numpy(dtype=float)
    else:
        price = np.ones(n, dtype=float)

    if dr_capacity_column in df.columns:
        capacity = df[dr_capacity_column].fillna(0.0).to_numpy(dtype=float)
    elif scalar_capacity is not None:
        capacity = np.full(n, scalar_capacity, dtype=float)
    else:
        raise ValueError(
            "Demand response requires either a column '{dr_capacity_column}' or a scalar "
            "--dr-capacity value."
        )

    capacity = np.clip(capacity, 0.0, None)

    shed = cp.Variable(n)
    objective = cp.Maximize(cp.sum(cp.multiply(price, shed)))
    constraints = [shed >= 0, shed <= capacity]
    if dr_budget is not None:
        constraints.append(cp.sum(shed) <= dr_budget)
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.ECOS, warm_start=True)

    if problem.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}:
        raise RuntimeError(f"Demand response optimization failed with status {problem.status}.")

    shed_values = np.array(shed.value).reshape(-1)
    adjusted = predicted - shed_values
    adjusted = np.maximum(adjusted, 0.0)

    return {
        "shed": shed_values,
        "adjusted": adjusted,
        "price": price,
        "capacity": capacity,
        "dr_objective": float(problem.value),
    }


def plot_time_series(
    df: pd.DataFrame,
    horizon: int,
    datetime_column: Optional[str],
    actual_column: str,
    prediction_column: str,
    adjusted: Optional[np.ndarray],
    output_dir: Path,
    include_pdf: bool,
) -> None:
    plt.figure()
    x = df[datetime_column] if datetime_column and datetime_column in df.columns else np.arange(len(df))
    plt.plot(x, df[actual_column], label="Actual", color=BLUE)
    plt.plot(x, df[prediction_column], label="Predicted", color=ORANGE, linestyle="--")
    if adjusted is not None:
        plt.plot(x, adjusted, label="DR-Adjusted", color=GRAY, linestyle="-")
    plt.title(f"{horizon}-Hour Ahead Forecast")
    plt.xlabel("Time")
    plt.ylabel("Load")
    plt.legend()
    plt.tight_layout()
    png_path = output_dir / f"{horizon}h_time_series.png"
    plt.savefig(png_path)
    if include_pdf:
        plt.savefig(output_dir / f"{horizon}h_time_series.pdf")
    plt.close()


def plot_parity(
    df: pd.DataFrame,
    horizon: int,
    actual_column: str,
    prediction_column: str,
    adjusted: Optional[np.ndarray],
    output_dir: Path,
    include_pdf: bool,
) -> None:
    plt.figure()
    plt.scatter(df[actual_column], df[prediction_column], label="Predicted", color=ORANGE, alpha=0.6)
    if adjusted is not None:
        plt.scatter(df[actual_column], adjusted, label="DR-Adjusted", color=GRAY, alpha=0.6)
    lims = [min(df[actual_column].min(), df[prediction_column].min()), max(df[actual_column].max(), df[prediction_column].max())]
    plt.plot(lims, lims, color=BLUE, linestyle=":", label="Ideal")
    plt.xlabel("Actual")
    plt.ylabel("Forecast")
    plt.title(f"Parity Plot - {horizon}h Ahead")
    plt.legend()
    plt.tight_layout()
    png_path = output_dir / f"{horizon}h_parity.png"
    plt.savefig(png_path)
    if include_pdf:
        plt.savefig(output_dir / f"{horizon}h_parity.pdf")
    plt.close()


def plot_demand_response(
    df: pd.DataFrame,
    horizon: int,
    datetime_column: Optional[str],
    actual_column: str,
    prediction_column: str,
    dr_results: Optional[Mapping[str, np.ndarray]],
    output_dir: Path,
    include_pdf: bool,
) -> None:
    if dr_results is None:
        return

    x = df[datetime_column] if datetime_column and datetime_column in df.columns else np.arange(len(df))
    plt.figure()
    plt.plot(x, df[prediction_column], label="Predicted", color=ORANGE, linestyle="--")
    plt.plot(x, dr_results["adjusted"], label="DR-Adjusted", color=GRAY)
    plt.fill_between(x, dr_results["adjusted"], df[prediction_column], color=ORANGE, alpha=0.2, label="Shed")
    plt.title(f"Demand Response Optimization - {horizon}h Ahead")
    plt.xlabel("Time")
    plt.ylabel("Load")
    plt.legend()
    plt.tight_layout()
    png_path = output_dir / f"{horizon}h_demand_response.png"
    plt.savefig(png_path)
    if include_pdf:
        plt.savefig(output_dir / f"{horizon}h_demand_response.pdf")
    plt.close()


def aggregate_metrics(artifacts: Iterable[HorizonArtifacts]) -> pd.DataFrame:
    records: List[MutableMapping[str, float]] = []
    for item in artifacts:
        record: MutableMapping[str, float] = {"Horizon": item.horizon, **item.metrics_pred}
        prefix = "DR_"
        if item.metrics_adjusted is not None:
            for key, value in item.metrics_adjusted.items():
                record[f"{prefix}{key}"] = value
        if item.dr_summary is not None:
            for key, value in item.dr_summary.items():
                record[f"DR_{key}"] = value
        records.append(record)
    return pd.DataFrame(records).sort_values("Horizon").reset_index(drop=True)


def export_timeseries(
    df: pd.DataFrame,
    horizon: int,
    adjusted: Optional[np.ndarray],
    dr_results: Optional[Mapping[str, np.ndarray]],
    output_dir: Path,
    datetime_column: str,
) -> None:
    export_df = df.copy()
    if adjusted is not None:
        export_df["DR_Adjusted"] = adjusted
    if dr_results is not None:
        export_df["DR_Shed"] = dr_results["shed"]
        export_df["DR_Capacity"] = dr_results["capacity"]
        export_df["DR_Price"] = dr_results["price"]
    export_path = output_dir / f"{horizon}h_timeseries.csv"
    export_df.to_csv(export_path, index=False)


def export_metadata(
    artifacts: Iterable[HorizonArtifacts],
    output_dir: Path,
) -> None:
    metadata: Dict[str, Mapping[str, float]] = {}
    for item in artifacts:
        entry: Dict[str, float] = {}
        entry.update({k: float(v) for k, v in item.metrics_pred.items()})
        if item.metrics_adjusted is not None:
            entry.update({f"DR_{k}": float(v) for k, v in item.metrics_adjusted.items()})
        if item.dr_summary is not None:
            entry.update({f"DR_{k}": float(v) for k, v in item.dr_summary.items()})
        metadata[str(item.horizon)] = entry
    output_path = output_dir / "summary.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)


def main() -> None:
    args = parse_args()
    set_ieee_style()
    ensure_directories(args.plots_dir, args.reports_dir, args.exports_dir)

    artifacts: List[HorizonArtifacts] = []

    for horizon in args.horizons:
        prediction_path = find_prediction_file(args.predictions_dir, args.pattern, horizon)
        df = load_predictions(
            prediction_path,
            datetime_column=args.datetime_column,
            actual_column=args.actual_column,
            prediction_column=args.prediction_column,
        )

        actual = df[args.actual_column].to_numpy(dtype=float)
        predicted = df[args.prediction_column].to_numpy(dtype=float)
        metrics_pred = compute_metrics(actual, predicted)

        dr_results: Optional[Dict[str, np.ndarray]] = None
        metrics_adjusted: Optional[Mapping[str, float]] = None
        dr_summary: Optional[Mapping[str, float]] = None

        try:
            dr_results = run_demand_response(
                df,
                prediction_column=args.prediction_column,
                price_column=args.price_column,
                dr_capacity_column=args.dr_capacity_column,
                scalar_capacity=args.dr_capacity,
                dr_budget=args.dr_budget,
            )
            adjusted = dr_results["adjusted"]
            metrics_adjusted = compute_metrics(actual, adjusted)
            dr_summary = {
                "Total_Shed": float(np.sum(dr_results["shed"])),
                "Max_Shed": float(np.max(dr_results["shed"])),
                "Objective": float(dr_results["dr_objective"]),
            }
        except ValueError:
            adjusted = None
        else:
            pass

        artifacts.append(
            HorizonArtifacts(
                horizon=horizon,
                dataframe=df,
                adjusted=None if dr_results is None else pd.Series(dr_results["adjusted"]),
                metrics_pred=metrics_pred,
                metrics_adjusted=metrics_adjusted,
                dr_summary=dr_summary,
                prediction_path=prediction_path,
                dr_metadata=None if dr_results is None else {
                    "Total_Capacity": float(np.sum(dr_results["capacity"])),
                    "Total_Price": float(np.sum(dr_results["price"])),
                },
            )
        )

        plot_dir = args.plots_dir / f"{horizon}h"
        ensure_directories(plot_dir)
        plot_time_series(
            df,
            horizon=horizon,
            datetime_column=args.datetime_column,
            actual_column=args.actual_column,
            prediction_column=args.prediction_column,
            adjusted=None if dr_results is None else dr_results["adjusted"],
            output_dir=plot_dir,
            include_pdf=not args.no_pdf,
        )
        plot_parity(
            df,
            horizon=horizon,
            actual_column=args.actual_column,
            prediction_column=args.prediction_column,
            adjusted=None if dr_results is None else dr_results["adjusted"],
            output_dir=plot_dir,
            include_pdf=not args.no_pdf,
        )
        plot_demand_response(
            df,
            horizon=horizon,
            datetime_column=args.datetime_column,
            actual_column=args.actual_column,
            prediction_column=args.prediction_column,
            dr_results=dr_results,
            output_dir=plot_dir,
            include_pdf=not args.no_pdf,
        )

        export_timeseries(
            df,
            horizon=horizon,
            adjusted=None if dr_results is None else dr_results["adjusted"],
            dr_results=dr_results,
            output_dir=args.exports_dir,
            datetime_column=args.datetime_column,
        )

    metrics_table = aggregate_metrics(artifacts)
    metrics_csv_path = args.reports_dir / "metrics_multi_horizon.csv"
    metrics_table.to_csv(metrics_csv_path, index=False)

    export_metadata(artifacts, args.exports_dir)

    print("Processed horizons:", ", ".join(str(h) for h in args.horizons))
    print(f"Metrics table saved to {metrics_csv_path}")
    print(f"Exports available in {args.exports_dir.resolve()}")


if __name__ == "__main__":
    main()
