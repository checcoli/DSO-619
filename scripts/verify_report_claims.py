"""Check that cached artifacts reproduce the report's headline claims."""

from __future__ import annotations

from pathlib import Path
import sys

sys.dont_write_bytecode = True

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ASSET_DIR = PROJECT_ROOT / "reports" / "assets"


def assert_close(label: str, actual: float, expected: float, tolerance: float = 0.006) -> None:
    if not np.isfinite(actual) or abs(actual - expected) > tolerance:
        raise AssertionError(f"{label}: expected {expected:.3f}, got {actual:.3f}")
    print(f"[ok] {label}: {actual:.3f}", flush=True)


def require_columns(frame: pd.DataFrame, columns: list[str], name: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise AssertionError(f"{name} is missing columns: {', '.join(missing)}")


def load_csv(name: str) -> pd.DataFrame:
    path = ASSET_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}")
    return pd.read_csv(path)


def verify_benchmark_a() -> None:
    frame = load_csv("benchmark_a_runs.csv")
    subset = frame[(frame["d"] == 20) & (frame["k"] == 3) & (frame["n_train"] == 400)]
    require_columns(
        subset,
        [
            "fixed_regret",
            "lasso_regret",
            "prediction_regret",
            "learned_regret",
            "sof_regret",
            "lasso_sof_regret",
            "lasso_signal_share",
            "prediction_signal_mass_share",
            "learned_signal_mass_share",
            "sof_signal_split_share",
            "lasso_sof_signal_split_share",
        ],
        "benchmark_a_runs.csv",
    )
    means = subset.mean(numeric_only=True)
    assert_close("A fixed regret at d=20,k=3,N=400", means["fixed_regret"], 3.539)
    assert_close("A LASSO regret at d=20,k=3,N=400", means["lasso_regret"], 2.571)
    assert_close("A prediction-trained regret at d=20,k=3,N=400", means["prediction_regret"], 1.230)
    assert_close("A decision-trained regret at d=20,k=3,N=400", means["learned_regret"], 1.154)
    assert_close("A SOF regret at d=20,k=3,N=400", means["sof_regret"], 1.113)
    assert_close("A LASSO->SOF regret at d=20,k=3,N=400", means["lasso_sof_regret"], 0.997)
    assert_close("A LASSO signal share", means["lasso_signal_share"], 0.31, tolerance=0.015)
    assert_close("A prediction signal mass", means["prediction_signal_mass_share"], 0.73, tolerance=0.015)
    assert_close("A decision signal mass", means["learned_signal_mass_share"], 0.74, tolerance=0.015)
    assert_close("A SOF signal split", means["sof_signal_split_share"], 0.46, tolerance=0.015)
    assert_close("A LASSO->SOF split", means["lasso_sof_signal_split_share"], 0.59, tolerance=0.015)


def verify_benchmark_b() -> None:
    frame = load_csv("benchmark_b_runs.csv")
    means = frame.mean(numeric_only=True)
    assert_close("B fixed regret", means["fixed_regret"], 0.774)
    assert_close("B LASSO regret", means["lasso_regret"], 0.774)
    assert_close("B prediction-trained regret", means["prediction_regret"], 0.490)
    assert_close("B decision-trained regret", means["learned_regret"], 0.426)
    assert_close("B neural regret", means["neural_regret"], 0.405)
    assert_close("B SOF regret", means["sof_regret"], 0.922)
    assert_close("B LASSO->SOF regret", means["lasso_sof_regret"], 0.923)
    assert_close("B prediction-trained MSE", means["prediction_pred_mse"], 0.746)
    assert_close("B decision-trained MSE", means["learned_pred_mse"], 1.003)


def verify_benchmark_c() -> None:
    frame = load_csv("benchmark_c_runs.csv")
    means = frame.mean(numeric_only=True)
    assert_close("C fixed regret", means["fixed_regret"], 0.193)
    assert_close("C prediction-trained regret", means["prediction_regret"], 0.109)
    assert_close("C decision-trained regret", means["learned_regret"], 0.106)
    assert_close("C spectral regret", means["spectral_regret"], 0.173)
    assert_close("C neural regret", means["neural_regret"], 0.072)
    assert_close("C LASSO->SOF regret", means["lasso_sof_regret"], 0.311)
    assert_close("C neural MSE", means["neural_pred_mse"], 0.696)


def verify_transfer() -> None:
    summary = load_csv("benchmark_bc_plugin_summary.csv").set_index("method")
    assert_close("Transfer B SOF-raw regret", summary.loc["SOF-raw", "benchmark_b_regret"], 1.304)
    assert_close("Transfer B SOF-pred regret", summary.loc["SOF-pred", "benchmark_b_regret"], 0.339)
    assert_close("Transfer B SOF-dec regret", summary.loc["SOF-dec", "benchmark_b_regret"], 0.303)
    assert_close("Transfer B SOF-pred MSE", summary.loc["SOF-pred", "benchmark_b_pred_mse"], 0.687)
    assert_close("Transfer B SOF-dec MSE", summary.loc["SOF-dec", "benchmark_b_pred_mse"], 0.846)
    assert_close("Transfer C SOF-raw regret", summary.loc["SOF-raw", "benchmark_c_regret"], 0.324)
    assert_close("Transfer C SOF-PCA regret", summary.loc["SOF-PCA", "benchmark_c_regret"], 0.058)
    assert_close("Transfer C SOF-pred regret", summary.loc["SOF-pred", "benchmark_c_regret"], 0.082)
    assert_close("Transfer C SOF-dec regret", summary.loc["SOF-dec", "benchmark_c_regret"], 0.096)

    decomp = load_csv("benchmark_bc_decomposition_summary.csv").set_index(["benchmark", "method"])
    assert_close("Decomposition B SOF-raw regret", decomp.loc[("B", "SOF-raw"), "regret"], 1.304)
    assert_close("Decomposition B SOF-dec regret", decomp.loc[("B", "SOF-dec"), "regret"], 0.303)
    assert_close("Decomposition B SOF-dec+kernel regret", decomp.loc[("B", "SOF-dec+kernel"), "regret"], 0.321)
    assert_close("Decomposition C SOF-raw regret", decomp.loc[("C", "SOF-raw"), "regret"], 0.324)
    assert_close("Decomposition C SOF-dec regret", decomp.loc[("C", "SOF-dec"), "regret"], 0.096)
    assert_close("Decomposition C SOF-dec+kernel regret", decomp.loc[("C", "SOF-dec+kernel"), "regret"], 0.083)


def main() -> None:
    verify_benchmark_a()
    verify_benchmark_b()
    verify_benchmark_c()
    verify_transfer()
    print("[verify] all checked report claims match cached artifacts", flush=True)


if __name__ == "__main__":
    main()
