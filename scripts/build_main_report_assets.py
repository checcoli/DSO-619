"""Rebuild the main report tables and figures from cached benchmark CSV files.

This script is intentionally data-driven: it does not rerun expensive
experiments. Use ``scripts/run_main_benchmarks.py`` first if you want to
replace the cached ``benchmark_*_runs.csv`` files with fresh simulations.
"""

from __future__ import annotations

import os
from pathlib import Path
import sys
import tempfile

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "dfkm_matplotlib_cache"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from experiment_suite import (  # noqa: E402
    benchmark_a_exponents,
    benchmark_a_importance_profile,
    benchmark_a_k3_curves,
)

ASSET_DIR = PROJECT_ROOT / "reports" / "assets"
FIGURE_DIR = ASSET_DIR / "figures"


def ensure_dirs() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def latex_table(
    frame: pd.DataFrame,
    columns: list[str],
    headers: list[str],
    float_formats: dict[str, str],
) -> str:
    align = "l" + "r" * (len(columns) - 1)
    lines = [rf"\begin{{tabular}}{{{align}}}", r"\toprule"]
    lines.append(" & ".join(headers) + r" \\")
    lines.append(r"\midrule")
    for _, row in frame.iterrows():
        values = []
        for column in columns:
            value = row[column]
            if column in float_formats:
                values.append(format(float(value), float_formats[column]))
            else:
                values.append(str(value))
        lines.append(" & ".join(values) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    return "\n".join(lines)


def save_figure(fig: plt.Figure, stem: str) -> None:
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / f"{stem}.png", dpi=220, bbox_inches="tight")
    fig.savefig(FIGURE_DIR / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")
    print(f"[main-assets] wrote {path.relative_to(PROJECT_ROOT)}", flush=True)


def load_csv(name: str) -> pd.DataFrame:
    path = ASSET_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run scripts/run_main_benchmarks.py first.")
    return pd.read_csv(path)


def build_benchmark_a_assets(frame: pd.DataFrame) -> None:
    exponents = benchmark_a_exponents(frame)
    exponents.to_csv(ASSET_DIR / "benchmark_a_exponents.csv", index=False)

    exponent_table = (
        exponents.pivot(index=["d", "k"], columns="method", values="effective_exponent")
        .reset_index()
        .rename(
            columns={
                "fixed": "fixed_exponent",
                "pca": "pca_exponent",
                "lasso": "lasso_exponent",
                "prediction": "prediction_exponent",
                "learned": "learned_exponent",
                "sof": "sof_exponent",
                "lasso_sof": "lasso_sof_exponent",
            }
        )
        .sort_values(["d", "k"])
    )
    write_text(
        ASSET_DIR / "benchmark_a_exponent_table.tex",
        latex_table(
            exponent_table,
            columns=[
                "d",
                "k",
                "fixed_exponent",
                "pca_exponent",
                "lasso_exponent",
                "prediction_exponent",
                "learned_exponent",
                "sof_exponent",
                "lasso_sof_exponent",
            ],
            headers=[
                "$d$",
                "$k$",
                "Fixed",
                "PCA",
                "LASSO",
                "Pred.-trained repr.",
                "Decision-trained repr.",
                "SOF",
                "LASSO$\\to$SOF",
            ],
            float_formats={column: ".2f" for column in exponent_table.columns if column not in {"d", "k"}},
        ),
    )

    snapshot = (
        frame[frame["n_train"] == 400]
        .groupby(["d", "k"])
        .agg(
            fixed_regret=("fixed_regret", "mean"),
            pca_regret=("pca_regret", "mean"),
            lasso_regret=("lasso_regret", "mean"),
            prediction_regret=("prediction_regret", "mean"),
            learned_regret=("learned_regret", "mean"),
            sof_regret=("sof_regret", "mean"),
            lasso_sof_regret=("lasso_sof_regret", "mean"),
            lasso_signal_share=("lasso_signal_share", "mean"),
            prediction_signal_mass_share=("prediction_signal_mass_share", "mean"),
            learned_signal_mass_share=("learned_signal_mass_share", "mean"),
            sof_signal_split_share=("sof_signal_split_share", "mean"),
            lasso_sof_signal_split_share=("lasso_sof_signal_split_share", "mean"),
        )
        .reset_index()
        .sort_values(["d", "k"])
    )
    write_text(
        ASSET_DIR / "benchmark_a_snapshot_table.tex",
        latex_table(
            snapshot,
            columns=[
                "d",
                "k",
                "fixed_regret",
                "pca_regret",
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
            headers=[
                "$d$",
                "$k$",
                "Fixed",
                "PCA",
                "LASSO",
                "Pred.-trained repr.",
                "Decision-trained repr.",
                "SOF",
                "LASSO$\\to$SOF",
                "LASSO signal share",
                "Pred. signal mass",
                "Decision signal mass",
                "SOF signal split",
                "LASSO$\\to$SOF split",
            ],
            float_formats={
                **{column: ".3f" for column in snapshot.columns if column.endswith("_regret")},
                **{column: ".2f" for column in snapshot.columns if column.endswith("_share")},
            },
        ),
    )

    curves = benchmark_a_k3_curves(frame)
    styles = {
        "fixed_regret": ("Fixed", "#457b9d", "-"),
        "pca_regret": ("PCA", "#adb5bd", "--"),
        "lasso_regret": ("LASSO", "#d4a373", "-."),
        "prediction_regret": ("Pred.", "#90be6d", ":"),
        "learned_regret": ("Decision", "#2a9d8f", "-"),
        "sof_regret": ("SOF", "#6c757d", "--"),
        "lasso_sof_regret": ("LASSO->SOF", "#7f5539", "-."),
    }
    fig, axes = plt.subplots(1, 3, figsize=(14.4, 4.2), sharey=True)
    for ax, dimension in zip(axes, [5, 20, 100]):
        subset = curves[curves["d"] == dimension]
        for column, (label, color, linestyle) in styles.items():
            ax.plot(subset["n_train"], subset[column], marker="o", color=color, linestyle=linestyle, label=label)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(f"Benchmark A: d={dimension}, k=3")
        ax.set_xlabel("Training size N")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Average regret")
    axes[-1].legend(frameon=False, fontsize=8)
    save_figure(fig, "benchmark_a_loglog_k3")

    importance = benchmark_a_importance_profile(frame)
    importance.to_csv(ASSET_DIR / "benchmark_a_importance_profile.csv", index=False)
    if not importance.empty:
        fig, ax = plt.subplots(figsize=(10.6, 4.2))
        for method, subset in importance.groupby("method"):
            ax.plot(subset["feature_index"], subset["importance_share"], marker="o", linewidth=1.5, label=method)
        ax.axvline(2.5, color="0.45", linestyle=(0, (4, 4)), linewidth=1.0)
        ax.set_xlabel("Feature index")
        ax.set_ylabel("Normalized emphasis")
        ax.set_title("Benchmark A: feature emphasis at d=20, k=3, N=400")
        ax.grid(alpha=0.22)
        ax.legend(frameon=False, fontsize=8, ncol=2)
        save_figure(fig, "benchmark_a_importance")

    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    exponent_subset = exponents[(exponents["k"] == 3) & (exponents["d"].isin([5, 20, 100]))]
    pivot = exponent_subset.pivot(index="method", columns="d", values="effective_exponent")
    pivot.loc[["fixed", "pca", "lasso", "prediction", "learned", "sof", "lasso_sof"]].plot(kind="bar", ax=ax)
    ax.set_ylabel("Effective exponent")
    ax.set_xlabel("")
    ax.set_title("Benchmark A: effective regret exponents, k=3")
    ax.tick_params(axis="x", rotation=25)
    ax.grid(axis="y", alpha=0.25)
    save_figure(fig, "benchmark_a_exponents")


def method_summary(frame: pd.DataFrame, methods: list[tuple[str, str]]) -> pd.DataFrame:
    rows = []
    for label, prefix in methods:
        rows.append(
            {
                "method": label,
                "avg_cost": frame[f"{prefix}_cost"].mean(),
                "avg_regret": frame[f"{prefix}_regret"].mean(),
                "avg_pred_mse": frame[f"{prefix}_pred_mse"].mean(),
            }
        )
    return pd.DataFrame(rows)


def build_benchmark_b_assets(frame: pd.DataFrame) -> None:
    methods = [
        ("Global SAA", "global"),
        ("Point prediction", "point"),
        ("Fixed kernel", "fixed"),
        ("LASSO + kernel", "lasso"),
        ("Pred.-trained repr.", "prediction"),
        ("Decision-trained repr.", "learned"),
        ("Spectral mixture", "spectral"),
        ("Neural embedding", "neural"),
        ("SOF", "sof"),
        ("LASSO$\\to$SOF", "lasso_sof"),
    ]
    summary = method_summary(frame, methods)
    write_text(
        ASSET_DIR / "benchmark_b_summary_table.tex",
        latex_table(
            summary,
            columns=["method", "avg_cost", "avg_regret", "avg_pred_mse"],
            headers=["Method", "Avg. cost", "Avg. regret", "Avg. pred. MSE"],
            float_formats={"avg_cost": ".3f", "avg_regret": ".3f", "avg_pred_mse": ".3f"},
        ),
    )
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    axes[0].bar(summary["method"], summary["avg_regret"], color="#457b9d")
    axes[0].set_title("Benchmark B: average regret")
    axes[0].set_ylabel("Regret")
    axes[1].bar(summary["method"], summary["avg_pred_mse"], color="#2a9d8f")
    axes[1].set_title("Benchmark B: conditional-mean MSE")
    axes[1].set_ylabel("MSE")
    for ax in axes:
        ax.tick_params(axis="x", rotation=35)
        ax.grid(axis="y", alpha=0.25)
    save_figure(fig, "benchmark_b_summary")


def build_benchmark_c_assets(frame: pd.DataFrame, boundary_curve: pd.DataFrame) -> None:
    methods = [
        ("Global SAA", "global"),
        ("Point prediction", "point"),
        ("Fixed kernel", "fixed"),
        ("Pred.-trained repr.", "prediction"),
        ("Decision-trained repr.", "learned"),
        ("Spectral mixture", "spectral"),
        ("Neural embedding", "neural"),
        ("LASSO$\\to$SOF", "lasso_sof"),
    ]
    summary = method_summary(frame, methods)
    write_text(
        ASSET_DIR / "benchmark_c_summary_table.tex",
        latex_table(
            summary,
            columns=["method", "avg_cost", "avg_regret", "avg_pred_mse"],
            headers=["Method", "Avg. cost", "Avg. regret", "Avg. pred. MSE"],
            float_formats={"avg_cost": ".3f", "avg_regret": ".3f", "avg_pred_mse": ".3f"},
        ),
    )
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.2))
    axes[0].bar(summary["method"], summary["avg_regret"], color="#457b9d")
    axes[0].set_title("Benchmark C: average regret")
    axes[0].set_ylabel("Regret")
    axes[1].bar(summary["method"], summary["avg_pred_mse"], color="#2a9d8f")
    axes[1].set_title("Benchmark C: reward-vector MSE")
    axes[1].set_ylabel("MSE")
    if not boundary_curve.empty:
        columns = [
            ("oracle_choose_project_0", "Oracle", "#264653"),
            ("fixed_choose_project_0", "Fixed", "#457b9d"),
            ("learned_choose_project_0", "Decision", "#1d3557"),
            ("neural_choose_project_0", "Neural", "#b56576"),
            ("lasso_sof_choose_project_0", "LASSO->SOF", "#7f5539"),
        ]
        for column, label, color in columns:
            if column in boundary_curve:
                axes[2].plot(boundary_curve["u_center"], boundary_curve[column], label=label, color=color, linewidth=2.0)
        axes[2].set_title("Benchmark C: boundary slice")
        axes[2].set_xlabel("Latent coordinate u")
        axes[2].set_ylabel("Pr(choose project 0)")
        axes[2].set_ylim(-0.05, 1.05)
        axes[2].legend(frameon=False, fontsize=8)
    for ax in axes[:2]:
        ax.tick_params(axis="x", rotation=35)
        ax.grid(axis="y", alpha=0.25)
    axes[2].grid(alpha=0.25)
    save_figure(fig, "benchmark_c_summary")


def main() -> None:
    ensure_dirs()
    build_benchmark_a_assets(load_csv("benchmark_a_runs.csv"))
    build_benchmark_b_assets(load_csv("benchmark_b_runs.csv"))
    boundary_path = ASSET_DIR / "benchmark_c_boundary_curve.csv"
    boundary_curve = pd.read_csv(boundary_path) if boundary_path.exists() else pd.DataFrame()
    build_benchmark_c_assets(load_csv("benchmark_c_runs.csv"), boundary_curve)
    print("[main-assets] done", flush=True)


if __name__ == "__main__":
    main()
