"""Generate SOF-transfer report artifacts.

The script loads existing CSV files in ``reports/assets`` and computes only
missing replications. It then rebuilds the TeX tables and figures used in the
SOF-transfer sections of the report.
"""

from __future__ import annotations

import argparse
from math import cos, pi, sin
import os
from pathlib import Path
from statistics import NormalDist
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

from representation_aware_sof import (
    DEFAULT_PARALLEL_JOBS,
    PARALLEL_BACKEND,
    SOF_DECOMPOSITION_METHOD_ORDER,
    SOF_PLUGIN_METHOD_ORDER,
    benchmark_a_plugin_curves,
    benchmark_a_plugin_exponents,
    build_cross_benchmark_summary,
    run_benchmark_a_plugin,
    run_benchmark_b_plugin,
    run_benchmark_c_plugin,
    summarize_decomposition,
)


ASSET_DIR = PROJECT_ROOT / "reports" / "assets"
FIGURE_DIR = ASSET_DIR / "figures"


def ensure_dirs() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def load_frame(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_csv(path)


def save_figure(fig: plt.Figure, stem: str) -> None:
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / f"{stem}.png", dpi=220, bbox_inches="tight")
    fig.savefig(FIGURE_DIR / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


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
                values.append(format(value, float_formats[column]))
            else:
                values.append(str(value))
        lines.append(" & ".join(values) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    return "\n".join(lines)


def plot_benchmark_bc_summary(summary: pd.DataFrame) -> None:
    colors = {
        "SOF-raw": "#6c757d",
        "SOF-PCA": "#adb5bd",
        "SOF-LASSO": "#d4a373",
        "SOF-pred": "#90be6d",
        "SOF-dec": "#2a9d8f",
        "Kernel-fixed": "#457b9d",
        "Kernel-dec": "#1d3557",
        "Neural-dec": "#b56576",
    }
    ordered = summary.set_index("method").loc[SOF_PLUGIN_METHOD_ORDER].reset_index()
    bar_colors = [colors[method] for method in ordered["method"]]

    fig, axes = plt.subplots(2, 2, figsize=(15.0, 8.6))
    panels = [
        ("benchmark_b_regret", "Benchmark B: average regret"),
        ("benchmark_b_pred_mse", "Benchmark B: conditional-mean MSE"),
        ("benchmark_c_regret", "Benchmark C: average regret"),
        ("benchmark_c_pred_mse", "Benchmark C: reward-vector MSE"),
    ]
    for ax, (column, title) in zip(axes.flatten(), panels):
        ax.bar(ordered["method"], ordered[column], color=bar_colors)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=30)
        ax.grid(axis="y", alpha=0.25)
    axes[0, 0].set_ylabel("Regret")
    axes[0, 1].set_ylabel("MSE")
    axes[1, 0].set_ylabel("Regret")
    axes[1, 1].set_ylabel("MSE")
    save_figure(fig, "benchmark_bc_plugin_summary")


def plot_benchmark_b_local_neighborhood() -> None:
    highsigma = "#d85a30"
    lowsigma = "#1d9e75"
    querycol = "#534ab7"
    lightcoral = "#faece7"
    lightteal = "#e1f5ee"

    angle = pi / 4.0
    c = cos(angle)
    s = sin(angle)

    x_query = np.array([0.2, 1.4], dtype=float)
    points = pd.DataFrame(
        {
            "index": [1, 2, 3, 4, 5],
            "x1": [-1.0, -1.0, 1.3, 1.5, 3.0],
            "x2": [2.5, 2.2, 0.3, 0.4, -1.0],
            "y": [38.2, 36.8, 33.5, 34.1, 32.9],
        }
    )
    point_coordinates = points[["x1", "x2"]].to_numpy(dtype=float)

    def uv_transform(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        u = c * x[..., 0] + s * x[..., 1]
        v = -s * x[..., 0] + c * x[..., 1]
        return np.stack([u, v], axis=-1)

    uv_query = uv_transform(x_query[None, :])[0]
    uv_points = uv_transform(point_coordinates)
    delta_u = uv_query[0] - uv_points[:, 0]
    delta_v = uv_query[1] - uv_points[:, 1]

    metric_specs = {
        "pred": {
            "title": "(a) Prediction-trained kernel",
            "formula": r"$d_{\mathrm{pred}}^2 = 4\,\Delta u^2 + 0.3\,\Delta v^2$",
            "coefficients": (4.0, 0.3),
            "color": highsigma,
            "expected_d2": np.array([0.813, 0.920, 0.726, 0.974, 4.376]),
            "expected_weights": np.array([0.245, 0.232, 0.256, 0.226, 0.041]),
            "label_text": {1: "(.24)", 2: "(.23)", 3: "(.26)", 4: "(.23)", 5: "(.04)"},
        },
        "dec": {
            "title": "(b) Decision-trained kernel",
            "formula": r"$d_{\mathrm{dec}}^2 = 3.5\,\Delta u^2 + 2.5\,\Delta v^2$",
            "coefficients": (3.5, 2.5),
            "color": lowsigma,
            "expected_d2": np.array([6.630, 5.280, 6.050, 6.770, 34.080]),
            "expected_weights": np.array([0.191, 0.375, 0.255, 0.178, 0.000]),
            "label_text": {1: "(.19)", 2: "(.38)", 3: "(.26)", 4: "(.18)", 5: r"($\approx 0$)"},
        },
    }

    def compute_metric_values(coefficients: tuple[float, float]) -> tuple[np.ndarray, np.ndarray]:
        a, b = coefficients
        squared_distance = a * delta_u**2 + b * delta_v**2
        weights = np.exp(-squared_distance / 2.0)
        weights /= weights.sum()
        return squared_distance, weights

    sorted_order = np.argsort(points["y"].to_numpy(dtype=float))
    sorted_y = points["y"].to_numpy(dtype=float)[sorted_order]

    for spec in metric_specs.values():
        squared_distance, weights = compute_metric_values(spec["coefficients"])
        np.testing.assert_allclose(squared_distance, spec["expected_d2"], atol=5e-3)
        np.testing.assert_allclose(weights, spec["expected_weights"], atol=5e-3)
        spec["squared_distance"] = squared_distance
        spec["weights"] = weights
        spec["kernel_values"] = np.exp(-squared_distance / 2.0)
        spec["cdf"] = np.cumsum(weights[sorted_order])

    np.testing.assert_allclose(metric_specs["pred"]["cdf"], np.array([0.041, 0.297, 0.523, 0.755, 1.0]), atol=5e-3)
    np.testing.assert_allclose(metric_specs["dec"]["cdf"], np.array([0.000, 0.255, 0.434, 0.809, 1.0]), atol=5e-3)

    alpha = 0.8
    normal = NormalDist()
    mu_query = 25.0 + 8.0 * np.tanh(2.0 * uv_query[0])
    sigma_query = 2.0 * (0.8 + 1.2 * float(uv_query[1] > 0.0) + 0.2 * abs(uv_query[0]))
    q_star = mu_query + sigma_query * normal.inv_cdf(alpha)
    np.testing.assert_allclose([mu_query, sigma_query, q_star], [32.829, 4.453, 36.576], atol=2e-3)

    def weighted_quantile(cdf_values: np.ndarray) -> float:
        return float(sorted_y[np.flatnonzero(cdf_values >= alpha)[0]])

    q_pred = weighted_quantile(metric_specs["pred"]["cdf"])
    q_dec = weighted_quantile(metric_specs["dec"]["cdf"])
    np.testing.assert_allclose([q_pred, q_dec], [38.2, 36.8], atol=1e-9)

    figure, axes = plt.subplots(
        1,
        3,
        figsize=(13.8, 4.8),
        gridspec_kw={"width_ratios": [1.0, 1.0, 1.15]},
    )

    x_limits = (-1.85, 3.55)
    y_limits = (-1.55, 3.15)
    grid_x1 = np.linspace(x_limits[0], x_limits[1], 500)
    grid_x2 = np.linspace(y_limits[0], y_limits[1], 500)
    x1_mesh, x2_mesh = np.meshgrid(grid_x1, grid_x2)
    grid_uv = uv_transform(np.stack([x1_mesh, x2_mesh], axis=-1))
    grid_u = grid_uv[..., 0]
    grid_v = grid_uv[..., 1]

    point_colors = [highsigma, highsigma, lowsigma, lowsigma, lowsigma]
    label_offsets = {
        1: (-0.18, 0.16, "right", "bottom"),
        2: (-0.18, -0.16, "right", "top"),
        3: (0.20, -0.20, "left", "top"),
        4: (0.22, 0.18, "left", "bottom"),
        5: (0.22, 0.00, "left", "center"),
    }

    for ax, key in zip(axes[:2], ["pred", "dec"]):
        spec = metric_specs[key]
        a, b = spec["coefficients"]
        d2_grid = a * (uv_query[0] - grid_u) ** 2 + b * (uv_query[1] - grid_v) ** 2

        ax.contourf(
            x1_mesh,
            x2_mesh,
            np.where(grid_v > 0.0, 1.0, 0.0),
            levels=[-0.5, 0.5, 1.5],
            colors=[lightteal, lightcoral],
            alpha=0.9,
        )
        ax.contourf(x1_mesh, x2_mesh, d2_grid, levels=[0.0, 1.0], colors=[spec["color"]], alpha=0.08)
        for level, linestyle, linewidth, alpha_level in [
            (1.0, "solid", 1.6, 0.70),
            (3.0, (0, (5, 5)), 1.2, 0.45),
            (6.0, "solid", 0.9, 0.22),
        ]:
            ax.contour(
                x1_mesh,
                x2_mesh,
                d2_grid,
                levels=[level],
                colors=[spec["color"]],
                linewidths=[linewidth],
                linestyles=[linestyle],
                alpha=alpha_level,
            )

        if key == "dec":
            # Auxiliary exact contours through points 2 and 1 make the roughly 2:1
            # kernel-weight gap visible in panel (b): K_2 ≈ 0.071, K_1 ≈ 0.036.
            for level, linestyle, alpha_level in [
                (spec["squared_distance"][1], (0, (2, 2)), 0.60),
                (spec["squared_distance"][0], (0, (1, 2)), 0.55),
            ]:
                ax.contour(
                    x1_mesh,
                    x2_mesh,
                    d2_grid,
                    levels=[level],
                    colors=[spec["color"]],
                    linewidths=[1.0],
                    linestyles=[linestyle],
                    alpha=alpha_level,
                )

        boundary_min = max(x_limits[0], y_limits[0])
        boundary_max = min(x_limits[1], y_limits[1])
        boundary_points = np.linspace(boundary_min, boundary_max, 200)
        ax.plot(boundary_points, boundary_points, color="0.55", linestyle=(0, (5, 5)), linewidth=1.0)
        ax.text(2.25, 2.05, r"$v=0$", color="0.45", fontsize=10, rotation=45)
        ax.text(-0.35, 2.82, r"high $\sigma$", color=highsigma, fontsize=10)
        ax.text(2.60, -1.18, r"low $\sigma$", color=lowsigma, fontsize=10, ha="right")

        ax.axhline(0.0, color="0.55", linewidth=0.7)
        ax.axvline(0.0, color="0.55", linewidth=0.7)
        point_sizes = 30.0 + 280.0 * spec["weights"]
        ax.scatter(point_coordinates[:, 0], point_coordinates[:, 1], s=point_sizes, color=point_colors, edgecolors="white", linewidths=0.5, zorder=3)
        ax.scatter([x_query[0]], [x_query[1]], s=64, color=querycol, edgecolors="white", linewidths=0.5, zorder=4)
        ax.text(x_query[0] + 0.10, x_query[1] + 0.10, r"$x_0$", color=querycol, fontsize=12, weight="bold")

        for idx, x1, x2 in points[["index", "x1", "x2"]].itertuples(index=False, name=None):
            dx, dy, ha, va = label_offsets[int(idx)]
            ax.text(
                x1 + dx,
                x2 + dy,
                rf"{idx} {spec['label_text'][int(idx)]}",
                color="0.35",
                fontsize=10,
                ha=ha,
                va=va,
            )

        ax.set_xlim(*x_limits)
        ax.set_ylim(*y_limits)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xticks([-1, 0, 1, 2, 3])
        ax.set_yticks([-1, 0, 1, 2, 3])
        ax.tick_params(colors="0.45", labelsize=9)
        for spine_name in ("top", "right", "left", "bottom"):
            ax.spines[spine_name].set_visible(False)
        ax.text(x_limits[1] - 0.02, -0.14, r"$x_1$", color="0.40", fontsize=11, ha="right", va="top")
        ax.text(x_limits[0] + 0.32, y_limits[1] + 0.02, r"$x_2$", color="0.40", fontsize=11, ha="center", va="bottom")
        ax.set_title(spec["title"], fontsize=16, weight="bold", color="0.15", pad=24)
        ax.text(0.5, 1.01, spec["formula"], transform=ax.transAxes, ha="center", va="bottom", fontsize=14, color="0.20")

    cdf_ax = axes[2]
    y_axis = np.linspace(32.4, 38.8, 500)
    gaussian_cdf = np.array([NormalDist(mu=mu_query, sigma=sigma_query).cdf(value) for value in y_axis])
    cdf_ax.plot(y_axis, gaussian_cdf, color="0.70", linewidth=2.0, label=r"true $F_{Y \mid x_0}$")

    def step_series(cdf_values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return np.r_[32.4, sorted_y, 38.8], np.r_[0.0, cdf_values, 1.0]

    pred_x, pred_y = step_series(metric_specs["pred"]["cdf"])
    dec_x, dec_y = step_series(metric_specs["dec"]["cdf"])
    cdf_ax.step(pred_x, pred_y, where="post", color=highsigma, linewidth=2.1, label="pred. kernel")
    cdf_ax.step(dec_x, dec_y, where="post", color=lowsigma, linewidth=2.1, label="dec. kernel")
    cdf_ax.axhline(alpha, color=querycol, linewidth=1.1, linestyle=(0, (4, 4)))
    cdf_ax.text(32.48, alpha + 0.02, r"$\alpha = 0.8$", color=querycol, fontsize=10, va="bottom")
    cdf_ax.vlines(q_star, 0.0, 1.03, color=querycol, linewidth=1.0, linestyles=(0, (1, 3)))
    cdf_ax.text(q_star, 1.04, r"$q^*$", color=querycol, fontsize=10, ha="center", va="bottom")
    cdf_ax.vlines(q_pred, 0.0, alpha, color=highsigma, linewidth=1.1, linestyles=(0, (1, 2)))
    cdf_ax.vlines(q_dec, 0.0, alpha, color=lowsigma, linewidth=1.1, linestyles=(0, (1, 2)))
    cdf_ax.text(q_pred, -0.07, r"$\hat q_{\mathrm{pred}}=38.2$", color=highsigma, fontsize=10, ha="center", va="top", transform=cdf_ax.get_xaxis_transform())
    cdf_ax.text(q_dec, -0.07, r"$\hat q_{\mathrm{dec}}=36.8$", color=lowsigma, fontsize=10, ha="center", va="top", transform=cdf_ax.get_xaxis_transform())
    cdf_ax.set_xlim(32.4, 38.8)
    cdf_ax.set_ylim(0.0, 1.08)
    cdf_ax.set_xticks([33, 34, 35, 36, 37, 38])
    cdf_ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    cdf_ax.grid(alpha=0.18)
    cdf_ax.tick_params(colors="0.45", labelsize=9)
    cdf_ax.set_title("(c) Weighted empirical CDF", fontsize=11, weight="bold", color="0.15")
    cdf_ax.set_xlabel(r"$y$", fontsize=11, color="0.40")
    cdf_ax.set_ylabel("CDF", fontsize=11, color="0.40")
    cdf_ax.spines["top"].set_visible(False)
    cdf_ax.spines["right"].set_visible(False)
    cdf_ax.legend(frameon=False, fontsize=9, loc="lower right")

    save_figure(figure, "benchmark_b_local_neighborhood")


def plot_benchmark_c_boundary(boundary_curve: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    ax.plot(
        boundary_curve["u_center"],
        boundary_curve["oracle_choose_project_0"],
        label="Oracle",
        color="#264653",
        linewidth=2.3,
    )
    ax.plot(
        boundary_curve["u_center"],
        boundary_curve["kernel_fixed_choose_project_0"],
        label="Kernel-fixed",
        color="#457b9d",
        linewidth=2.0,
    )
    ax.plot(
        boundary_curve["u_center"],
        boundary_curve["kernel_dec_choose_project_0"],
        label="Kernel-dec",
        color="#1d3557",
        linewidth=2.0,
    )
    ax.plot(
        boundary_curve["u_center"],
        boundary_curve["sof_raw_choose_project_0"],
        label="SOF-raw",
        color="#6c757d",
        linewidth=2.0,
    )
    ax.plot(
        boundary_curve["u_center"],
        boundary_curve["sof_dec_choose_project_0"],
        label="SOF-dec",
        color="#2a9d8f",
        linewidth=2.0,
    )
    ax.plot(
        boundary_curve["u_center"],
        boundary_curve["neural_dec_choose_project_0"],
        label="Neural-dec",
        color="#b56576",
        linewidth=2.0,
    )
    ax.set_xlabel("Latent boundary coordinate $u$")
    ax.set_ylabel(r"$\Pr(\mathrm{choose\ project\ 0})$")
    ax.set_title("Benchmark C: decision-boundary slice")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=9)
    save_figure(fig, "benchmark_c_plugin_boundary")


def plot_benchmark_a_sof(curves: pd.DataFrame) -> None:
    styles = {
        "SOF-raw": ("#6c757d", "--"),
        "SOF-LASSO": ("#d4a373", "-."),
        "SOF-pred": ("#90be6d", ":"),
        "SOF-dec": ("#2a9d8f", "-"),
    }
    fig, axes = plt.subplots(1, 3, figsize=(14.4, 4.2), sharey=True)
    for ax, d in zip(axes, [20, 50, 100]):
        subset = curves[(curves["d"] == d) & (curves["k"] == 3)]
        for method in ["SOF-raw", "SOF-LASSO", "SOF-pred", "SOF-dec"]:
            method_subset = subset[subset["method"] == method]
            color, linestyle = styles[method]
            ax.plot(
                method_subset["n_train"],
                method_subset["regret"],
                marker="o",
                linestyle=linestyle,
                color=color,
                label=method,
            )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Training size $N$")
        ax.set_title(f"Benchmark A: $(d,k)=({d},3)$")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Average regret")
    axes[-1].legend(frameon=False, fontsize=8, loc="upper right")
    save_figure(fig, "benchmark_a_plugin_sof_loglog")


def plot_decomposition(summary: pd.DataFrame) -> None:
    colors = {
        "SOF-raw": "#6c757d",
        "SOF-raw+kernel": "#8d99ae",
        "SOF-dec": "#2a9d8f",
        "SOF-dec+kernel": "#52b788",
        "Kernel-dec": "#1d3557",
    }
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.2), sharey=True)
    for ax, benchmark in zip(axes, ["B", "C"]):
        subset = (
            summary[summary["benchmark"] == benchmark]
            .set_index("method")
            .loc[SOF_DECOMPOSITION_METHOD_ORDER]
            .reset_index()
        )
        ax.bar(subset["method"], subset["regret"], color=[colors[method] for method in subset["method"]])
        ax.set_title(f"Benchmark {benchmark}: regret decomposition")
        ax.set_ylabel("Average regret")
        ax.tick_params(axis="x", rotation=28)
        ax.grid(axis="y", alpha=0.25)
    save_figure(fig, "benchmark_bc_decomposition")


def build_benchmark_a_exponent_table(exponents: pd.DataFrame) -> pd.DataFrame:
    return (
        exponents.pivot(index=["d", "k"], columns="method", values="effective_exponent")
        .reset_index()
        .rename(
            columns={
                "SOF-raw": "sof_raw_exponent",
                "SOF-LASSO": "sof_lasso_exponent",
                "SOF-pred": "sof_pred_exponent",
                "SOF-dec": "sof_dec_exponent",
            }
        )
        .sort_values(["d", "k"])
    )


def parse_benchmarks(specification: str) -> set[str]:
    value = specification.strip().lower()
    if value == "all":
        return {"A", "B", "C"}
    selected = {token.strip().upper() for token in specification.split(",") if token.strip()}
    invalid = selected - {"A", "B", "C"}
    if invalid:
        invalid_text = ", ".join(sorted(invalid))
        raise ValueError(f"Unknown benchmark selection: {invalid_text}")
    if not selected:
        raise ValueError("At least one benchmark must be selected")
    return selected


def write_benchmark_a_assets(frame_a: pd.DataFrame) -> None:
    exponents_a = benchmark_a_plugin_exponents(frame_a)
    curves_a = benchmark_a_plugin_curves(frame_a)
    frame_a.to_csv(ASSET_DIR / "benchmark_a_plugin_sof_runs.csv", index=False)
    exponents_a.to_csv(ASSET_DIR / "benchmark_a_plugin_sof_exponents.csv", index=False)
    curves_a.to_csv(ASSET_DIR / "benchmark_a_plugin_sof_curves.csv", index=False)

    exponent_table = build_benchmark_a_exponent_table(exponents_a)
    (ASSET_DIR / "benchmark_a_plugin_sof_exponent_table.tex").write_text(
        latex_table(
            exponent_table,
            columns=["d", "k", "sof_raw_exponent", "sof_lasso_exponent", "sof_pred_exponent", "sof_dec_exponent"],
            headers=["$d$", "$k$", "SOF-raw", "SOF-LASSO", "SOF-pred", "SOF-dec"],
            float_formats={
                "sof_raw_exponent": ".2f",
                "sof_lasso_exponent": ".2f",
                "sof_pred_exponent": ".2f",
                "sof_dec_exponent": ".2f",
            },
        ),
        encoding="utf-8",
    )
    plot_benchmark_a_sof(curves_a)
    print("[artifacts] saved Benchmark A assets", flush=True)


def write_benchmark_b_assets(frame_b: pd.DataFrame, decomp_b: pd.DataFrame) -> None:
    frame_b.to_csv(ASSET_DIR / "benchmark_b_plugin_runs.csv", index=False)
    decomp_b.to_csv(ASSET_DIR / "benchmark_b_decomposition_runs.csv", index=False)
    print("[artifacts] saved Benchmark B assets", flush=True)


def write_benchmark_c_assets(frame_c: pd.DataFrame, decomp_c: pd.DataFrame, boundary_curve: pd.DataFrame) -> None:
    frame_c.to_csv(ASSET_DIR / "benchmark_c_plugin_runs.csv", index=False)
    decomp_c.to_csv(ASSET_DIR / "benchmark_c_decomposition_runs.csv", index=False)
    boundary_curve.to_csv(ASSET_DIR / "benchmark_c_plugin_boundary_curve.csv", index=False)
    if not boundary_curve.empty:
        plot_benchmark_c_boundary(boundary_curve)
    print("[artifacts] saved Benchmark C assets", flush=True)


def write_cross_benchmark_assets(
    frame_b: pd.DataFrame | None,
    frame_c: pd.DataFrame | None,
    decomp_b: pd.DataFrame | None,
    decomp_c: pd.DataFrame | None,
) -> None:
    if frame_b is not None and not frame_b.empty and frame_c is not None and not frame_c.empty:
        benchmark_bc_summary = build_cross_benchmark_summary(frame_b, frame_c)
        benchmark_bc_summary.to_csv(ASSET_DIR / "benchmark_bc_plugin_summary.csv", index=False)
        (ASSET_DIR / "benchmark_bc_plugin_summary_table.tex").write_text(
            latex_table(
                benchmark_bc_summary,
                columns=[
                    "method",
                    "benchmark_b_regret",
                    "benchmark_b_pred_mse",
                    "benchmark_b_win_count",
                    "benchmark_c_regret",
                    "benchmark_c_pred_mse",
                    "benchmark_c_win_count",
                ],
                headers=["Method", "B regret", "B MSE", "B wins", "C regret", "C MSE", "C wins"],
                float_formats={
                    "benchmark_b_regret": ".3f",
                    "benchmark_b_pred_mse": ".3f",
                    "benchmark_c_regret": ".3f",
                    "benchmark_c_pred_mse": ".3f",
                },
            ),
            encoding="utf-8",
        )
        plot_benchmark_bc_summary(benchmark_bc_summary)
        print("[artifacts] saved Benchmark B/C summary assets", flush=True)

    if decomp_b is not None and not decomp_b.empty and decomp_c is not None and not decomp_c.empty:
        decomposition_summary = summarize_decomposition(pd.concat([decomp_b, decomp_c], ignore_index=True))
        decomposition_summary.to_csv(ASSET_DIR / "benchmark_bc_decomposition_summary.csv", index=False)
        plot_decomposition(decomposition_summary)
        print("[artifacts] saved decomposition assets", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate report assets for representation-aware SOF experiments.")
    parser.add_argument(
        "--benchmarks",
        default="all",
        help="Comma-separated subset of benchmarks to run: A, B, C, or all.",
    )
    args = parser.parse_args()
    selected = parse_benchmarks(args.benchmarks)

    ensure_dirs()
    print("[artifacts] loading caches", flush=True)
    print(f"[artifacts] selected benchmarks={','.join(sorted(selected))}", flush=True)
    print(f"[artifacts] parallel backend={PARALLEL_BACKEND}, n_jobs={DEFAULT_PARALLEL_JOBS}", flush=True)

    existing_main_a = load_frame(ASSET_DIR / "benchmark_a_runs.csv")
    frame_a = load_frame(ASSET_DIR / "benchmark_a_plugin_sof_runs.csv")
    frame_b = load_frame(ASSET_DIR / "benchmark_b_plugin_runs.csv")
    decomp_b = load_frame(ASSET_DIR / "benchmark_b_decomposition_runs.csv")
    frame_c = load_frame(ASSET_DIR / "benchmark_c_plugin_runs.csv")
    decomp_c = load_frame(ASSET_DIR / "benchmark_c_decomposition_runs.csv")
    boundary_curve = load_frame(ASSET_DIR / "benchmark_c_plugin_boundary_curve.csv")

    if "A" in selected:
        print("[artifacts] running Benchmark A plugin SOF experiments", flush=True)
        frame_a = run_benchmark_a_plugin(existing_plugin=frame_a, existing_main=existing_main_a)
        write_benchmark_a_assets(frame_a)

    if "B" in selected:
        print("[artifacts] running Benchmark B plugin SOF experiments", flush=True)
        frame_b, decomp_b = run_benchmark_b_plugin(
            existing_summary=frame_b,
            existing_decomposition=decomp_b,
        )
        write_benchmark_b_assets(frame_b, decomp_b)

    if "C" in selected:
        print("[artifacts] running Benchmark C plugin SOF experiments", flush=True)
        frame_c, decomp_c, boundary_curve = run_benchmark_c_plugin(
            existing_summary=frame_c,
            existing_decomposition=decomp_c,
            existing_boundary=boundary_curve,
        )
        write_benchmark_c_assets(frame_c, decomp_c, boundary_curve)

    print("[artifacts] building derived tables and figures", flush=True)
    plot_benchmark_b_local_neighborhood()
    if frame_a is not None and not frame_a.empty:
        write_benchmark_a_assets(frame_a)
    if frame_c is not None and boundary_curve is not None and not boundary_curve.empty:
        plot_benchmark_c_boundary(boundary_curve)
    write_cross_benchmark_assets(frame_b, frame_c, decomp_b, decomp_c)
    print("[artifacts] done", flush=True)


if __name__ == "__main__":
    main()
