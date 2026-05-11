"""Cost-dependent embedding experiment (Benchmark D).

Constructs a deliberately cost-sensitive newsvendor DGP in which the
prediction-relevant direction and the tail-relevant direction are different.
The conditional mean depends on the first coordinate only, but the conditional
standard deviation depends on the second coordinate.  Consequently:

  * a prediction-trained metric should ignore the second coordinate at any
    service level;
  * a decision-trained metric should ignore the second coordinate when
    alpha = 0.5 (median = mean for symmetric noise) but increasingly emphasise
    it as alpha moves into the tail.

Per-alpha we re-learn a diagonal Mahalanobis metric and report the share of
metric mass placed on the variance-relevant coordinate.

Produces:
  reports/assets/service_level_runs.csv
  reports/assets/service_level_summary.csv
  reports/assets/service_level_summary_table.tex
  reports/assets/figures/service_level_summary.{pdf,png}
  reports/assets/figures/service_level_embedding.{pdf,png}
"""

from __future__ import annotations

import sys
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd
from scipy.optimize import minimize

sys.dont_write_bytecode = True

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from experiment_suite import (  # noqa: E402
    bandwidth_candidates,
    kernel_average_from_squared_distances,
    orders_from_squared_distances,
    pairwise_squared_distances,
    sorted_response,
    standardize,
)

SERVICE_LEVELS = (0.50, 0.70, 0.85, 0.90, 0.95, 0.99)
REPLICATIONS = 20
N_TRAIN = 250
N_VAL = 150
N_TEST = 2000

# Cost-dependent DGP parameters: y = beta * x1 + exp(gamma * x2) * eps.
# Calibrated so that (i) the mean signal beta * x1 dominates predictive MSE for
# any alpha, ensuring the prediction-trained metric ignores x2 universally, and
# (ii) the variance signal exp(gamma * x2) grows large enough that the optimal
# upper-tail order quantity at alpha close to 1 is materially influenced by x2.
BETA_MEAN = 3.0
GAMMA_LOG_SIGMA = 1.2


def generate_benchmark_d(rng: np.random.Generator, n: int):
    x = rng.normal(size=(n, 2))
    mu = BETA_MEAN * x[:, 0]
    sigma = np.exp(GAMMA_LOG_SIGMA * x[:, 1])
    y = mu + sigma * rng.normal(size=n)
    return x, y, mu, sigma


def diagonal_transform(x: np.ndarray, raw_log_scales: np.ndarray) -> np.ndarray:
    centered = raw_log_scales - raw_log_scales.mean()
    return x * np.sqrt(np.exp(centered))


def diagonal_scales(raw_log_scales: np.ndarray) -> np.ndarray:
    centered = raw_log_scales - raw_log_scales.mean()
    return np.exp(centered)


def learn_diagonal_metric(
    x_train: np.ndarray,
    y_train: np.ndarray,
    *,
    objective_kind: str,
    alpha: float,
    fit_size: int = 200,
    reg: float = 0.002,
    maxfev: int = 500,
) -> np.ndarray:
    fit_size = min(len(x_train), fit_size)
    x_metric = x_train[:fit_size]
    y_metric = y_train[:fit_size]
    y_sorted, y_order = sorted_response(y_metric)

    def objective(raw_log_scales: np.ndarray) -> float:
        transformed = diagonal_transform(x_metric, raw_log_scales)
        squared_distances = pairwise_squared_distances(transformed, transformed)
        if objective_kind == "decision":
            loo_orders = orders_from_squared_distances(
                squared_distances,
                y_sorted,
                y_order,
                alpha,
                bandwidth=1.0,
                exclude_self=True,
            )
            cu = alpha
            co = 1.0 - alpha
            shortage = np.maximum(y_metric - loo_orders, 0.0)
            overage = np.maximum(loo_orders - y_metric, 0.0)
            loss = float((cu * shortage + co * overage).mean())
        elif objective_kind == "prediction":
            loo_prediction = kernel_average_from_squared_distances(
                squared_distances,
                y_metric,
                bandwidth=1.0,
                exclude_self=True,
            )
            loss = float(np.mean((loo_prediction - y_metric) ** 2))
        else:
            raise ValueError(objective_kind)

        centered = raw_log_scales - raw_log_scales.mean()
        penalty = float(np.mean(centered ** 2))
        return loss + reg * penalty

    # Coarse grid search over (s1_log, s2_log) plus local refinement.
    grid_values = np.linspace(-3.0, 3.0, 7)
    starts: list[np.ndarray] = []
    for s1 in grid_values:
        for s2 in grid_values:
            starts.append(np.array([s1, s2]))

    best_result = None
    for start in starts:
        result = minimize(
            objective,
            start,
            method="Powell",
            options={"maxfev": maxfev, "xtol": 1e-4, "ftol": 1e-4},
        )
        if best_result is None or result.fun < best_result.fun:
            best_result = result
    return best_result.x


def tune_bandwidth(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    *,
    alpha: float,
) -> float:
    y_sorted, y_order = sorted_response(y_train)
    candidates = bandwidth_candidates(x_train)
    best_bw = float(candidates[0])
    best_cost = float("inf")
    cu = alpha
    co = 1.0 - alpha
    val_train_sq = pairwise_squared_distances(x_val, x_train)
    for bw in candidates:
        orders = orders_from_squared_distances(
            val_train_sq,
            y_sorted,
            y_order,
            alpha,
            bandwidth=float(bw),
            exclude_self=False,
        )
        shortage = np.maximum(y_val - orders, 0.0)
        overage = np.maximum(orders - y_val, 0.0)
        cost = float((cu * shortage + co * overage).mean())
        if cost < best_cost:
            best_cost = cost
            best_bw = float(bw)
    return best_bw


def evaluate_regret(
    x_train_metric: np.ndarray,
    y_train: np.ndarray,
    x_val_metric: np.ndarray,
    y_val: np.ndarray,
    x_test_metric: np.ndarray,
    y_test: np.ndarray,
    oracle_orders: np.ndarray,
    *,
    alpha: float,
) -> tuple[float, float]:
    bw = tune_bandwidth(x_train_metric, y_train, x_val_metric, y_val, alpha=alpha)
    y_sorted, y_order = sorted_response(y_train)
    test_train_sq = pairwise_squared_distances(x_test_metric, x_train_metric)
    orders = orders_from_squared_distances(
        test_train_sq,
        y_sorted,
        y_order,
        alpha,
        bandwidth=bw,
        exclude_self=False,
    )
    cu = alpha
    co = 1.0 - alpha
    shortage = np.maximum(y_test - orders, 0.0)
    overage = np.maximum(orders - y_test, 0.0)
    realized = float((cu * shortage + co * overage).mean())
    shortage_o = np.maximum(y_test - oracle_orders, 0.0)
    overage_o = np.maximum(oracle_orders - y_test, 0.0)
    oracle = float((cu * shortage_o + co * overage_o).mean())
    return realized - oracle, bw


def x2_share(raw_log_scales: np.ndarray) -> float:
    scales = diagonal_scales(raw_log_scales)
    return float(scales[1] / (scales[0] + scales[1]))


def run() -> None:
    rows: list[dict] = []

    for replication in range(REPLICATIONS):
        seed = 30000 + replication
        rng = np.random.default_rng(seed)
        x_train, y_train, _, _ = generate_benchmark_d(rng, N_TRAIN)
        x_val, y_val, _, _ = generate_benchmark_d(rng, N_VAL)
        x_test, y_test, mu_test, sigma_test = generate_benchmark_d(rng, N_TEST)

        x_train_std, x_val_std = standardize(x_train, x_val)
        _, x_test_std = standardize(x_train, x_test)

        pred_params = learn_diagonal_metric(
            x_train_std,
            y_train,
            objective_kind="prediction",
            alpha=0.5,
        )
        pred_share = x2_share(pred_params)

        for alpha in SERVICE_LEVELS:
            oracle_orders = mu_test + sigma_test * NormalDist().inv_cdf(alpha)

            dec_params = learn_diagonal_metric(
                x_train_std,
                y_train,
                objective_kind="decision",
                alpha=alpha,
            )
            dec_share = x2_share(dec_params)

            pred_train = diagonal_transform(x_train_std, pred_params)
            pred_val = diagonal_transform(x_val_std, pred_params)
            pred_test = diagonal_transform(x_test_std, pred_params)
            pred_regret, pred_bw = evaluate_regret(
                pred_train, y_train, pred_val, y_val, pred_test, y_test, oracle_orders, alpha=alpha,
            )

            dec_train = diagonal_transform(x_train_std, dec_params)
            dec_val = diagonal_transform(x_val_std, dec_params)
            dec_test = diagonal_transform(x_test_std, dec_params)
            dec_regret, dec_bw = evaluate_regret(
                dec_train, y_train, dec_val, y_val, dec_test, y_test, oracle_orders, alpha=alpha,
            )

            row = {
                "replication": replication,
                "alpha": alpha,
                "pred_scale_x1": diagonal_scales(pred_params)[0],
                "pred_scale_x2": diagonal_scales(pred_params)[1],
                "pred_x2_share": pred_share,
                "pred_regret": pred_regret,
                "pred_bandwidth": pred_bw,
                "dec_scale_x1": diagonal_scales(dec_params)[0],
                "dec_scale_x2": diagonal_scales(dec_params)[1],
                "dec_x2_share": dec_share,
                "dec_regret": dec_regret,
                "dec_bandwidth": dec_bw,
            }
            rows.append(row)
            print(
                f"rep={replication:02d} alpha={alpha:.2f}  "
                f"pred_x2_share={pred_share:.3f}  dec_x2_share={dec_share:.3f}  "
                f"pred_regret={pred_regret:.4f}  dec_regret={dec_regret:.4f}",
                flush=True,
            )

    df = pd.DataFrame(rows)
    out_dir = ROOT / "reports" / "assets"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "service_level_runs.csv", index=False)

    summary = (
        df.groupby("alpha")
        .agg(
            pred_x2_share_mean=("pred_x2_share", "mean"),
            pred_x2_share_std=("pred_x2_share", "std"),
            pred_x2_share_median=("pred_x2_share", "median"),
            dec_x2_share_mean=("dec_x2_share", "mean"),
            dec_x2_share_std=("dec_x2_share", "std"),
            dec_x2_share_median=("dec_x2_share", "median"),
            pred_regret_mean=("pred_regret", "mean"),
            pred_regret_std=("pred_regret", "std"),
            dec_regret_mean=("dec_regret", "mean"),
            dec_regret_std=("dec_regret", "std"),
        )
        .reset_index()
    )
    summary.to_csv(out_dir / "service_level_summary.csv", index=False)

    write_summary_table(summary, out_dir / "service_level_summary_table.tex")
    plot_summary(df, summary, out_dir / "figures")


def write_summary_table(summary: pd.DataFrame, path: Path) -> None:
    # The prediction-trained metric is alpha-invariant, so we report it once.
    lines = []
    lines.append("\\begin{tabular}{cccccc}")
    lines.append("\\toprule")
    lines.append(
        "$\\alpha$ & median pred.\\ $x_2$-share & median dec.\\ $x_2$-share & "
        "mean dec.\\ $x_2$-share & pred.\\ regret & dec.\\ regret \\\\"
    )
    lines.append("\\midrule")
    for _, row in summary.iterrows():
        lines.append(
            f"{row['alpha']:.2f} & {row['pred_x2_share_median']:.3f} & "
            f"{row['dec_x2_share_median']:.3f} & {row['dec_x2_share_mean']:.3f} & "
            f"{row['pred_regret_mean']:.3f} & {row['dec_regret_mean']:.3f} \\\\"
        )
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    path.write_text("\n".join(lines))


def plot_summary(runs: pd.DataFrame, summary: pd.DataFrame, fig_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir.mkdir(parents=True, exist_ok=True)
    alphas = sorted(runs["alpha"].unique())

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))

    ax = axes[0]
    # Use median for embedding mass share — the prediction-trained metric is
    # bimodal across replications (variance-reduction trap on a minority of
    # seeds), so the median is a more faithful summary of "typical" behaviour.
    pred_med = [runs[runs["alpha"] == a]["pred_x2_share"].median() for a in alphas]
    pred_q25 = [runs[runs["alpha"] == a]["pred_x2_share"].quantile(0.25) for a in alphas]
    pred_q75 = [runs[runs["alpha"] == a]["pred_x2_share"].quantile(0.75) for a in alphas]
    dec_med = [runs[runs["alpha"] == a]["dec_x2_share"].median() for a in alphas]
    dec_q25 = [runs[runs["alpha"] == a]["dec_x2_share"].quantile(0.25) for a in alphas]
    dec_q75 = [runs[runs["alpha"] == a]["dec_x2_share"].quantile(0.75) for a in alphas]
    import numpy as _np
    ax.plot(alphas, pred_med, marker="o", color="#534AB7",
            label="Prediction-trained embedding (median)", linewidth=2)
    ax.fill_between(alphas, pred_q25, pred_q75, color="#534AB7", alpha=0.15)
    ax.plot(alphas, dec_med, marker="s", color="#D85A30",
            label="Decision-trained embedding (median)", linewidth=2)
    ax.fill_between(alphas, dec_q25, dec_q75, color="#D85A30", alpha=0.15)
    ax.axhline(0.5, color="gray", linestyle=":", linewidth=0.8, alpha=0.7)
    ax.set_xlabel("Service level $\\alpha$")
    ax.set_ylabel("Mass share on variance coordinate $x_2$")
    ax.set_title("Decision-trained embedding shifts with the cost function")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    pred_r = [runs[runs["alpha"] == a]["pred_regret"].mean() for a in alphas]
    dec_r = [runs[runs["alpha"] == a]["dec_regret"].mean() for a in alphas]
    pred_r_std = [runs[runs["alpha"] == a]["pred_regret"].std() for a in alphas]
    dec_r_std = [runs[runs["alpha"] == a]["dec_regret"].std() for a in alphas]
    ax.errorbar(
        alphas, pred_r, yerr=pred_r_std, marker="o", color="#534AB7",
        label="Prediction-trained metric", capsize=3, linewidth=2,
    )
    ax.errorbar(
        alphas, dec_r, yerr=dec_r_std, marker="s", color="#D85A30",
        label="Decision-trained metric", capsize=3, linewidth=2,
    )
    ax.set_xlabel("Service level $\\alpha$")
    ax.set_ylabel("Average newsvendor regret")
    ax.set_title("Decision-trained metric tracks the cost function")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(fig_dir / "service_level_summary.pdf")
    fig.savefig(fig_dir / "service_level_summary.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    run()
