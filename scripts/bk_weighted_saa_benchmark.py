"""Minimal prescriptive benchmark for weighted SAA and learned kernel weighting.

This script keeps the original Bertsimas-Kallus contextual weighted-SAA benchmark
and adds the project's proposed method: a learned diagonal Mahalanobis metric
trained on leave-one-out prescriptive loss.

Methods compared:
1. Global SAA baseline that ignores context.
2. Point-prediction baseline from linear regression.
3. Fixed Gaussian-kernel weighted SAA.
4. Learned-metric Gaussian-kernel weighted SAA.

The decision problem is a contextual newsvendor, so each prescription reduces to
an empirical quantile of historical demands under different weights.

Run:
    python3 scripts/bk_weighted_saa_benchmark.py
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from statistics import NormalDist

import numpy as np
from scipy.optimize import minimize


@dataclass(frozen=True)
class Config:
    reps: int = 20
    n_train: int = 400
    n_val: int = 200
    n_test: int = 2000
    d: int = 5
    signal_dims: int = 2
    sigma: float = 4.0
    underage_cost: float = 4.0
    overage_cost: float = 1.0
    seed: int = 7
    metric_fit_size: int = 120
    metric_reg: float = 0.02
    metric_maxfev: int = 60


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reps", type=int, default=20)
    parser.add_argument("--n-train", type=int, default=400)
    parser.add_argument("--n-val", type=int, default=200)
    parser.add_argument("--n-test", type=int, default=2000)
    parser.add_argument("--d", type=int, default=5, help="Total context dimension.")
    parser.add_argument(
        "--signal-dims",
        type=int,
        default=2,
        help="Number of context features that truly affect demand.",
    )
    parser.add_argument("--sigma", type=float, default=4.0)
    parser.add_argument("--underage-cost", type=float, default=4.0)
    parser.add_argument("--overage-cost", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--metric-fit-size",
        type=int,
        default=120,
        help="Subset size used to fit the learned metric with leave-one-out loss.",
    )
    parser.add_argument(
        "--metric-reg",
        type=float,
        default=0.02,
        help="L2 penalty on log feature scales for the learned metric.",
    )
    parser.add_argument(
        "--metric-maxfev",
        type=int,
        default=60,
        help="Maximum objective evaluations for Powell search.",
    )
    args = parser.parse_args()

    if args.signal_dims > args.d:
        parser.error("--signal-dims must be <= --d")
    integer_values = [
        args.reps,
        args.n_train,
        args.n_val,
        args.n_test,
        args.d,
        args.signal_dims,
        args.metric_fit_size,
        args.metric_maxfev,
    ]
    if min(integer_values) <= 0:
        parser.error("all integer arguments must be positive")
    if args.sigma <= 0 or args.underage_cost <= 0 or args.overage_cost <= 0:
        parser.error("sigma and cost parameters must be positive")
    if args.metric_reg < 0:
        parser.error("--metric-reg must be nonnegative")

    return Config(
        reps=args.reps,
        n_train=args.n_train,
        n_val=args.n_val,
        n_test=args.n_test,
        d=args.d,
        signal_dims=args.signal_dims,
        sigma=args.sigma,
        underage_cost=args.underage_cost,
        overage_cost=args.overage_cost,
        seed=args.seed,
        metric_fit_size=args.metric_fit_size,
        metric_reg=args.metric_reg,
        metric_maxfev=args.metric_maxfev,
    )


def critical_fractile(underage_cost: float, overage_cost: float) -> float:
    return underage_cost / (underage_cost + overage_cost)


def generate_data(
    rng: np.random.Generator,
    n: int,
    d: int,
    signal_dims: int,
    sigma: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create a contextual demand process with nuisance covariates."""
    x = rng.normal(size=(n, d))

    beta = np.zeros(d)
    magnitudes = np.linspace(6.0, 1.0, signal_dims)
    signs = np.where(np.arange(signal_dims) % 2 == 0, 1.0, -1.0)
    beta[:signal_dims] = signs * magnitudes

    mean_demand = 25.0 + x @ beta
    demand = mean_demand + sigma * rng.normal(size=n)
    return x, demand, mean_demand


def standardize(
    x_train: np.ndarray,
    x_other: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0)
    std[std == 0.0] = 1.0
    return (x_train - mean) / std, (x_other - mean) / std


def newsvendor_cost(
    order_quantity: np.ndarray,
    demand: np.ndarray,
    underage_cost: float,
    overage_cost: float,
) -> np.ndarray:
    shortage = np.maximum(demand - order_quantity, 0.0)
    overage = np.maximum(order_quantity - demand, 0.0)
    return underage_cost * shortage + overage_cost * overage


def sorted_response(y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(y)
    return y[order], order


def weighted_quantile_from_sorted(
    y_sorted: np.ndarray,
    weights_sorted: np.ndarray,
    alpha: float,
) -> float:
    total_weight = weights_sorted.sum()
    if total_weight <= 0.0:
        weights_sorted = np.ones_like(y_sorted, dtype=float)
        total_weight = weights_sorted.sum()

    cumulative = np.cumsum(weights_sorted / total_weight)
    index = int(np.searchsorted(cumulative, alpha, side="left"))
    return float(y_sorted[min(index, len(y_sorted) - 1)])


def weighted_quantile(values: np.ndarray, weights: np.ndarray, alpha: float) -> float:
    y_sorted, order = sorted_response(values)
    return weighted_quantile_from_sorted(y_sorted, weights[order], alpha)


def global_saa_order(y_train: np.ndarray, alpha: float) -> float:
    return weighted_quantile(y_train, np.ones_like(y_train, dtype=float), alpha)


def fit_linear_regression(x_train: np.ndarray, y_train: np.ndarray) -> np.ndarray:
    design = np.column_stack([np.ones(len(x_train)), x_train])
    coefficients, *_ = np.linalg.lstsq(design, y_train, rcond=None)
    return coefficients


def point_prediction_orders(x: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    design = np.column_stack([np.ones(len(x)), x])
    return design @ coefficients


def pairwise_squared_distances(
    x_query: np.ndarray,
    x_train: np.ndarray,
) -> np.ndarray:
    query_norm = np.sum(x_query * x_query, axis=1, keepdims=True)
    train_norm = np.sum(x_train * x_train, axis=1)
    squared = query_norm + train_norm[None, :] - 2.0 * (x_query @ x_train.T)
    return np.maximum(squared, 0.0)


def distance_scale(x_train: np.ndarray) -> float:
    sample = x_train[: min(len(x_train), 200)]
    pairwise = pairwise_squared_distances(sample, sample) ** 0.5
    upper = pairwise[np.triu_indices_from(pairwise, k=1)]
    median_distance = float(np.median(upper))
    if not np.isfinite(median_distance) or median_distance <= 0.0:
        return 1.0
    return median_distance


def orders_from_squared_distances(
    squared_distances: np.ndarray,
    y_sorted: np.ndarray,
    y_order: np.ndarray,
    alpha: float,
    bandwidth: float,
    exclude_self: bool = False,
) -> np.ndarray:
    orders = np.empty(squared_distances.shape[0])
    denominator = 2.0 * bandwidth**2

    for row_index in range(squared_distances.shape[0]):
        weights = np.exp(-squared_distances[row_index] / denominator)
        if exclude_self:
            weights[row_index] = 0.0
        orders[row_index] = weighted_quantile_from_sorted(y_sorted, weights[y_order], alpha)

    return orders


def tune_bandwidth(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    alpha: float,
    underage_cost: float,
    overage_cost: float,
) -> float:
    y_sorted, y_order = sorted_response(y_train)
    scale = distance_scale(x_train)
    candidates = scale * np.array([0.20, 0.35, 0.50, 0.75, 1.00, 1.50, 2.00])
    squared_distances = pairwise_squared_distances(x_val, x_train)

    best_bandwidth = float(candidates[0])
    best_cost = float("inf")
    for bandwidth in candidates:
        candidate_orders = orders_from_squared_distances(
            squared_distances,
            y_sorted,
            y_order,
            alpha,
            float(bandwidth),
        )
        avg_cost = newsvendor_cost(
            candidate_orders,
            y_val,
            underage_cost,
            overage_cost,
        ).mean()
        if avg_cost < best_cost:
            best_cost = float(avg_cost)
            best_bandwidth = float(bandwidth)

    return best_bandwidth


def metric_scales(raw_metric: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    centered = raw_metric - raw_metric.mean()
    scales = np.exp(centered)
    return centered, scales


def transform_with_metric(x: np.ndarray, raw_metric: np.ndarray) -> np.ndarray:
    _, scales = metric_scales(raw_metric)
    return x * np.sqrt(scales)


def learn_diagonal_metric(
    x_train: np.ndarray,
    y_train: np.ndarray,
    alpha: float,
    underage_cost: float,
    overage_cost: float,
    metric_fit_size: int,
    metric_reg: float,
    metric_maxfev: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Learn diagonal Mahalanobis scales from leave-one-out prescriptive loss."""
    fit_size = min(len(x_train), metric_fit_size)
    x_metric = x_train[:fit_size]
    y_metric = y_train[:fit_size]
    y_sorted, y_order = sorted_response(y_metric)

    feature_differences = x_metric[:, None, :] - x_metric[None, :, :]
    squared_differences = feature_differences * feature_differences

    def objective(raw_metric: np.ndarray) -> float:
        centered, scales = metric_scales(raw_metric)
        squared_distances = np.tensordot(squared_differences, scales, axes=([2], [0]))
        loo_orders = orders_from_squared_distances(
            squared_distances,
            y_sorted,
            y_order,
            alpha,
            bandwidth=1.0,
            exclude_self=True,
        )
        prescriptive_loss = newsvendor_cost(
            loo_orders,
            y_metric,
            underage_cost,
            overage_cost,
        ).mean()
        return float(prescriptive_loss + metric_reg * np.mean(centered**2))

    result = minimize(
        objective,
        np.zeros(x_train.shape[1], dtype=float),
        method="Powell",
        options={"maxfev": metric_maxfev, "xtol": 1e-2, "ftol": 1e-3},
    )
    centered, scales = metric_scales(result.x)
    return centered, scales, float(result.fun)


def evaluate_replication(config: Config, seed: int) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    alpha = critical_fractile(config.underage_cost, config.overage_cost)

    x_train, y_train, _ = generate_data(
        rng,
        n=config.n_train,
        d=config.d,
        signal_dims=config.signal_dims,
        sigma=config.sigma,
    )
    x_val, y_val, _ = generate_data(
        rng,
        n=config.n_val,
        d=config.d,
        signal_dims=config.signal_dims,
        sigma=config.sigma,
    )
    x_test, y_test, mean_test = generate_data(
        rng,
        n=config.n_test,
        d=config.d,
        signal_dims=config.signal_dims,
        sigma=config.sigma,
    )

    x_train_std, x_val_std = standardize(x_train, x_val)
    _, x_test_std = standardize(x_train, x_test)

    global_order = global_saa_order(y_train, alpha)
    linear_coefficients = fit_linear_regression(x_train, y_train)
    point_orders = point_prediction_orders(x_test, linear_coefficients)

    y_sorted, y_order = sorted_response(y_train)

    fixed_bandwidth = tune_bandwidth(
        x_train_std,
        y_train,
        x_val_std,
        y_val,
        alpha,
        config.underage_cost,
        config.overage_cost,
    )
    fixed_orders = orders_from_squared_distances(
        pairwise_squared_distances(x_test_std, x_train_std),
        y_sorted,
        y_order,
        alpha,
        fixed_bandwidth,
    )

    raw_metric, learned_scales, metric_objective = learn_diagonal_metric(
        x_train_std,
        y_train,
        alpha,
        config.underage_cost,
        config.overage_cost,
        config.metric_fit_size,
        config.metric_reg,
        config.metric_maxfev,
    )
    x_train_metric = transform_with_metric(x_train_std, raw_metric)
    x_val_metric = transform_with_metric(x_val_std, raw_metric)
    x_test_metric = transform_with_metric(x_test_std, raw_metric)

    learned_bandwidth = tune_bandwidth(
        x_train_metric,
        y_train,
        x_val_metric,
        y_val,
        alpha,
        config.underage_cost,
        config.overage_cost,
    )
    learned_orders = orders_from_squared_distances(
        pairwise_squared_distances(x_test_metric, x_train_metric),
        y_sorted,
        y_order,
        alpha,
        learned_bandwidth,
    )

    oracle_shift = config.sigma * NormalDist().inv_cdf(alpha)
    oracle_orders = mean_test + oracle_shift

    oracle_cost = newsvendor_cost(
        oracle_orders,
        y_test,
        config.underage_cost,
        config.overage_cost,
    )
    global_cost = newsvendor_cost(
        np.full(config.n_test, global_order),
        y_test,
        config.underage_cost,
        config.overage_cost,
    )
    point_cost = newsvendor_cost(
        point_orders,
        y_test,
        config.underage_cost,
        config.overage_cost,
    )
    fixed_cost = newsvendor_cost(
        fixed_orders,
        y_test,
        config.underage_cost,
        config.overage_cost,
    )
    learned_cost = newsvendor_cost(
        learned_orders,
        y_test,
        config.underage_cost,
        config.overage_cost,
    )

    nuisance_mean = (
        float(learned_scales[config.signal_dims :].mean())
        if config.signal_dims < config.d
        else 1.0
    )
    signal_mean = float(learned_scales[: config.signal_dims].mean())
    scale_ratio = signal_mean / max(nuisance_mean, 1e-12)

    return {
        "oracle_cost": float(oracle_cost.mean()),
        "global_cost": float(global_cost.mean()),
        "point_cost": float(point_cost.mean()),
        "fixed_kernel_cost": float(fixed_cost.mean()),
        "learned_kernel_cost": float(learned_cost.mean()),
        "global_regret": float((global_cost - oracle_cost).mean()),
        "point_regret": float((point_cost - oracle_cost).mean()),
        "fixed_kernel_regret": float((fixed_cost - oracle_cost).mean()),
        "learned_kernel_regret": float((learned_cost - oracle_cost).mean()),
        "fixed_bandwidth": fixed_bandwidth,
        "learned_bandwidth": learned_bandwidth,
        "metric_objective": metric_objective,
        "signal_scale_mean": signal_mean,
        "nuisance_scale_mean": nuisance_mean,
        "signal_to_nuisance_scale": scale_ratio,
    }


def summarize(results: list[dict[str, float]]) -> dict[str, float]:
    keys = results[0].keys()
    return {key: float(np.mean([result[key] for result in results])) for key in keys}


def print_summary(config: Config, summary: dict[str, float]) -> None:
    learned_vs_global = 100.0 * (
        summary["global_cost"] - summary["learned_kernel_cost"]
    ) / summary["global_cost"]
    learned_vs_fixed = 100.0 * (
        summary["fixed_kernel_cost"] - summary["learned_kernel_cost"]
    ) / summary["fixed_kernel_cost"]
    learned_vs_point = 100.0 * (
        summary["point_cost"] - summary["learned_kernel_cost"]
    ) / summary["point_cost"]

    print("Minimal prescriptive benchmark: fixed-kernel vs learned-kernel weighted SAA")
    print(
        "Synthetic setup:",
        f"reps={config.reps}, train/val/test={config.n_train}/{config.n_val}/{config.n_test},",
        f"d={config.d}, signal_dims={config.signal_dims}, sigma={config.sigma:.1f}",
    )
    print(
        "Costs:",
        f"underage={config.underage_cost:.1f}, overage={config.overage_cost:.1f},",
        f"critical_fractile={critical_fractile(config.underage_cost, config.overage_cost):.2f}",
    )
    print(
        "Learned metric:",
        f"fit_size={min(config.n_train, config.metric_fit_size)},",
        f"reg={config.metric_reg:.3f}, maxfev={config.metric_maxfev}",
    )
    print()
    print("Average out-of-sample cost (lower is better)")
    print(f"  oracle policy                : {summary['oracle_cost']:.3f}")
    print(f"  global SAA baseline          : {summary['global_cost']:.3f}")
    print(f"  point-prediction baseline    : {summary['point_cost']:.3f}")
    print(f"  fixed kernel weighted SAA    : {summary['fixed_kernel_cost']:.3f}")
    print(f"  learned kernel weighted SAA  : {summary['learned_kernel_cost']:.3f}")
    print()
    print("Average prescriptive regret vs oracle")
    print(f"  global SAA baseline          : {summary['global_regret']:.3f}")
    print(f"  point-prediction baseline    : {summary['point_regret']:.3f}")
    print(f"  fixed kernel weighted SAA    : {summary['fixed_kernel_regret']:.3f}")
    print(f"  learned kernel weighted SAA  : {summary['learned_kernel_regret']:.3f}")
    print()
    print("Relative improvement of learned kernel weighted SAA")
    print(f"  vs global SAA                : {learned_vs_global:.1f}%")
    print(f"  vs point-prediction          : {learned_vs_point:.1f}%")
    print(f"  vs fixed kernel              : {learned_vs_fixed:.1f}%")
    print(f"  mean fixed bandwidth         : {summary['fixed_bandwidth']:.3f}")
    print(f"  mean learned bandwidth       : {summary['learned_bandwidth']:.3f}")
    if config.signal_dims < config.d:
        print(
            "  mean signal/nuisance scale  :",
            f"{summary['signal_to_nuisance_scale']:.2f}x",
        )


def main() -> None:
    config = parse_args()
    results = [
        evaluate_replication(config, seed=config.seed + replication)
        for replication in range(config.reps)
    ]
    print_summary(config, summarize(results))


if __name__ == "__main__":
    main()
