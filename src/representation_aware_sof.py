"""Representation-aware SOF transfer experiments.

The main report first learns representations for weighted SAA. This module
freezes those representations and plugs them into SOF-style forests to test
whether the learned context geometry transfers across local prescriptors.
"""

from __future__ import annotations

from contextlib import contextmanager
import os
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm.auto import tqdm

from experiment_suite import (
    ALPHA,
    BENCHMARK_PARALLEL_JOBS,
    OFFICIAL_SOF_VARIANT,
    apply_transform,
    bandwidth_candidates,
    benchmark_a_seed,
    build_stochopt_forest,
    choose_weighted_action,
    evaluate_allocation_transform,
    evaluate_newsvendor_transform,
    fit_official_newsvendor_sof,
    fit_pca_projection,
    generate_benchmark_a,
    generate_benchmark_b,
    generate_benchmark_c,
    kernel_average_from_weights,
    learn_neural_embedding_allocation,
    learn_neural_embedding_newsvendor,
    learn_rotated_metric_allocation,
    learn_rotated_metric_newsvendor,
    newsvendor_cost,
    orders_from_weights,
    pairwise_squared_distances,
    projection_from_selected_features,
    restore_lasso_selected_from_row,
    row_has_columns,
    select_lasso_features,
    select_multitarget_lasso_features,
    sorted_response,
    standardize,
    stochopt_forest_weight_matrix,
)


# ...existing code...
BENCHMARK_B_REPLICATIONS = int(os.environ.get("SOF_B_REPLICATIONS", "20"))
BENCHMARK_C_REPLICATIONS = int(os.environ.get("SOF_C_REPLICATIONS", "20"))
BENCHMARK_A_REPLICATIONS = int(os.environ.get("SOF_A_REPLICATIONS", "6"))
BENCHMARK_A_CONFIGS = ((20, 3), (50, 3), (100, 3))
BENCHMARK_A_TRAIN_SIZES = (100, 200, 400, 800)
EXPERIMENT_PROFILE = os.environ.get("SOF_PROFILE", "fast").lower()
PARALLEL_BACKEND = os.environ.get("SOF_PARALLEL_BACKEND", "multiprocessing")
DEFAULT_PARALLEL_JOBS = int(os.environ.get("SOF_N_JOBS", str(BENCHMARK_PARALLEL_JOBS)))
PROTOTYPE_SOF_MIN_LEAF_SIZE = 5


def _parse_float_grid(value: str | None, default: tuple[float, ...]) -> tuple[float, ...]:
    if not value:
        return default
    return tuple(float(token.strip()) for token in value.split(",") if token.strip())


def _parse_int_grid(value: str | None, default: tuple[int, ...]) -> tuple[int, ...]:
    if not value:
        return default
    return tuple(int(token.strip()) for token in value.split(",") if token.strip())


def _parse_bool(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _parse_optional_int(value: str | None, default: int | None) -> int | None:
    if value is None or value.strip() == "":
        return default
    parsed = int(value)
    if parsed <= 0:
        return None
    return parsed


if EXPERIMENT_PROFILE == "full":
    DEFAULT_A_SUBSAMPLE_GRID = (0.50, 0.75, 1.00)
    DEFAULT_A_DEPTH_GRID = (4, 6, 8)
    DEFAULT_A_NUM_TREES = 500
    DEFAULT_A_N_PROPOSALS = None
    DEFAULT_A_MTRY_MODE = "all"
    DEFAULT_A_BOOTSTRAP = True
    DEFAULT_B_SUBSAMPLE_GRID = (0.50, 0.75, 1.00)
    DEFAULT_B_DEPTH_GRID = (4, 6, 8)
    DEFAULT_B_NUM_TREES = 500
    DEFAULT_B_N_PROPOSALS = None
    DEFAULT_B_MTRY_MODE = "all"
    DEFAULT_B_BOOTSTRAP = True
    DEFAULT_C_SUBSAMPLE_GRID = (0.50, 0.75, 1.00)
    DEFAULT_C_DEPTH_GRID = (4, 6, 8)
    DEFAULT_C_NUM_TREES = 80
    DEFAULT_C_N_THRESHOLDS = 20
else:
    DEFAULT_A_SUBSAMPLE_GRID = (0.50, 1.00)
    DEFAULT_A_DEPTH_GRID = (4,)
    DEFAULT_A_NUM_TREES = 48
    DEFAULT_A_N_PROPOSALS = 48
    DEFAULT_A_MTRY_MODE = "sqrt"
    DEFAULT_A_BOOTSTRAP = False
    DEFAULT_B_SUBSAMPLE_GRID = (0.50, 1.00)
    DEFAULT_B_DEPTH_GRID = (4,)
    DEFAULT_B_NUM_TREES = 96
    DEFAULT_B_N_PROPOSALS = 64
    DEFAULT_B_MTRY_MODE = "sqrt"
    DEFAULT_B_BOOTSTRAP = False
    DEFAULT_C_SUBSAMPLE_GRID = (0.50, 1.00)
    DEFAULT_C_DEPTH_GRID = (4, 8)
    DEFAULT_C_NUM_TREES = 40
    DEFAULT_C_N_THRESHOLDS = 12

A_SOF_SUBSAMPLE_RATIO_GRID = _parse_float_grid(os.environ.get("SOF_A_SUBSAMPLE_GRID"), DEFAULT_A_SUBSAMPLE_GRID)
A_SOF_MAX_DEPTH_GRID = _parse_int_grid(os.environ.get("SOF_A_DEPTH_GRID"), DEFAULT_A_DEPTH_GRID)
A_OFFICIAL_SOF_NUM_TREES = int(os.environ.get("SOF_A_NUM_TREES", str(DEFAULT_A_NUM_TREES)))
A_OFFICIAL_SOF_N_PROPOSALS = _parse_optional_int(os.environ.get("SOF_A_N_PROPOSALS"), DEFAULT_A_N_PROPOSALS)
A_OFFICIAL_SOF_MTRY_MODE = os.environ.get("SOF_A_MTRY_MODE", DEFAULT_A_MTRY_MODE).lower()
A_OFFICIAL_SOF_BOOTSTRAP = _parse_bool(os.environ.get("SOF_A_BOOTSTRAP"), DEFAULT_A_BOOTSTRAP)
B_SOF_SUBSAMPLE_RATIO_GRID = _parse_float_grid(os.environ.get("SOF_B_SUBSAMPLE_GRID"), DEFAULT_B_SUBSAMPLE_GRID)
B_SOF_MAX_DEPTH_GRID = _parse_int_grid(os.environ.get("SOF_B_DEPTH_GRID"), DEFAULT_B_DEPTH_GRID)
B_OFFICIAL_SOF_NUM_TREES = int(os.environ.get("SOF_B_NUM_TREES", str(DEFAULT_B_NUM_TREES)))
B_OFFICIAL_SOF_N_PROPOSALS = _parse_optional_int(os.environ.get("SOF_B_N_PROPOSALS"), DEFAULT_B_N_PROPOSALS)
B_OFFICIAL_SOF_MTRY_MODE = os.environ.get("SOF_B_MTRY_MODE", DEFAULT_B_MTRY_MODE).lower()
B_OFFICIAL_SOF_BOOTSTRAP = _parse_bool(os.environ.get("SOF_B_BOOTSTRAP"), DEFAULT_B_BOOTSTRAP)
C_SOF_SUBSAMPLE_RATIO_GRID = _parse_float_grid(os.environ.get("SOF_C_SUBSAMPLE_GRID"), DEFAULT_C_SUBSAMPLE_GRID)
C_SOF_MAX_DEPTH_GRID = _parse_int_grid(os.environ.get("SOF_C_DEPTH_GRID"), DEFAULT_C_DEPTH_GRID)
C_PROTOTYPE_SOF_NUM_TREES = int(os.environ.get("SOF_C_NUM_TREES", str(DEFAULT_C_NUM_TREES)))
C_PROTOTYPE_SOF_N_THRESHOLDS = int(os.environ.get("SOF_C_N_THRESHOLDS", str(DEFAULT_C_N_THRESHOLDS)))


# ===== Method Registry =====
# Declarative method definitions for easier extensibility
METHOD_REGISTRY_B = {
    "SOF-raw": {
        "fit_fn": "fit_tuned_official_sof_transform",
        "task_kind": "newsvendor",
        "seed_offset": 90_000,
        "transform_kind": "identity",
        "transform_payload_key": None,
        "subsample_grid": "B_SOF_SUBSAMPLE_RATIO_GRID",
        "depth_grid": "B_SOF_MAX_DEPTH_GRID",
        "num_trees": "B_OFFICIAL_SOF_NUM_TREES",
        "n_proposals": "B_OFFICIAL_SOF_N_PROPOSALS",
        "mtry_mode": "B_OFFICIAL_SOF_MTRY_MODE",
        "bootstrap": "B_OFFICIAL_SOF_BOOTSTRAP",
    },
    "SOF-PCA": {
        "fit_fn": "fit_tuned_official_sof_transform",
        "task_kind": "newsvendor",
        "seed_offset": 91_000,
        "transform_kind": "linear_projection",
        "transform_payload_key": "pca_projection",
        "subsample_grid": "B_SOF_SUBSAMPLE_RATIO_GRID",
        "depth_grid": "B_SOF_MAX_DEPTH_GRID",
        "num_trees": "B_OFFICIAL_SOF_NUM_TREES",
        "n_proposals": "B_OFFICIAL_SOF_N_PROPOSALS",
        "mtry_mode": "B_OFFICIAL_SOF_MTRY_MODE",
        "bootstrap": "B_OFFICIAL_SOF_BOOTSTRAP",
    },
    "SOF-LASSO": {
        "fit_fn": "fit_tuned_official_sof_transform",
        "task_kind": "newsvendor",
        "seed_offset": 92_000,
        "transform_kind": "linear_projection",
        "transform_payload_key": "lasso_projection",
        "subsample_grid": "B_SOF_SUBSAMPLE_RATIO_GRID",
        "depth_grid": "B_SOF_MAX_DEPTH_GRID",
        "num_trees": "B_OFFICIAL_SOF_NUM_TREES",
        "n_proposals": "B_OFFICIAL_SOF_N_PROPOSALS",
        "mtry_mode": "B_OFFICIAL_SOF_MTRY_MODE",
        "bootstrap": "B_OFFICIAL_SOF_BOOTSTRAP",
    },
    "SOF-pred": {
        "fit_fn": "fit_tuned_official_sof_transform",
        "task_kind": "newsvendor",
        "seed_offset": 93_000,
        "transform_kind": "rotated_2d",
        "transform_payload_key": "prediction_parameters",
        "subsample_grid": "B_SOF_SUBSAMPLE_RATIO_GRID",
        "depth_grid": "B_SOF_MAX_DEPTH_GRID",
        "num_trees": "B_OFFICIAL_SOF_NUM_TREES",
        "n_proposals": "B_OFFICIAL_SOF_N_PROPOSALS",
        "mtry_mode": "B_OFFICIAL_SOF_MTRY_MODE",
        "bootstrap": "B_OFFICIAL_SOF_BOOTSTRAP",
    },
    "SOF-dec": {
        "fit_fn": "fit_tuned_official_sof_transform",
        "task_kind": "newsvendor",
        "seed_offset": 94_000,
        "transform_kind": "rotated_2d",
        "transform_payload_key": "decision_parameters",
        "subsample_grid": "B_SOF_SUBSAMPLE_RATIO_GRID",
        "depth_grid": "B_SOF_MAX_DEPTH_GRID",
        "num_trees": "B_OFFICIAL_SOF_NUM_TREES",
        "n_proposals": "B_OFFICIAL_SOF_N_PROPOSALS",
        "mtry_mode": "B_OFFICIAL_SOF_MTRY_MODE",
        "bootstrap": "B_OFFICIAL_SOF_BOOTSTRAP",
    },
}

SOF_PLUGIN_METHOD_ORDER = list(METHOD_REGISTRY_B.keys()) + ["Kernel-fixed", "Kernel-dec", "Neural-dec"]
SOF_DECOMPOSITION_METHOD_ORDER = ["SOF-raw", "SOF-raw+kernel", "SOF-dec", "SOF-dec+kernel", "Kernel-dec"]

# Verification: Ensure method order consistency
_REGISTRY_METHODS = set(METHOD_REGISTRY_B.keys())
_PLUGIN_METHODS = set(SOF_PLUGIN_METHOD_ORDER)
assert _REGISTRY_METHODS.issubset(_PLUGIN_METHODS), f"Registry methods must be in SOF_PLUGIN_METHOD_ORDER. Missing: {_REGISTRY_METHODS - _PLUGIN_METHODS}"


def _log(message: str) -> None:
    print(message, flush=True)


def _start_progress(name: str, pending: int) -> None:
    if pending == 0:
        _log(f"[{name}] no pending tasks; using cached results")
    else:
        _log(f"[{name}] starting {pending} task(s)")


@contextmanager
def _tqdm_joblib(description: str, total: int):
    progress_bar = tqdm(total=total, desc=description, dynamic_ncols=True)
    previous_callback = joblib.parallel.BatchCompletionCallBack

    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args: Any, **kwargs: Any) -> Any:
            progress_bar.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield progress_bar
    finally:
        joblib.parallel.BatchCompletionCallBack = previous_callback
        progress_bar.close()


def build_mixed_row_lookup(
    frame: pd.DataFrame | None,
    key_columns: tuple[str, ...],
) -> dict[tuple[Any, ...], dict[str, Any]]:
    if frame is None or frame.empty:
        return {}

    lookup: dict[tuple[Any, ...], dict[str, Any]] = {}
    for _, row in frame.iterrows():
        key = []
        for column in key_columns:
            value = row[column]
            if isinstance(value, (np.integer, int)):
                key.append(int(value))
            elif isinstance(value, (np.floating, float)) and float(value).is_integer():
                key.append(int(value))
            else:
                key.append(str(value))
        lookup[tuple(key)] = row.to_dict()
    return lookup


def restore_screened_diagonal_scales(
    row: dict[str, Any] | None,
    prefix: str,
    dimension: int,
) -> np.ndarray | None:
    if row is None:
        return None
    columns = [f"{prefix}_scale_{feature_index}" for feature_index in range(dimension)]
    if not row_has_columns(row, columns):
        return None
    return np.array([float(row[column]) for column in columns], dtype=float)


def official_sof_weight_matrix(
    model: Any,
    x_query: np.ndarray,
) -> np.ndarray:
    return np.vstack([model.get_weights(query) for query in x_query])


def _resolve_mtry(dimension: int, mode: str) -> int | None:
    normalized = mode.lower()
    if normalized == "all":
        return dimension
    if normalized == "sqrt":
        return max(1, int(np.ceil(np.sqrt(dimension))))
    if normalized == "half":
        return max(1, int(np.ceil(dimension / 2.0)))
    if normalized == "auto":
        return None
    raise ValueError(f"Unknown mtry mode: {mode}")


def prototype_split_frequency(
    forest: list[dict[str, Any]],
    dimension: int,
) -> np.ndarray:
    counts = np.zeros(dimension, dtype=float)

    def recurse(node: dict[str, Any]) -> None:
        feature = node["feature"]
        if feature is None:
            return
        counts[int(feature)] += 1.0
        recurse(node["left"])
        recurse(node["right"])

    for tree in forest:
        recurse(tree)
    total = counts.sum()
    if total <= 0.0:
        return np.full(dimension, 1.0 / max(dimension, 1))
    return counts / total


def average_newsvendor_cost_from_weights(
    weight_rows: np.ndarray,
    y_train: np.ndarray,
    y_eval: np.ndarray,
) -> float:
    y_sorted, y_order = sorted_response(y_train)
    orders = orders_from_weights(weight_rows, y_sorted, y_order, ALPHA)
    return float(newsvendor_cost(orders, y_eval).mean())


def evaluate_newsvendor_weight_rows(
    weight_rows: np.ndarray,
    y_train: np.ndarray,
    y_eval: np.ndarray,
    mean_eval: np.ndarray,
    oracle_orders: np.ndarray,
) -> dict[str, float]:
    avg_cost = average_newsvendor_cost_from_weights(weight_rows, y_train, y_eval)
    oracle_cost = float(newsvendor_cost(oracle_orders, y_eval).mean())
    mean_prediction = kernel_average_from_weights(weight_rows, y_train)
    pred_mse = float(np.mean((mean_prediction - mean_eval) ** 2))
    avg_support = float(np.mean(np.sum(weight_rows > 1e-12, axis=1)))
    return {
        "cost": avg_cost,
        "regret": avg_cost - oracle_cost,
        "pred_mse": pred_mse,
        "avg_leaf_support": avg_support,
    }


def average_allocation_cost_from_weights(
    weight_rows: np.ndarray,
    rewards_train: np.ndarray,
    rewards_eval: np.ndarray,
) -> float:
    decisions = np.empty(len(weight_rows), dtype=int)
    for row_index, row_weights in enumerate(weight_rows):
        decisions[row_index] = choose_weighted_action(row_weights, rewards_train)
    return float(np.mean(-rewards_eval[np.arange(len(rewards_eval)), decisions]))


def evaluate_allocation_weight_rows(
    weight_rows: np.ndarray,
    rewards_train: np.ndarray,
    rewards_eval: np.ndarray,
    mean_rewards_eval: np.ndarray,
) -> tuple[dict[str, float], np.ndarray]:
    decisions = np.empty(len(weight_rows), dtype=int)
    costs = np.empty(len(weight_rows))
    for row_index, (row_weights, reward) in enumerate(zip(weight_rows, rewards_eval)):
        decisions[row_index] = choose_weighted_action(row_weights, rewards_train)
        costs[row_index] = -reward[decisions[row_index]]
    predicted_rewards = kernel_average_from_weights(weight_rows, rewards_train)
    pred_mse = float(np.mean((predicted_rewards - mean_rewards_eval) ** 2))
    oracle_cost = float(np.mean(-np.max(mean_rewards_eval, axis=1)))
    avg_support = float(np.mean(np.sum(weight_rows > 1e-12, axis=1)))
    return (
        {
            "cost": float(costs.mean()),
            "regret": float(costs.mean() - oracle_cost),
            "pred_mse": pred_mse,
            "avg_leaf_support": avg_support,
        },
        decisions,
    )


def combine_forest_and_kernel_weights(
    base_weights: np.ndarray,
    x_query: np.ndarray,
    x_train: np.ndarray,
    bandwidth: float,
) -> np.ndarray:
    squared_distances = pairwise_squared_distances(x_query, x_train)
    kernel_weights = np.exp(-squared_distances / (2.0 * bandwidth**2))
    combined = base_weights * kernel_weights
    empty_rows = combined.sum(axis=1) <= 0.0
    if np.any(empty_rows):
        combined[empty_rows] = base_weights[empty_rows]
    return combined


def tune_leaf_kernel_bandwidth_newsvendor(
    base_weights_val: np.ndarray,
    x_train_t: np.ndarray,
    x_val_t: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
) -> float:
    candidates = bandwidth_candidates(x_train_t)
    best_bandwidth = float(candidates[0])
    best_cost = float("inf")
    for bandwidth in candidates:
        weight_rows = combine_forest_and_kernel_weights(base_weights_val, x_val_t, x_train_t, float(bandwidth))
        avg_cost = average_newsvendor_cost_from_weights(weight_rows, y_train, y_val)
        if avg_cost < best_cost:
            best_cost = avg_cost
            best_bandwidth = float(bandwidth)
    return best_bandwidth


def tune_leaf_kernel_bandwidth_allocation(
    base_weights_val: np.ndarray,
    x_train_t: np.ndarray,
    x_val_t: np.ndarray,
    rewards_train: np.ndarray,
    rewards_val: np.ndarray,
) -> float:
    candidates = bandwidth_candidates(x_train_t)
    best_bandwidth = float(candidates[0])
    best_cost = float("inf")
    for bandwidth in candidates:
        weight_rows = combine_forest_and_kernel_weights(base_weights_val, x_val_t, x_train_t, float(bandwidth))
        avg_cost = average_allocation_cost_from_weights(weight_rows, rewards_train, rewards_val)
        if avg_cost < best_cost:
            best_cost = avg_cost
            best_bandwidth = float(bandwidth)
    return best_bandwidth


def tune_official_newsvendor_sof(
    x_train_t: np.ndarray,
    y_train: np.ndarray,
    x_val_t: np.ndarray,
    y_val: np.ndarray,
    seed: int,
    subsample_grid: tuple[float, ...],
    depth_grid: tuple[int, ...],
    num_trees: int,
    n_proposals: int | None,
    mtry_mode: str,
    bootstrap: bool,
) -> dict[str, Any]:
    best: dict[str, Any] | None = None
    resolved_mtry = _resolve_mtry(x_train_t.shape[1], mtry_mode)
    for depth_index, max_depth in enumerate(depth_grid):
        for subsample_index, subsample_ratio in enumerate(subsample_grid):
            model = fit_official_newsvendor_sof(
                x_train_t,
                y_train,
                seed=seed + 100 * depth_index + 10 * subsample_index,
                variant=OFFICIAL_SOF_VARIANT,
                num_trees=num_trees,
                subsample_ratio=float(subsample_ratio),
                bootstrap=bootstrap,
                mtry=resolved_mtry,
                max_depth=int(max_depth),
                n_proposals=n_proposals,
            )
            val_weights = official_sof_weight_matrix(model, x_val_t)
            val_cost = average_newsvendor_cost_from_weights(val_weights, y_train, y_val)
            if best is None or val_cost < best["validation_cost"]:
                best = {
                    "model": model,
                    "validation_cost": float(val_cost),
                    "subsample_ratio": float(subsample_ratio),
                    "max_depth": int(max_depth),
                    "val_weights": val_weights,
                }
    if best is None:
        raise RuntimeError("Official SOF tuning failed to produce a model")
    return best


def tune_allocation_prototype_sof(
    x_train_t: np.ndarray,
    rewards_train: np.ndarray,
    x_val_t: np.ndarray,
    rewards_val: np.ndarray,
    seed: int,
    subsample_grid: tuple[float, ...],
    depth_grid: tuple[int, ...],
    num_trees: int,
    n_thresholds: int,
) -> dict[str, Any]:
    best: dict[str, Any] | None = None
    for depth_index, max_depth in enumerate(depth_grid):
        for subsample_index, subsample_ratio in enumerate(subsample_grid):
            forest = build_stochopt_forest(
                x_train_t,
                rewards_train,
                task_kind="allocation",
                n_trees=num_trees,
                subsample_ratio=float(subsample_ratio),
                max_depth=int(max_depth),
                min_leaf_size=PROTOTYPE_SOF_MIN_LEAF_SIZE,
                n_thresholds=n_thresholds,
                balancedness_tol=0.1,
                honest=False,
                seed=seed + 100 * depth_index + 10 * subsample_index,
            )
            val_weights = stochopt_forest_weight_matrix(forest, x_val_t, len(x_train_t))
            val_cost = average_allocation_cost_from_weights(val_weights, rewards_train, rewards_val)
            if best is None or val_cost < best["validation_cost"]:
                best = {
                    "forest": forest,
                    "validation_cost": float(val_cost),
                    "subsample_ratio": float(subsample_ratio),
                    "max_depth": int(max_depth),
                    "val_weights": val_weights,
                }
    if best is None:
        raise RuntimeError("Allocation SOF tuning failed to produce a forest")
    return best


def fit_tuned_official_sof_transform(
    x_train_std: np.ndarray,
    y_train: np.ndarray,
    x_val_std: np.ndarray,
    y_val: np.ndarray,
    mean_val: np.ndarray,
    x_test_std: np.ndarray,
    y_test: np.ndarray,
    mean_test: np.ndarray,
    oracle_orders: np.ndarray,
    transform_kind: str,
    transform_payload: np.ndarray | None,
    seed: int,
    subsample_grid: tuple[float, ...],
    depth_grid: tuple[int, ...],
    num_trees: int,
    n_proposals: int | None,
    mtry_mode: str,
    bootstrap: bool,
) -> dict[str, Any]:
    x_train_t, x_val_t, x_test_t, metadata = apply_transform(
        x_train_std,
        x_val_std,
        x_test_std,
        transform_kind,
        transform_payload,
    )
    tuning = tune_official_newsvendor_sof(
        x_train_t,
        y_train,
        x_val_t,
        y_val,
        seed=seed,
        subsample_grid=subsample_grid,
        depth_grid=depth_grid,
        num_trees=num_trees,
        n_proposals=n_proposals,
        mtry_mode=mtry_mode,
        bootstrap=bootstrap,
    )
    test_weights = official_sof_weight_matrix(tuning["model"], x_test_t)
    metrics = evaluate_newsvendor_weight_rows(test_weights, y_train, y_test, mean_test, oracle_orders)
    split_frequency = np.asarray(
        tuning["model"].compute_feature_split_freq(x_train_t.shape[1]),
        dtype=float,
    )
    total_splits = float(split_frequency.sum())
    if total_splits > 0.0:
        split_frequency = split_frequency / total_splits
    else:
        split_frequency = np.full(x_train_t.shape[1], 1.0 / max(x_train_t.shape[1], 1))
    metrics.update(metadata)
    metrics.update(
        {
            "subsample_ratio": tuning["subsample_ratio"],
            "max_depth": tuning["max_depth"],
            "validation_cost": tuning["validation_cost"],
        }
    )
    return {
        "metrics": metrics,
        "x_train_t": x_train_t,
        "x_val_t": x_val_t,
        "x_test_t": x_test_t,
        "val_weights": tuning["val_weights"],
        "test_weights": test_weights,
        "split_frequency": split_frequency,
    }


def fit_tuned_allocation_sof_transform(
    x_train_std: np.ndarray,
    rewards_train: np.ndarray,
    x_val_std: np.ndarray,
    rewards_val: np.ndarray,
    mean_rewards_val: np.ndarray,
    x_test_std: np.ndarray,
    rewards_test: np.ndarray,
    mean_rewards_test: np.ndarray,
    transform_kind: str,
    transform_payload: np.ndarray | None,
    seed: int,
    subsample_grid: tuple[float, ...],
    depth_grid: tuple[int, ...],
    num_trees: int,
    n_thresholds: int,
) -> dict[str, Any]:
    x_train_t, x_val_t, x_test_t, metadata = apply_transform(
        x_train_std,
        x_val_std,
        x_test_std,
        transform_kind,
        transform_payload,
    )
    tuning = tune_allocation_prototype_sof(
        x_train_t,
        rewards_train,
        x_val_t,
        rewards_val,
        seed=seed,
        subsample_grid=subsample_grid,
        depth_grid=depth_grid,
        num_trees=num_trees,
        n_thresholds=n_thresholds,
    )
    test_weights = stochopt_forest_weight_matrix(tuning["forest"], x_test_t, len(x_train_t))
    metrics, decisions = evaluate_allocation_weight_rows(
        test_weights,
        rewards_train,
        rewards_test,
        mean_rewards_test,
    )
    metrics.update(metadata)
    metrics.update(
        {
            "subsample_ratio": tuning["subsample_ratio"],
            "max_depth": tuning["max_depth"],
            "validation_cost": tuning["validation_cost"],
        }
    )
    return {
        "metrics": metrics,
        "decisions": decisions,
        "x_train_t": x_train_t,
        "x_val_t": x_val_t,
        "x_test_t": x_test_t,
        "val_weights": tuning["val_weights"],
        "test_weights": test_weights,
        "split_frequency": prototype_split_frequency(tuning["forest"], x_train_t.shape[1]),
    }


def summary_rows_from_method_results(
    benchmark: str,
    replication: int,
    method_results: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for method, payload in method_results.items():
        metrics = payload["metrics"]
        rows.append(
            {
                "benchmark": benchmark,
                "replication": replication,
                "method": method,
                "cost": float(metrics["cost"]),
                "regret": float(metrics["regret"]),
                "pred_mse": float(metrics["pred_mse"]),
                "avg_leaf_support": float(metrics.get("avg_leaf_support", np.nan)),
                "subsample_ratio": float(metrics.get("subsample_ratio", np.nan)),
                "max_depth": float(metrics.get("max_depth", np.nan)),
                "validation_cost": float(metrics.get("validation_cost", np.nan)),
            }
        )
    return rows


def decomposition_rows_newsvendor(
    benchmark: str,
    replication: int,
    raw_sof: dict[str, Any],
    dec_sof: dict[str, Any],
    kernel_dec_metrics: dict[str, float],
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
    mean_test: np.ndarray,
    oracle_orders: np.ndarray,
) -> list[dict[str, Any]]:
    raw_bandwidth = tune_leaf_kernel_bandwidth_newsvendor(
        raw_sof["val_weights"],
        raw_sof["x_train_t"],
        raw_sof["x_val_t"],
        y_train,
        y_val,
    )
    dec_bandwidth = tune_leaf_kernel_bandwidth_newsvendor(
        dec_sof["val_weights"],
        dec_sof["x_train_t"],
        dec_sof["x_val_t"],
        y_train,
        y_val,
    )
    raw_kernel_weights = combine_forest_and_kernel_weights(
        raw_sof["test_weights"],
        raw_sof["x_test_t"],
        raw_sof["x_train_t"],
        raw_bandwidth,
    )
    dec_kernel_weights = combine_forest_and_kernel_weights(
        dec_sof["test_weights"],
        dec_sof["x_test_t"],
        dec_sof["x_train_t"],
        dec_bandwidth,
    )
    raw_kernel_metrics = evaluate_newsvendor_weight_rows(raw_kernel_weights, y_train, y_test, mean_test, oracle_orders)
    dec_kernel_metrics = evaluate_newsvendor_weight_rows(dec_kernel_weights, y_train, y_test, mean_test, oracle_orders)
    rows = [
        {
            "benchmark": benchmark,
            "replication": replication,
            "method": "SOF-raw",
            "regret": float(raw_sof["metrics"]["regret"]),
            "cost": float(raw_sof["metrics"]["cost"]),
            "pred_mse": float(raw_sof["metrics"]["pred_mse"]),
        },
        {
            "benchmark": benchmark,
            "replication": replication,
            "method": "SOF-raw+kernel",
            "regret": float(raw_kernel_metrics["regret"]),
            "cost": float(raw_kernel_metrics["cost"]),
            "pred_mse": float(raw_kernel_metrics["pred_mse"]),
        },
        {
            "benchmark": benchmark,
            "replication": replication,
            "method": "SOF-dec",
            "regret": float(dec_sof["metrics"]["regret"]),
            "cost": float(dec_sof["metrics"]["cost"]),
            "pred_mse": float(dec_sof["metrics"]["pred_mse"]),
        },
        {
            "benchmark": benchmark,
            "replication": replication,
            "method": "SOF-dec+kernel",
            "regret": float(dec_kernel_metrics["regret"]),
            "cost": float(dec_kernel_metrics["cost"]),
            "pred_mse": float(dec_kernel_metrics["pred_mse"]),
        },
        {
            "benchmark": benchmark,
            "replication": replication,
            "method": "Kernel-dec",
            "regret": float(kernel_dec_metrics["regret"]),
            "cost": float(kernel_dec_metrics["cost"]),
            "pred_mse": float(kernel_dec_metrics["pred_mse"]),
        },
    ]
    return rows


def decomposition_rows_allocation(
    benchmark: str,
    replication: int,
    raw_sof: dict[str, Any],
    dec_sof: dict[str, Any],
    kernel_dec_metrics: dict[str, float],
    rewards_train: np.ndarray,
    rewards_val: np.ndarray,
    rewards_test: np.ndarray,
    mean_rewards_test: np.ndarray,
) -> list[dict[str, Any]]:
    raw_bandwidth = tune_leaf_kernel_bandwidth_allocation(
        raw_sof["val_weights"],
        raw_sof["x_train_t"],
        raw_sof["x_val_t"],
        rewards_train,
        rewards_val,
    )
    dec_bandwidth = tune_leaf_kernel_bandwidth_allocation(
        dec_sof["val_weights"],
        dec_sof["x_train_t"],
        dec_sof["x_val_t"],
        rewards_train,
        rewards_val,
    )
    raw_kernel_weights = combine_forest_and_kernel_weights(
        raw_sof["test_weights"],
        raw_sof["x_test_t"],
        raw_sof["x_train_t"],
        raw_bandwidth,
    )
    dec_kernel_weights = combine_forest_and_kernel_weights(
        dec_sof["test_weights"],
        dec_sof["x_test_t"],
        dec_sof["x_train_t"],
        dec_bandwidth,
    )
    raw_kernel_metrics, _ = evaluate_allocation_weight_rows(
        raw_kernel_weights,
        rewards_train,
        rewards_test,
        mean_rewards_test,
    )
    dec_kernel_metrics, _ = evaluate_allocation_weight_rows(
        dec_kernel_weights,
        rewards_train,
        rewards_test,
        mean_rewards_test,
    )
    rows = [
        {
            "benchmark": benchmark,
            "replication": replication,
            "method": "SOF-raw",
            "regret": float(raw_sof["metrics"]["regret"]),
            "cost": float(raw_sof["metrics"]["cost"]),
            "pred_mse": float(raw_sof["metrics"]["pred_mse"]),
        },
        {
            "benchmark": benchmark,
            "replication": replication,
            "method": "SOF-raw+kernel",
            "regret": float(raw_kernel_metrics["regret"]),
            "cost": float(raw_kernel_metrics["cost"]),
            "pred_mse": float(raw_kernel_metrics["pred_mse"]),
        },
        {
            "benchmark": benchmark,
            "replication": replication,
            "method": "SOF-dec",
            "regret": float(dec_sof["metrics"]["regret"]),
            "cost": float(dec_sof["metrics"]["cost"]),
            "pred_mse": float(dec_sof["metrics"]["pred_mse"]),
        },
        {
            "benchmark": benchmark,
            "replication": replication,
            "method": "SOF-dec+kernel",
            "regret": float(dec_kernel_metrics["regret"]),
            "cost": float(dec_kernel_metrics["cost"]),
            "pred_mse": float(dec_kernel_metrics["pred_mse"]),
        },
        {
            "benchmark": benchmark,
            "replication": replication,
            "method": "Kernel-dec",
            "regret": float(kernel_dec_metrics["regret"]),
            "cost": float(kernel_dec_metrics["cost"]),
            "pred_mse": float(kernel_dec_metrics["pred_mse"]),
        },
    ]
    return rows


def run_benchmark_b_plugin_replication(
    replication: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    seed = 2000 + replication
    rng = np.random.default_rng(seed)
    x_train, y_train, _, _ = generate_benchmark_b(rng, 250)
    x_val, y_val, mean_val, _ = generate_benchmark_b(rng, 150)
    x_test, y_test, mean_test, oracle_orders = generate_benchmark_b(rng, 2000)

    x_train_std, x_val_std = standardize(x_train, x_val)
    _, x_test_std = standardize(x_train, x_test)

    pca_projection = fit_pca_projection(x_train_std, n_components=2)
    lasso_selected, _, _ = select_lasso_features(x_train_std, y_train, x_val_std, y_val)
    lasso_projection = projection_from_selected_features(x_train_std.shape[1], lasso_selected)

    prediction_parameters, _ = learn_rotated_metric_newsvendor(
        x_train_std,
        y_train,
        objective_kind="prediction",
        fit_size=120,
        reg=0.02,
        maxfev=160,
    )
    decision_parameters, _ = learn_rotated_metric_newsvendor(
        x_train_std,
        y_train,
        objective_kind="decision",
        initial_parameters=prediction_parameters,
        fit_size=120,
        reg=0.02,
        maxfev=160,
    )
    neural_prediction_parameters, _ = learn_neural_embedding_newsvendor(
        x_train_std,
        y_train,
        objective_kind="prediction",
        fit_size=120,
        reg=0.01,
        maxfev=220,
    )
    neural_parameters, _ = learn_neural_embedding_newsvendor(
        x_train_std,
        y_train,
        objective_kind="decision",
        initial_parameters=neural_prediction_parameters,
        fit_size=120,
        reg=0.01,
        maxfev=220,
    )

    # Build payload mapping for method registry
    payloads = {
        "pca_projection": pca_projection,
        "lasso_projection": lasso_projection,
        "prediction_parameters": prediction_parameters,
        "decision_parameters": decision_parameters,
    }

    # Fit SOF methods from registry
    method_results = {}
    for method_name, spec in METHOD_REGISTRY_B.items():
        transform_payload = payloads.get(spec["transform_payload_key"]) if spec["transform_payload_key"] else None
        
        method_results[method_name] = fit_tuned_official_sof_transform(
            x_train_std,
            y_train,
            x_val_std,
            y_val,
            mean_val,
            x_test_std,
            y_test,
            mean_test,
            oracle_orders,
            transform_kind=spec["transform_kind"],
            transform_payload=transform_payload,
            seed=spec["seed_offset"] + replication,
            subsample_grid=globals()[spec["subsample_grid"]],
            depth_grid=globals()[spec["depth_grid"]],
            num_trees=globals()[spec["num_trees"]],
            n_proposals=globals()[spec["n_proposals"]],
            mtry_mode=globals()[spec["mtry_mode"]],
            bootstrap=globals()[spec["bootstrap"]],
        )

    # Fit kernel and neural methods (non-registry)
    method_results.update(
        {
            "Kernel-fixed": {
                "metrics": evaluate_newsvendor_transform(
                    x_train_std,
                    y_train,
                    x_val_std,
                    y_val,
                    mean_val,
                    x_test_std,
                    y_test,
                    mean_test,
                    oracle_orders,
                    transform_kind="identity",
                    transform_payload=None,
                )
            },
            "Kernel-dec": {
                "metrics": evaluate_newsvendor_transform(
                    x_train_std,
                    y_train,
                    x_val_std,
                    y_val,
                    mean_val,
                    x_test_std,
                    y_test,
                    mean_test,
                    oracle_orders,
                    transform_kind="rotated_2d",
                    transform_payload=decision_parameters,
                )
            },
            "Neural-dec": {
                "metrics": evaluate_newsvendor_transform(
                    x_train_std,
                    y_train,
                    x_val_std,
                    y_val,
                    mean_val,
                    x_test_std,
                    y_test,
                    mean_test,
                    oracle_orders,
                    transform_kind="neural_embedding",
                    transform_payload=neural_parameters,
                )
            },
        }
    )

    summary_rows = summary_rows_from_method_results("B", replication, method_results)
    decomposition_rows = decomposition_rows_newsvendor(
        benchmark="B",
        replication=replication,
        raw_sof=method_results["SOF-raw"],
        dec_sof=method_results["SOF-dec"],
        kernel_dec_metrics=method_results["Kernel-dec"]["metrics"],
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
        mean_test=mean_test,
        oracle_orders=oracle_orders,
    )
    return summary_rows, decomposition_rows


def run_benchmark_c_plugin_replication(
    replication: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], pd.DataFrame]:
    seed = 3000 + replication
    rng = np.random.default_rng(seed)
    x_train, rewards_train, _, _ = generate_benchmark_c(rng, 300)
    x_val, rewards_val, mean_rewards_val, _ = generate_benchmark_c(rng, 150)
    x_test, rewards_test, mean_rewards_test, signal_test = generate_benchmark_c(rng, 3000)

    x_train_std, x_val_std = standardize(x_train, x_val)
    _, x_test_std = standardize(x_train, x_test)

    pca_projection = fit_pca_projection(x_train_std, n_components=2)
    lasso_selected, _, _ = select_multitarget_lasso_features(
        x_train_std,
        rewards_train,
        x_val_std,
        rewards_val,
    )
    lasso_projection = projection_from_selected_features(x_train_std.shape[1], lasso_selected)

    prediction_parameters, _ = learn_rotated_metric_allocation(
        x_train_std,
        rewards_train,
        objective_kind="prediction",
        fit_size=120,
        reg=0.005,
        maxfev=220,
    )
    decision_parameters, _ = learn_rotated_metric_allocation(
        x_train_std,
        rewards_train,
        objective_kind="decision",
        initial_parameters=prediction_parameters,
        fit_size=120,
        reg=0.005,
        maxfev=220,
    )
    neural_prediction_parameters, _ = learn_neural_embedding_allocation(
        x_train_std,
        rewards_train,
        objective_kind="prediction",
        fit_size=120,
        reg=0.005,
        maxfev=220,
    )
    neural_parameters, _ = learn_neural_embedding_allocation(
        x_train_std,
        rewards_train,
        objective_kind="decision",
        initial_parameters=neural_prediction_parameters,
        fit_size=120,
        reg=0.005,
        maxfev=220,
    )

    kernel_fixed_metrics, kernel_fixed_decisions = evaluate_allocation_transform(
        x_train_std,
        rewards_train,
        x_val_std,
        rewards_val,
        mean_rewards_val,
        x_test_std,
        rewards_test,
        mean_rewards_test,
        transform_kind="identity",
        transform_payload=None,
    )
    kernel_dec_metrics, kernel_decisions = evaluate_allocation_transform(
        x_train_std,
        rewards_train,
        x_val_std,
        rewards_val,
        mean_rewards_val,
        x_test_std,
        rewards_test,
        mean_rewards_test,
        transform_kind="rotated_2d",
        transform_payload=decision_parameters,
    )
    neural_metrics, neural_decisions = evaluate_allocation_transform(
        x_train_std,
        rewards_train,
        x_val_std,
        rewards_val,
        mean_rewards_val,
        x_test_std,
        rewards_test,
        mean_rewards_test,
        transform_kind="neural_embedding",
        transform_payload=neural_parameters,
    )

    raw_sof = fit_tuned_allocation_sof_transform(
        x_train_std,
        rewards_train,
        x_val_std,
        rewards_val,
        mean_rewards_val,
        x_test_std,
        rewards_test,
        mean_rewards_test,
        transform_kind="identity",
        transform_payload=None,
        seed=110_000 + replication,
        subsample_grid=C_SOF_SUBSAMPLE_RATIO_GRID,
        depth_grid=C_SOF_MAX_DEPTH_GRID,
        num_trees=C_PROTOTYPE_SOF_NUM_TREES,
        n_thresholds=C_PROTOTYPE_SOF_N_THRESHOLDS,
    )
    pca_sof = fit_tuned_allocation_sof_transform(
        x_train_std,
        rewards_train,
        x_val_std,
        rewards_val,
        mean_rewards_val,
        x_test_std,
        rewards_test,
        mean_rewards_test,
        transform_kind="linear_projection",
        transform_payload=pca_projection,
        seed=111_000 + replication,
        subsample_grid=C_SOF_SUBSAMPLE_RATIO_GRID,
        depth_grid=C_SOF_MAX_DEPTH_GRID,
        num_trees=C_PROTOTYPE_SOF_NUM_TREES,
        n_thresholds=C_PROTOTYPE_SOF_N_THRESHOLDS,
    )
    lasso_sof = fit_tuned_allocation_sof_transform(
        x_train_std,
        rewards_train,
        x_val_std,
        rewards_val,
        mean_rewards_val,
        x_test_std,
        rewards_test,
        mean_rewards_test,
        transform_kind="linear_projection",
        transform_payload=lasso_projection,
        seed=112_000 + replication,
        subsample_grid=C_SOF_SUBSAMPLE_RATIO_GRID,
        depth_grid=C_SOF_MAX_DEPTH_GRID,
        num_trees=C_PROTOTYPE_SOF_NUM_TREES,
        n_thresholds=C_PROTOTYPE_SOF_N_THRESHOLDS,
    )
    pred_sof = fit_tuned_allocation_sof_transform(
        x_train_std,
        rewards_train,
        x_val_std,
        rewards_val,
        mean_rewards_val,
        x_test_std,
        rewards_test,
        mean_rewards_test,
        transform_kind="rotated_2d",
        transform_payload=prediction_parameters,
        seed=113_000 + replication,
        subsample_grid=C_SOF_SUBSAMPLE_RATIO_GRID,
        depth_grid=C_SOF_MAX_DEPTH_GRID,
        num_trees=C_PROTOTYPE_SOF_NUM_TREES,
        n_thresholds=C_PROTOTYPE_SOF_N_THRESHOLDS,
    )
    dec_sof = fit_tuned_allocation_sof_transform(
        x_train_std,
        rewards_train,
        x_val_std,
        rewards_val,
        mean_rewards_val,
        x_test_std,
        rewards_test,
        mean_rewards_test,
        transform_kind="rotated_2d",
        transform_payload=decision_parameters,
        seed=114_000 + replication,
        subsample_grid=C_SOF_SUBSAMPLE_RATIO_GRID,
        depth_grid=C_SOF_MAX_DEPTH_GRID,
        num_trees=C_PROTOTYPE_SOF_NUM_TREES,
        n_thresholds=C_PROTOTYPE_SOF_N_THRESHOLDS,
    )

    method_results = {
        "Kernel-fixed": {"metrics": kernel_fixed_metrics, "decisions": kernel_fixed_decisions},
        "Kernel-dec": {"metrics": kernel_dec_metrics, "decisions": kernel_decisions},
        "Neural-dec": {"metrics": neural_metrics, "decisions": neural_decisions},
        "SOF-raw": raw_sof,
        "SOF-PCA": pca_sof,
        "SOF-LASSO": lasso_sof,
        "SOF-pred": pred_sof,
        "SOF-dec": dec_sof,
    }

    summary_rows = summary_rows_from_method_results("C", replication, method_results)
    decomposition_rows = decomposition_rows_allocation(
        benchmark="C",
        replication=replication,
        raw_sof=raw_sof,
        dec_sof=dec_sof,
        kernel_dec_metrics=kernel_dec_metrics,
        rewards_train=rewards_train,
        rewards_val=rewards_val,
        rewards_test=rewards_test,
        mean_rewards_test=mean_rewards_test,
    )

    boundary_curve = pd.DataFrame()
    if replication == 0:
        oracle_action = np.argmax(mean_rewards_test, axis=1)
        bins = np.linspace(signal_test.min(), signal_test.max(), 25)
        boundary_rows: list[dict[str, float]] = []
        for left, right in zip(bins[:-1], bins[1:]):
            mask = (signal_test >= left) & (signal_test < right)
            if not mask.any():
                continue
            boundary_rows.append(
                {
                    "u_center": float((left + right) / 2.0),
                    "oracle_choose_project_0": float(np.mean(oracle_action[mask] == 0)),
                    "kernel_fixed_choose_project_0": float(np.mean(kernel_fixed_decisions[mask] == 0)),
                    "kernel_dec_choose_project_0": float(np.mean(kernel_decisions[mask] == 0)),
                    "neural_dec_choose_project_0": float(np.mean(neural_decisions[mask] == 0)),
                    "sof_raw_choose_project_0": float(np.mean(raw_sof["decisions"][mask] == 0)),
                    "sof_dec_choose_project_0": float(np.mean(dec_sof["decisions"][mask] == 0)),
                }
            )
        boundary_curve = pd.DataFrame(boundary_rows)

    return summary_rows, decomposition_rows, boundary_curve


def run_benchmark_a_plugin_replication(
    d: int,
    k: int,
    n_train: int,
    replication: int,
    existing_main_row: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    seed = benchmark_a_seed(d, k, n_train, replication)
    rng = np.random.default_rng(seed)
    n_val = max(100, n_train // 2)
    x_train, y_train, _ = generate_benchmark_a(rng, n_train, d=d, k=k)
    x_val, y_val, mean_val = generate_benchmark_a(rng, n_val, d=d, k=k)
    x_test, y_test, mean_test = generate_benchmark_a(rng, 2000, d=d, k=k)

    x_train_std, x_val_std = standardize(x_train, x_val)
    _, x_test_std = standardize(x_train, x_test)
    oracle_orders = mean_test + 4.0 * 0.8416212335729143

    lasso_selected = restore_lasso_selected_from_row(existing_main_row, d)
    if len(lasso_selected) == 0:
        lasso_selected, _, _ = select_lasso_features(x_train_std, y_train, x_val_std, y_val)
    lasso_projection = projection_from_selected_features(d, lasso_selected)

    prediction_scales = restore_screened_diagonal_scales(existing_main_row, "prediction", d)
    learned_scales = restore_screened_diagonal_scales(existing_main_row, "learned", d)
    if prediction_scales is None or learned_scales is None:
        from experiment_suite import learn_screened_diagonal_metric, select_linear_features

        selected = select_linear_features(x_train_std, y_train, top_k=min(10, d))
        _, prediction_scales, _ = learn_screened_diagonal_metric(
            x_train_std,
            y_train,
            objective_kind="prediction",
            selected_features=selected,
            fit_size=min(80, n_train),
            top_k=min(10, d),
            reg=0.02,
            maxfev=90,
        )
        _, learned_scales, _ = learn_screened_diagonal_metric(
            x_train_std,
            y_train,
            objective_kind="decision",
            selected_features=selected,
            initial_scales=prediction_scales,
            fit_size=min(80, n_train),
            top_k=min(10, d),
            reg=0.02,
            maxfev=90,
        )

    method_results = {
        "SOF-raw": fit_tuned_official_sof_transform(
            x_train_std,
            y_train,
            x_val_std,
            y_val,
            mean_val,
            x_test_std,
            y_test,
            mean_test,
            oracle_orders,
            transform_kind="identity",
            transform_payload=None,
            seed=50_000 + seed,
            subsample_grid=A_SOF_SUBSAMPLE_RATIO_GRID,
            depth_grid=A_SOF_MAX_DEPTH_GRID,
            num_trees=A_OFFICIAL_SOF_NUM_TREES,
            n_proposals=A_OFFICIAL_SOF_N_PROPOSALS,
            mtry_mode=A_OFFICIAL_SOF_MTRY_MODE,
            bootstrap=A_OFFICIAL_SOF_BOOTSTRAP,
        ),
        "SOF-LASSO": fit_tuned_official_sof_transform(
            x_train_std,
            y_train,
            x_val_std,
            y_val,
            mean_val,
            x_test_std,
            y_test,
            mean_test,
            oracle_orders,
            transform_kind="linear_projection",
            transform_payload=lasso_projection,
            seed=60_000 + seed,
            subsample_grid=A_SOF_SUBSAMPLE_RATIO_GRID,
            depth_grid=A_SOF_MAX_DEPTH_GRID,
            num_trees=A_OFFICIAL_SOF_NUM_TREES,
            n_proposals=A_OFFICIAL_SOF_N_PROPOSALS,
            mtry_mode=A_OFFICIAL_SOF_MTRY_MODE,
            bootstrap=A_OFFICIAL_SOF_BOOTSTRAP,
        ),
        "SOF-pred": fit_tuned_official_sof_transform(
            x_train_std,
            y_train,
            x_val_std,
            y_val,
            mean_val,
            x_test_std,
            y_test,
            mean_test,
            oracle_orders,
            transform_kind="screened_diagonal",
            transform_payload=prediction_scales,
            seed=70_000 + seed,
            subsample_grid=A_SOF_SUBSAMPLE_RATIO_GRID,
            depth_grid=A_SOF_MAX_DEPTH_GRID,
            num_trees=A_OFFICIAL_SOF_NUM_TREES,
            n_proposals=A_OFFICIAL_SOF_N_PROPOSALS,
            mtry_mode=A_OFFICIAL_SOF_MTRY_MODE,
            bootstrap=A_OFFICIAL_SOF_BOOTSTRAP,
        ),
        "SOF-dec": fit_tuned_official_sof_transform(
            x_train_std,
            y_train,
            x_val_std,
            y_val,
            mean_val,
            x_test_std,
            y_test,
            mean_test,
            oracle_orders,
            transform_kind="screened_diagonal",
            transform_payload=learned_scales,
            seed=80_000 + seed,
            subsample_grid=A_SOF_SUBSAMPLE_RATIO_GRID,
            depth_grid=A_SOF_MAX_DEPTH_GRID,
            num_trees=A_OFFICIAL_SOF_NUM_TREES,
            n_proposals=A_OFFICIAL_SOF_N_PROPOSALS,
            mtry_mode=A_OFFICIAL_SOF_MTRY_MODE,
            bootstrap=A_OFFICIAL_SOF_BOOTSTRAP,
        ),
    }

    rows = []
    for method, payload in method_results.items():
        rows.append(
            {
                "benchmark": "A",
                "d": d,
                "k": k,
                "n_train": n_train,
                "replication": replication,
                "method": method,
                "regret": float(payload["metrics"]["regret"]),
                "cost": float(payload["metrics"]["cost"]),
                "pred_mse": float(payload["metrics"]["pred_mse"]),
                "avg_leaf_support": float(payload["metrics"]["avg_leaf_support"]),
                "subsample_ratio": float(payload["metrics"]["subsample_ratio"]),
                "max_depth": float(payload["metrics"]["max_depth"]),
                "validation_cost": float(payload["metrics"]["validation_cost"]),
            }
        )
    return rows


def _frame_complete(
    lookup: dict[tuple[Any, ...], dict[str, Any]],
    replication_key: tuple[Any, ...],
    methods: list[str],
    required_columns: list[str],
) -> bool:
    for method in methods:
        key = (*replication_key, method)
        if not row_has_columns(lookup.get(key), required_columns):
            return False
    return True


def run_benchmark_b_plugin(
    existing_summary: pd.DataFrame | None = None,
    existing_decomposition: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_lookup = build_mixed_row_lookup(existing_summary, ("replication", "method"))
    decomposition_lookup = build_mixed_row_lookup(existing_decomposition, ("replication", "method"))
    pending_replications = []
    for replication in range(BENCHMARK_B_REPLICATIONS):
        summary_ready = _frame_complete(summary_lookup, (replication,), SOF_PLUGIN_METHOD_ORDER, ["regret", "pred_mse"])
        decomp_ready = _frame_complete(
            decomposition_lookup,
            (replication,),
            SOF_DECOMPOSITION_METHOD_ORDER,
            ["regret"],
        )
        if not (summary_ready and decomp_ready):
            pending_replications.append(replication)

    _start_progress("B", len(pending_replications))
    summary_rows = [] if existing_summary is None else existing_summary.to_dict("records")
    decomposition_rows = [] if existing_decomposition is None else existing_decomposition.to_dict("records")
    if pending_replications:
        with _tqdm_joblib("Benchmark B", len(pending_replications)):
            computed = Parallel(
                n_jobs=DEFAULT_PARALLEL_JOBS,
                verbose=0,
                backend=PARALLEL_BACKEND,
            )(
                delayed(run_benchmark_b_plugin_replication)(replication)
                for replication in pending_replications
            )
        for replication_summary, replication_decomposition in computed:
            summary_rows.extend(replication_summary)
            decomposition_rows.extend(replication_decomposition)
    _log("[B] summary and decomposition frames ready")

    summary_frame = pd.DataFrame(summary_rows).drop_duplicates(subset=["replication", "method"], keep="last")
    decomposition_frame = pd.DataFrame(decomposition_rows).drop_duplicates(subset=["replication", "method"], keep="last")
    return (
        summary_frame.sort_values(["replication", "method"]).reset_index(drop=True),
        decomposition_frame.sort_values(["replication", "method"]).reset_index(drop=True),
    )


def run_benchmark_c_plugin(
    existing_summary: pd.DataFrame | None = None,
    existing_decomposition: pd.DataFrame | None = None,
    existing_boundary: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary_lookup = build_mixed_row_lookup(existing_summary, ("replication", "method"))
    decomposition_lookup = build_mixed_row_lookup(existing_decomposition, ("replication", "method"))
    boundary_ready = existing_boundary is not None and not existing_boundary.empty and "sof_dec_choose_project_0" in existing_boundary.columns

    pending_replications = []
    for replication in range(BENCHMARK_C_REPLICATIONS):
        summary_ready = _frame_complete(summary_lookup, (replication,), SOF_PLUGIN_METHOD_ORDER, ["regret", "pred_mse"])
        decomp_ready = _frame_complete(
            decomposition_lookup,
            (replication,),
            SOF_DECOMPOSITION_METHOD_ORDER,
            ["regret"],
        )
        if not (summary_ready and decomp_ready and (replication != 0 or boundary_ready)):
            pending_replications.append(replication)

    _start_progress("C", len(pending_replications))
    summary_rows = [] if existing_summary is None else existing_summary.to_dict("records")
    decomposition_rows = [] if existing_decomposition is None else existing_decomposition.to_dict("records")
    boundary_curve = pd.DataFrame() if existing_boundary is None else existing_boundary.copy()
    if pending_replications:
        with _tqdm_joblib("Benchmark C", len(pending_replications)):
            computed = Parallel(
                n_jobs=DEFAULT_PARALLEL_JOBS,
                verbose=0,
                backend=PARALLEL_BACKEND,
            )(
                delayed(run_benchmark_c_plugin_replication)(replication)
                for replication in pending_replications
            )
        for replication_summary, replication_decomposition, replication_boundary in computed:
            summary_rows.extend(replication_summary)
            decomposition_rows.extend(replication_decomposition)
            if not replication_boundary.empty:
                boundary_curve = replication_boundary
    _log("[C] summary, decomposition, and boundary frames ready")

    summary_frame = pd.DataFrame(summary_rows).drop_duplicates(subset=["replication", "method"], keep="last")
    decomposition_frame = pd.DataFrame(decomposition_rows).drop_duplicates(subset=["replication", "method"], keep="last")
    return (
        summary_frame.sort_values(["replication", "method"]).reset_index(drop=True),
        decomposition_frame.sort_values(["replication", "method"]).reset_index(drop=True),
        boundary_curve,
    )


def run_benchmark_a_plugin(
    existing_plugin: pd.DataFrame | None = None,
    existing_main: pd.DataFrame | None = None,
) -> pd.DataFrame:
    plugin_lookup = build_mixed_row_lookup(existing_plugin, ("d", "k", "n_train", "replication", "method"))
    main_lookup = build_mixed_row_lookup(existing_main, ("d", "k", "n_train", "replication"))

    pending_tasks = []
    for d, k in BENCHMARK_A_CONFIGS:
        for n_train in BENCHMARK_A_TRAIN_SIZES:
            for replication in range(BENCHMARK_A_REPLICATIONS):
                complete = _frame_complete(
                    plugin_lookup,
                    (d, k, n_train, replication),
                    ["SOF-raw", "SOF-LASSO", "SOF-pred", "SOF-dec"],
                    ["regret", "pred_mse"],
                )
                if not complete:
                    pending_tasks.append((d, k, n_train, replication, main_lookup.get((d, k, n_train, replication))))

    _start_progress("A", len(pending_tasks))
    rows = [] if existing_plugin is None else existing_plugin.to_dict("records")
    if pending_tasks:
        with _tqdm_joblib("Benchmark A", len(pending_tasks)):
            computed = Parallel(
                n_jobs=DEFAULT_PARALLEL_JOBS,
                verbose=0,
                backend=PARALLEL_BACKEND,
            )(
                delayed(run_benchmark_a_plugin_replication)(d, k, n_train, replication, existing_main_row)
                for d, k, n_train, replication, existing_main_row in pending_tasks
            )
        for replication_rows in computed:
            rows.extend(replication_rows)
    _log("[A] plugin SOF frame ready")

    frame = pd.DataFrame(rows).drop_duplicates(subset=["d", "k", "n_train", "replication", "method"], keep="last")
    return frame.sort_values(["d", "k", "n_train", "replication", "method"]).reset_index(drop=True)


def summarize_plugin_benchmark(frame: pd.DataFrame, benchmark: str) -> pd.DataFrame:
    subset = frame[frame["benchmark"] == benchmark].copy()
    raw_regret = subset[subset["method"] == "SOF-raw"][["replication", "regret"]].rename(columns={"regret": "raw_regret"})
    merged = subset.merge(raw_regret, on="replication", how="left")
    merged["win_vs_raw"] = (merged["regret"] < merged["raw_regret"] - 1e-12).astype(int)
    summary = (
        merged.groupby("method")[["regret", "pred_mse", "win_vs_raw"]]
        .agg(avg_regret=("regret", "mean"), avg_pred_mse=("pred_mse", "mean"), win_count=("win_vs_raw", "sum"))
        .reset_index()
    )
    order = {method: index for index, method in enumerate(SOF_PLUGIN_METHOD_ORDER)}
    return summary.sort_values("method", key=lambda series: series.map(order)).reset_index(drop=True)


def build_cross_benchmark_summary(
    frame_b: pd.DataFrame,
    frame_c: pd.DataFrame,
) -> pd.DataFrame:
    summary_b = summarize_plugin_benchmark(frame_b, "B").rename(
        columns={
            "avg_regret": "benchmark_b_regret",
            "avg_pred_mse": "benchmark_b_pred_mse",
            "win_count": "benchmark_b_win_count",
        }
    )
    summary_c = summarize_plugin_benchmark(frame_c, "C").rename(
        columns={
            "avg_regret": "benchmark_c_regret",
            "avg_pred_mse": "benchmark_c_pred_mse",
            "win_count": "benchmark_c_win_count",
        }
    )
    merged = summary_b.merge(summary_c, on="method", how="outer")
    order = {method: index for index, method in enumerate(SOF_PLUGIN_METHOD_ORDER)}
    return merged.sort_values("method", key=lambda series: series.map(order)).reset_index(drop=True)


def summarize_decomposition(frame: pd.DataFrame) -> pd.DataFrame:
    summary = (
        frame.groupby(["benchmark", "method"])[["regret", "pred_mse"]]
        .mean()
        .reset_index()
    )
    order = {method: index for index, method in enumerate(SOF_DECOMPOSITION_METHOD_ORDER)}
    return summary.sort_values(["benchmark", "method"], key=lambda series: series.map(order) if series.name == "method" else series).reset_index(drop=True)


def benchmark_a_plugin_exponents(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (d, k, method), group in frame.groupby(["d", "k", "method"]):
        summary = group.groupby("n_train")["regret"].mean().reset_index()
        log_n = np.log(summary["n_train"].to_numpy(dtype=float))
        log_regret = np.log(np.clip(summary["regret"].to_numpy(dtype=float), 1e-8, None))
        slope, intercept = np.polyfit(log_n, log_regret, 1)
        rows.append(
            {
                "d": int(d),
                "k": int(k),
                "method": method,
                "slope": float(slope),
                "effective_exponent": float(-slope),
                "intercept": float(intercept),
            }
        )
    order = {method: index for index, method in enumerate(["SOF-raw", "SOF-LASSO", "SOF-pred", "SOF-dec"])}
    return pd.DataFrame(rows).sort_values(["d", "k", "method"], key=lambda series: series.map(order) if series.name == "method" else series).reset_index(drop=True)


def benchmark_a_plugin_curves(frame: pd.DataFrame) -> pd.DataFrame:
    return (
        frame.groupby(["d", "k", "n_train", "method"])["regret"]
        .mean()
        .reset_index()
        .sort_values(["d", "k", "method", "n_train"])
        .reset_index(drop=True)
    )
