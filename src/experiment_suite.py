"""Core synthetic benchmark code for the decision-focused kernel report.

The module is organized in layers:

1. shared weighted-SAA, kernel, and SOF utility functions;
2. representation learners for prediction-trained and decision-trained metrics;
3. benchmark runners for the three report settings.

The public scripts call the benchmark runners near the bottom of this file.
Those runners are cache-aware so an interrupted long run can be resumed from
the CSV files in ``reports/assets``.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import types
from functools import partial
from math import cos, pi, sin
from pathlib import Path
from statistics import NormalDist
from typing import Any

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.optimize import minimize
from sklearn.decomposition import PCA
from sklearn.linear_model import Lasso


UNDERAGE_COST = 4.0
OVERAGE_COST = 1.0
ALPHA = UNDERAGE_COST / (UNDERAGE_COST + OVERAGE_COST)
ORACLE_SHIFT_UNIT = NormalDist().inv_cdf(ALPHA)
BANDWIDTH_GRID = np.array([0.20, 0.35, 0.50, 0.75, 1.00, 1.50, 2.00])
LASSO_ALPHA_GRID = np.geomspace(1e-3, 1.0, 12)
SPECTRAL_NUM_COMPONENTS = 2
NEURAL_HIDDEN_DIM = 3
OFFICIAL_SOF_NUM_TREES = 500
OFFICIAL_SOF_VARIANT = "approx_sol"
BENCHMARK_PARALLEL_JOBS = min(12, os.cpu_count() or 1)

_OFFICIAL_SOF_NV_MODULE: Any | None = None


def build_existing_row_lookup(
    frame: pd.DataFrame | None,
    key_columns: tuple[str, ...],
) -> dict[tuple[int, ...], dict[str, Any]]:
    if frame is None or frame.empty:
        return {}
    lookup: dict[tuple[int, ...], dict[str, Any]] = {}
    for _, row in frame.iterrows():
        key = tuple(int(row[column]) for column in key_columns)
        lookup[key] = row.to_dict()
    return lookup


def row_has_columns(
    row: dict[str, Any] | None,
    columns: list[str],
) -> bool:
    if row is None:
        return False
    for column in columns:
        if column not in row:
            return False
        value = row[column]
        if isinstance(value, str):
            continue
        if pd.isna(value):
            return False
    return True


def normalized_profile(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    total = float(array.sum())
    if total <= 0.0:
        return np.full(len(array), 1.0 / len(array))
    return array / total


def lift_profile_to_dimension(
    selected_features: np.ndarray,
    profile: np.ndarray,
    dimension: int,
) -> np.ndarray:
    lifted = np.zeros(dimension, dtype=float)
    selected = np.asarray(selected_features, dtype=int)
    if len(selected) == 0:
        return normalized_profile(np.ones(dimension, dtype=float))
    lifted[selected] = profile
    return lifted


def standardize(
    x_train: np.ndarray,
    x_other: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0)
    std[std == 0.0] = 1.0
    return (x_train - mean) / std, (x_other - mean) / std


def fit_pca_projection(
    x_train: np.ndarray,
    n_components: int,
) -> np.ndarray:
    n_components = min(n_components, x_train.shape[0], x_train.shape[1])
    pca = PCA(n_components=n_components)
    pca.fit(x_train)
    return pca.components_


def projection_from_selected_features(
    dimension: int,
    selected_features: np.ndarray,
) -> np.ndarray:
    selected = np.sort(np.unique(np.asarray(selected_features, dtype=int)))
    if len(selected) == 0:
        raise ValueError("selected_features must be non-empty")
    return np.eye(dimension)[selected]


def select_lasso_features(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
) -> tuple[np.ndarray, float, float]:
    best_alpha = float(LASSO_ALPHA_GRID[0])
    best_mse = float("inf")
    best_coef = np.zeros(x_train.shape[1])

    for alpha in LASSO_ALPHA_GRID:
        model = Lasso(alpha=float(alpha), fit_intercept=True, max_iter=20000)
        model.fit(x_train, y_train)
        prediction = model.predict(x_val)
        mse = float(np.mean((prediction - y_val) ** 2))
        if mse < best_mse:
            best_alpha = float(alpha)
            best_mse = mse
            best_coef = np.asarray(model.coef_, dtype=float)

    selected = np.flatnonzero(np.abs(best_coef) > 1e-8)
    if len(selected) == 0:
        centered_y = y_train - y_train.mean()
        association = np.abs((x_train * centered_y[:, None]).mean(axis=0))
        selected = np.array([int(np.argmax(association))])

    return np.sort(selected), best_alpha, best_mse


def select_multitarget_lasso_features(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
) -> tuple[np.ndarray, float, float]:
    if y_train.ndim != 2 or y_val.ndim != 2:
        raise ValueError("select_multitarget_lasso_features expects 2D response arrays")

    best_alpha = float(LASSO_ALPHA_GRID[0])
    best_mse = float("inf")
    best_coef = np.zeros((y_train.shape[1], x_train.shape[1]))

    for alpha in LASSO_ALPHA_GRID:
        predictions = np.zeros_like(y_val, dtype=float)
        coefficients = np.zeros_like(best_coef)
        for target_index in range(y_train.shape[1]):
            model = Lasso(alpha=float(alpha), fit_intercept=True, max_iter=20000)
            model.fit(x_train, y_train[:, target_index])
            predictions[:, target_index] = model.predict(x_val)
            coefficients[target_index] = np.asarray(model.coef_, dtype=float)
        mse = float(np.mean((predictions - y_val) ** 2))
        if mse < best_mse:
            best_alpha = float(alpha)
            best_mse = mse
            best_coef = coefficients

    selected = np.flatnonzero(np.any(np.abs(best_coef) > 1e-8, axis=0))
    if len(selected) == 0:
        centered_y = y_train - y_train.mean(axis=0, keepdims=True)
        association = np.abs(x_train.T @ centered_y) / len(x_train)
        selected = np.array([int(np.argmax(association.mean(axis=1)))])

    return np.sort(selected), best_alpha, best_mse


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
    pairwise = np.sqrt(pairwise_squared_distances(sample, sample))
    upper = pairwise[np.triu_indices_from(pairwise, k=1)]
    median_distance = float(np.median(upper))
    if not np.isfinite(median_distance) or median_distance <= 0.0:
        return 1.0
    return median_distance


def bandwidth_candidates(x_train: np.ndarray) -> np.ndarray:
    return distance_scale(x_train) * BANDWIDTH_GRID


def kernel_weight_matrix(
    squared_distances: np.ndarray,
    bandwidth: float,
    exclude_self: bool = False,
) -> np.ndarray:
    weights = np.exp(-squared_distances / (2.0 * bandwidth**2))
    if exclude_self:
        diagonal = min(weights.shape[0], weights.shape[1])
        row_index = np.arange(diagonal)
        weights[row_index, row_index] = 0.0
    return weights


def kernel_average_from_squared_distances(
    squared_distances: np.ndarray,
    responses_train: np.ndarray,
    bandwidth: float,
    exclude_self: bool = False,
) -> np.ndarray:
    weights = kernel_weight_matrix(squared_distances, bandwidth, exclude_self=exclude_self)
    if responses_train.ndim == 1:
        responses_train = responses_train[:, None]
        squeeze_output = True
    else:
        squeeze_output = False

    totals = weights.sum(axis=1, keepdims=True)
    empty_rows = totals[:, 0] <= 0.0
    if np.any(empty_rows):
        weights[empty_rows] = 1.0
        if exclude_self and weights.shape[0] == weights.shape[1]:
            diagonal = min(weights.shape[0], weights.shape[1])
            row_index = np.arange(diagonal)
            weights[row_index, row_index] = 0.0
        totals = weights.sum(axis=1, keepdims=True)

    predictions = weights @ responses_train / totals
    if squeeze_output:
        return predictions[:, 0]
    return predictions


def kernel_average_from_weights(
    weights: np.ndarray,
    responses_train: np.ndarray,
    exclude_self: bool = False,
) -> np.ndarray:
    if responses_train.ndim == 1:
        responses_train = responses_train[:, None]
        squeeze_output = True
    else:
        squeeze_output = False

    weights = weights.copy()
    totals = weights.sum(axis=1, keepdims=True)
    empty_rows = totals[:, 0] <= 0.0
    if np.any(empty_rows):
        weights[empty_rows] = 1.0
        if exclude_self and weights.shape[0] == weights.shape[1]:
            diagonal = min(weights.shape[0], weights.shape[1])
            row_index = np.arange(diagonal)
            weights[row_index, row_index] = 0.0
        totals = weights.sum(axis=1, keepdims=True)

    predictions = weights @ responses_train / totals
    if squeeze_output:
        return predictions[:, 0]
    return predictions


def newsvendor_cost(
    order_quantity: np.ndarray,
    demand: np.ndarray,
    underage_cost: float = UNDERAGE_COST,
    overage_cost: float = OVERAGE_COST,
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
    alpha: float = ALPHA,
) -> float:
    total_weight = weights_sorted.sum()
    if total_weight <= 0.0:
        weights_sorted = np.ones_like(y_sorted, dtype=float)
        total_weight = weights_sorted.sum()

    cumulative = np.cumsum(weights_sorted / total_weight)
    index = int(np.searchsorted(cumulative, alpha, side="left"))
    return float(y_sorted[min(index, len(y_sorted) - 1)])


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


def orders_from_weights(
    weights: np.ndarray,
    y_sorted: np.ndarray,
    y_order: np.ndarray,
    alpha: float,
) -> np.ndarray:
    orders = np.empty(weights.shape[0])
    for row_index in range(weights.shape[0]):
        orders[row_index] = weighted_quantile_from_sorted(y_sorted, weights[row_index][y_order], alpha)
    return orders


def unpack_spectral_mixture_parameters(
    parameters: np.ndarray,
    dimension: int,
    num_components: int = SPECTRAL_NUM_COMPONENTS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    offset = 0
    raw_weights = parameters[offset : offset + num_components]
    offset += num_components
    raw_variances = parameters[offset : offset + num_components * dimension].reshape(num_components, dimension)
    offset += num_components * dimension
    frequencies = parameters[offset : offset + num_components * dimension].reshape(num_components, dimension)

    centered_weights = raw_weights - raw_weights.max()
    mixture_weights = np.exp(centered_weights)
    variances = np.exp(raw_variances)
    return mixture_weights, variances, frequencies


def spectral_mixture_weight_matrix(
    x_query: np.ndarray,
    x_train: np.ndarray,
    parameters: np.ndarray,
    bandwidth: float,
    exclude_self: bool = False,
) -> np.ndarray:
    mixture_weights, variances, frequencies = unpack_spectral_mixture_parameters(
        parameters,
        x_train.shape[1],
    )
    tau = (x_query[:, None, :] - x_train[None, :, :]) / max(float(bandwidth), 1e-8)
    weights = np.zeros((len(x_query), len(x_train)))
    for component_weight, component_variance, component_frequency in zip(
        mixture_weights,
        variances,
        frequencies,
    ):
        gaussian_envelope = np.exp(-2.0 * pi**2 * np.sum(tau * tau * component_variance[None, None, :], axis=2))
        periodic_factor = np.prod((1.0 + np.cos(2.0 * pi * tau * component_frequency[None, None, :])) / 2.0, axis=2)
        weights += component_weight * gaussian_envelope * periodic_factor

    if exclude_self and weights.shape[0] == weights.shape[1]:
        diagonal = min(weights.shape[0], weights.shape[1])
        row_index = np.arange(diagonal)
        weights[row_index, row_index] = 0.0
    return weights


def spectral_mixture_metadata(parameters: np.ndarray, dimension: int) -> dict[str, float]:
    mixture_weights, _, frequencies = unpack_spectral_mixture_parameters(parameters, dimension)
    normalized_weights = mixture_weights / max(mixture_weights.sum(), 1e-12)
    frequency_norms = np.linalg.norm(frequencies, axis=1)
    return {
        "spectral_dominant_weight": float(normalized_weights.max()),
        "spectral_avg_frequency_norm": float(frequency_norms.mean()),
    }


def unpack_neural_embedding_parameters(
    parameters: np.ndarray,
    input_dim: int,
    hidden_dim: int = NEURAL_HIDDEN_DIM,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    offset = 0
    w1 = parameters[offset : offset + hidden_dim * input_dim].reshape(hidden_dim, input_dim)
    offset += hidden_dim * input_dim
    b1 = parameters[offset : offset + hidden_dim]
    offset += hidden_dim
    w2 = parameters[offset : offset + input_dim * hidden_dim].reshape(input_dim, hidden_dim)
    offset += input_dim * hidden_dim
    b2 = parameters[offset : offset + input_dim]
    return w1, b1, w2, b2


def neural_embedding_parameter_count(
    input_dim: int,
    hidden_dim: int = NEURAL_HIDDEN_DIM,
) -> int:
    return hidden_dim * input_dim + hidden_dim + input_dim * hidden_dim + input_dim


def neural_embedding_transform(
    x: np.ndarray,
    parameters: np.ndarray,
    hidden_dim: int = NEURAL_HIDDEN_DIM,
) -> np.ndarray:
    input_dim = x.shape[1]
    w1, b1, w2, b2 = unpack_neural_embedding_parameters(parameters, input_dim, hidden_dim=hidden_dim)
    hidden = np.tanh(x @ w1.T + b1)
    return x + hidden @ w2.T + b2


def neural_embedding_initial_parameters(
    input_dim: int,
    hidden_dim: int = NEURAL_HIDDEN_DIM,
) -> np.ndarray:
    return np.zeros(neural_embedding_parameter_count(input_dim, hidden_dim=hidden_dim))


def tune_bandwidth_newsvendor(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    underage_cost: float = UNDERAGE_COST,
    overage_cost: float = OVERAGE_COST,
) -> float:
    y_sorted, y_order = sorted_response(y_train)
    candidates = bandwidth_candidates(x_train)
    squared_distances = pairwise_squared_distances(x_val, x_train)

    best_bandwidth = float(candidates[0])
    best_cost = float("inf")
    for bandwidth in candidates:
        candidate_orders = orders_from_squared_distances(
            squared_distances,
            y_sorted,
            y_order,
            ALPHA,
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


def tune_bandwidth_scalar_prediction(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    mean_val: np.ndarray,
) -> float:
    candidates = bandwidth_candidates(x_train)
    squared_distances = pairwise_squared_distances(x_val, x_train)

    best_bandwidth = float(candidates[0])
    best_mse = float("inf")
    for bandwidth in candidates:
        prediction = kernel_average_from_squared_distances(
            squared_distances,
            y_train,
            float(bandwidth),
        )
        mse = float(np.mean((prediction - mean_val) ** 2))
        if mse < best_mse:
            best_mse = mse
            best_bandwidth = float(bandwidth)
    return best_bandwidth


def choose_weighted_action(
    weights: np.ndarray,
    rewards_train: np.ndarray,
) -> int:
    total_weight = weights.sum()
    if total_weight <= 0.0:
        weights = np.ones(len(rewards_train), dtype=float)
        total_weight = weights.sum()
    weighted_rewards = weights @ rewards_train / total_weight
    return int(np.argmax(weighted_rewards))


def empirical_newsvendor_solution_and_risk(
    y: np.ndarray,
    alpha: float = ALPHA,
) -> tuple[float, float]:
    order = float(np.quantile(y, alpha))
    risk = float(newsvendor_cost(np.full(len(y), order), y).mean())
    return order, risk


def empirical_allocation_solution_and_risk(
    rewards: np.ndarray,
) -> tuple[int, float]:
    action = int(np.argmax(rewards.mean(axis=0)))
    risk = float(np.mean(-rewards[:, action]))
    return action, risk


def candidate_thresholds(
    values: np.ndarray,
    n_thresholds: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if len(values) <= 1:
        return np.array([])
    unique_values = np.unique(values)
    if len(unique_values) <= 1:
        return np.array([])
    thresholds = (unique_values[:-1] + unique_values[1:]) / 2.0
    if len(thresholds) <= n_thresholds:
        return thresholds
    selected = rng.choice(len(thresholds), size=n_thresholds, replace=False)
    return np.sort(thresholds[selected])


def build_stochopt_tree(
    x_train: np.ndarray,
    response_train: np.ndarray,
    task_kind: str,
    rng: np.random.Generator,
    max_depth: int = 4,
    min_leaf_size: int = 15,
    mtry: int | None = None,
    n_thresholds: int = 8,
    balancedness_tol: float = 0.2,
) -> dict[str, Any]:
    dimension = x_train.shape[1]
    if mtry is None:
        mtry = min(dimension, max(1, int(np.ceil(np.sqrt(dimension)))))

    def leaf_risk(indices: np.ndarray) -> float:
        if task_kind == "newsvendor":
            _, risk = empirical_newsvendor_solution_and_risk(response_train[indices])
            return risk
        if task_kind == "allocation":
            _, risk = empirical_allocation_solution_and_risk(response_train[indices])
            return risk
        raise ValueError(f"Unknown task kind: {task_kind}")

    def recurse(indices: np.ndarray, depth: int) -> dict[str, Any]:
        node = {"indices": indices, "feature": None, "threshold": None, "left": None, "right": None}
        if depth >= max_depth or len(indices) < 2 * min_leaf_size:
            return node

        best_score = float("inf")
        best_feature = None
        best_threshold = None
        best_left = None
        best_right = None
        features = rng.choice(dimension, size=min(mtry, dimension), replace=False)
        node_size = len(indices)
        lower_bound = max(min_leaf_size, int(np.ceil(balancedness_tol * node_size)))
        upper_bound = min(node_size - min_leaf_size, int(np.floor((1.0 - balancedness_tol) * node_size)))
        if lower_bound > upper_bound:
            return node

        for feature in features:
            values = x_train[indices, feature]
            thresholds = candidate_thresholds(values, n_thresholds=n_thresholds, rng=rng)
            for threshold in thresholds:
                side = values < threshold
                left_indices = indices[side]
                right_indices = indices[~side]
                if len(left_indices) < lower_bound or len(left_indices) > upper_bound:
                    continue
                if len(right_indices) < lower_bound or len(right_indices) > upper_bound:
                    continue
                left_risk = leaf_risk(left_indices)
                right_risk = leaf_risk(right_indices)
                score = (len(left_indices) * left_risk + len(right_indices) * right_risk) / node_size
                if score < best_score:
                    best_score = score
                    best_feature = int(feature)
                    best_threshold = float(threshold)
                    best_left = left_indices
                    best_right = right_indices

        if best_feature is None:
            return node

        node["feature"] = best_feature
        node["threshold"] = best_threshold
        node["left"] = recurse(best_left, depth + 1)
        node["right"] = recurse(best_right, depth + 1)
        return node

    return recurse(np.arange(len(x_train)), 0)


def find_stochopt_leaf(
    tree: dict[str, Any],
    x_query: np.ndarray,
) -> np.ndarray:
    node = tree
    while node["feature"] is not None:
        if x_query[node["feature"]] < node["threshold"]:
            node = node["left"]
        else:
            node = node["right"]
    if "leaf_est_indices" in node and len(node["leaf_est_indices"]) > 0:
        return np.asarray(node["leaf_est_indices"], dtype=int)
    return np.asarray(node["indices"], dtype=int)


def initialize_leaf_estimation_indices(
    tree: dict[str, Any],
) -> None:
    if tree["feature"] is None:
        tree["leaf_est_indices"] = []
        return
    initialize_leaf_estimation_indices(tree["left"])
    initialize_leaf_estimation_indices(tree["right"])


def assign_estimation_indices_to_tree(
    tree: dict[str, Any],
    x_train: np.ndarray,
    est_indices: np.ndarray,
) -> None:
    initialize_leaf_estimation_indices(tree)
    for index in est_indices:
        node = tree
        query = x_train[index]
        while node["feature"] is not None:
            if query[node["feature"]] < node["threshold"]:
                node = node["left"]
            else:
                node = node["right"]
        node["leaf_est_indices"].append(int(index))


def build_stochopt_forest(
    x_train: np.ndarray,
    response_train: np.ndarray,
    task_kind: str,
    n_trees: int = 40,
    subsample_ratio: float = 0.8,
    max_depth: int = 5,
    min_leaf_size: int = 12,
    mtry: int | None = None,
    n_thresholds: int = 10,
    balancedness_tol: float = 0.2,
    honest: bool = False,
    seed: int = 0,
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    subsample_size = max(2 * min_leaf_size, int(np.ceil(subsample_ratio * len(x_train))))
    trees = []
    for _ in range(n_trees):
        sample_indices = rng.choice(len(x_train), size=min(subsample_size, len(x_train)), replace=False)
        if honest:
            split_point = len(sample_indices) // 2
            grow_indices = sample_indices[:split_point]
            est_indices = sample_indices[split_point:]
            if len(grow_indices) < 2 * min_leaf_size or len(est_indices) < min_leaf_size:
                continue
            tree = build_stochopt_tree(
                x_train[grow_indices],
                response_train[grow_indices],
                task_kind=task_kind,
                rng=rng,
                max_depth=max_depth,
                min_leaf_size=min_leaf_size,
                mtry=mtry,
                n_thresholds=n_thresholds,
                balancedness_tol=balancedness_tol,
            )

            def remap(node: dict[str, Any]) -> dict[str, Any]:
                node = node.copy()
                node["indices"] = grow_indices[node["indices"]]
                if node["left"] is not None:
                    node["left"] = remap(node["left"])
                if node["right"] is not None:
                    node["right"] = remap(node["right"])
                return node

            tree = remap(tree)
            assign_estimation_indices_to_tree(tree, x_train, est_indices)
        else:
            if len(sample_indices) < 2 * min_leaf_size:
                continue
            tree = build_stochopt_tree(
                x_train[sample_indices],
                response_train[sample_indices],
                task_kind=task_kind,
                rng=rng,
                max_depth=max_depth,
                min_leaf_size=min_leaf_size,
                mtry=mtry,
                n_thresholds=n_thresholds,
                balancedness_tol=balancedness_tol,
            )

            def remap(node: dict[str, Any]) -> dict[str, Any]:
                node = node.copy()
                node["indices"] = sample_indices[node["indices"]]
                if node["left"] is not None:
                    node["left"] = remap(node["left"])
                if node["right"] is not None:
                    node["right"] = remap(node["right"])
                return node

            tree = remap(tree)
        trees.append(tree)
    return trees


def stochopt_forest_weight_matrix(
    forest: list[dict[str, Any]],
    x_query: np.ndarray,
    n_train: int,
) -> np.ndarray:
    weights = np.zeros((len(x_query), n_train))
    for tree in forest:
        for row_index, query in enumerate(x_query):
            leaf_indices = find_stochopt_leaf(tree, query)
            leaf_weight = 1.0 / len(leaf_indices)
            weights[row_index, leaf_indices] += leaf_weight
    return weights / max(len(forest), 1)


def tune_bandwidth_allocation(
    x_train: np.ndarray,
    rewards_train: np.ndarray,
    x_val: np.ndarray,
    rewards_val: np.ndarray,
) -> float:
    candidates = bandwidth_candidates(x_train)
    best_bandwidth = float(candidates[0])
    best_cost = float("inf")

    for bandwidth in candidates:
        costs = []
        for x_query, reward in zip(x_val, rewards_val):
            squared = np.sum((x_train - x_query) ** 2, axis=1)
            weights = np.exp(-squared / (2.0 * bandwidth**2))
            action = choose_weighted_action(weights, rewards_train)
            costs.append(-reward[action])
        avg_cost = float(np.mean(costs))
        if avg_cost < best_cost:
            best_cost = avg_cost
            best_bandwidth = float(bandwidth)

    return best_bandwidth


def tune_bandwidth_vector_prediction(
    x_train: np.ndarray,
    rewards_train: np.ndarray,
    x_val: np.ndarray,
    mean_rewards_val: np.ndarray,
) -> float:
    candidates = bandwidth_candidates(x_train)
    squared_distances = pairwise_squared_distances(x_val, x_train)

    best_bandwidth = float(candidates[0])
    best_mse = float("inf")
    for bandwidth in candidates:
        prediction = kernel_average_from_squared_distances(
            squared_distances,
            rewards_train,
            float(bandwidth),
        )
        mse = float(np.mean((prediction - mean_rewards_val) ** 2))
        if mse < best_mse:
            best_mse = mse
            best_bandwidth = float(bandwidth)
    return best_bandwidth


def evaluate_allocation_kernel(
    x_train: np.ndarray,
    rewards_train: np.ndarray,
    x_val: np.ndarray,
    rewards_val: np.ndarray,
    x_test: np.ndarray,
    rewards_test: np.ndarray,
) -> tuple[float, float, np.ndarray]:
    bandwidth = tune_bandwidth_allocation(x_train, rewards_train, x_val, rewards_val)
    decisions = np.empty(len(x_test), dtype=int)
    costs = np.empty(len(x_test))
    for index, (x_query, reward) in enumerate(zip(x_test, rewards_test)):
        squared = np.sum((x_train - x_query) ** 2, axis=1)
        weights = np.exp(-squared / (2.0 * bandwidth**2))
        action = choose_weighted_action(weights, rewards_train)
        decisions[index] = action
        costs[index] = -reward[action]
    return float(costs.mean()), bandwidth, decisions


def select_linear_features(
    x_train: np.ndarray,
    y_train: np.ndarray,
    top_k: int,
) -> np.ndarray:
    design = np.column_stack([np.ones(len(x_train)), x_train])
    coefficients, *_ = np.linalg.lstsq(design, y_train, rcond=None)
    feature_scores = np.abs(coefficients[1:])
    order = np.argsort(feature_scores)[::-1]
    return np.sort(order[: min(top_k, x_train.shape[1])])


def screened_diagonal_scales(
    raw_parameters: np.ndarray,
    dimension: int,
    selected_features: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    raw_full = np.empty(dimension)
    raw_full[selected_features] = raw_parameters[:-1]
    mask = np.ones(dimension, dtype=bool)
    mask[selected_features] = False
    raw_full[mask] = raw_parameters[-1]
    centered = raw_full - raw_full.mean()
    scales = np.exp(centered)
    return centered, scales


def screened_diagonal_initial_parameters(
    x_train: np.ndarray,
    y_train: np.ndarray,
    selected_features: np.ndarray,
    initial_scales: np.ndarray | None = None,
) -> np.ndarray:
    if initial_scales is not None:
        mask = np.ones(len(initial_scales), dtype=bool)
        mask[selected_features] = False
        nuisance_value = float(np.log(np.clip(initial_scales[mask][0], 1e-8, None))) if mask.any() else 0.0
        selected_values = np.log(np.clip(initial_scales[selected_features], 1e-8, None))
        return np.concatenate([selected_values, np.array([nuisance_value])])

    fit_size = len(x_train)
    initial_design = np.column_stack([np.ones(fit_size), x_train[:, selected_features]])
    initial_beta, *_ = np.linalg.lstsq(initial_design, y_train, rcond=None)
    initial_scales = np.log(np.abs(initial_beta[1:]) + 1e-3)
    return np.concatenate([initial_scales, np.array([np.log(0.3)])])


def learn_screened_diagonal_metric(
    x_train: np.ndarray,
    y_train: np.ndarray,
    objective_kind: str,
    selected_features: np.ndarray | None = None,
    initial_scales: np.ndarray | None = None,
    fit_size: int = 80,
    top_k: int = 10,
    reg: float = 0.02,
    maxfev: int = 90,
) -> tuple[np.ndarray, np.ndarray, float]:
    if selected_features is None:
        selected_features = select_linear_features(x_train, y_train, top_k=top_k)
    fit_size = min(len(x_train), fit_size)
    x_metric = x_train[:fit_size]
    y_metric = y_train[:fit_size]
    y_sorted, y_order = sorted_response(y_metric)
    feature_differences = x_metric[:, None, :] - x_metric[None, :, :]
    squared_differences = feature_differences * feature_differences
    initial = screened_diagonal_initial_parameters(
        x_metric,
        y_metric,
        selected_features,
        initial_scales=initial_scales,
    )

    def objective(raw_parameters: np.ndarray) -> float:
        centered, scales = screened_diagonal_scales(
            raw_parameters,
            x_train.shape[1],
            selected_features,
        )
        squared_distances = np.tensordot(squared_differences, scales, axes=([2], [0]))
        if objective_kind == "decision":
            loo_orders = orders_from_squared_distances(
                squared_distances,
                y_sorted,
                y_order,
                ALPHA,
                bandwidth=1.0,
                exclude_self=True,
            )
            loss = newsvendor_cost(loo_orders, y_metric).mean()
        elif objective_kind == "prediction":
            loo_prediction = kernel_average_from_squared_distances(
                squared_distances,
                y_metric,
                bandwidth=1.0,
                exclude_self=True,
            )
            loss = np.mean((loo_prediction - y_metric) ** 2)
        else:
            raise ValueError(f"Unknown objective kind: {objective_kind}")
        return float(loss + reg * np.mean(centered**2))

    result = minimize(
        objective,
        initial,
        method="Powell",
        options={"maxfev": maxfev, "xtol": 1e-2, "ftol": 1e-3},
    )
    _, scales = screened_diagonal_scales(result.x, x_train.shape[1], selected_features)
    return selected_features, scales, float(result.fun)


def transform_screened_diagonal(
    x: np.ndarray,
    scales: np.ndarray,
) -> np.ndarray:
    return x * np.sqrt(scales)


def rotated_metric_transform(
    x: np.ndarray,
    parameters: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    angle, raw_scale_1, raw_scale_2 = parameters
    rotation = np.array(
        [
            [cos(angle), sin(angle)],
            [-sin(angle), cos(angle)],
        ]
    )
    centered = np.array([raw_scale_1, raw_scale_2])
    centered = centered - centered.mean()
    scales = np.exp(centered)
    transformed = (x @ rotation.T) * np.sqrt(scales)
    return transformed, rotation, scales


def learn_rotated_metric_newsvendor(
    x_train: np.ndarray,
    y_train: np.ndarray,
    objective_kind: str,
    initial_parameters: np.ndarray | None = None,
    fit_size: int = 120,
    reg: float = 0.02,
    maxfev: int = 160,
) -> tuple[np.ndarray, float]:
    fit_size = min(len(x_train), fit_size)
    x_metric = x_train[:fit_size]
    y_metric = y_train[:fit_size]
    y_sorted, y_order = sorted_response(y_metric)

    def objective(parameters: np.ndarray) -> float:
        transformed, _, _ = rotated_metric_transform(x_metric, parameters)
        squared_distances = pairwise_squared_distances(transformed, transformed)
        if objective_kind == "decision":
            loo_orders = orders_from_squared_distances(
                squared_distances,
                y_sorted,
                y_order,
                ALPHA,
                bandwidth=1.0,
                exclude_self=True,
            )
            loss = newsvendor_cost(loo_orders, y_metric).mean()
        elif objective_kind == "prediction":
            loo_prediction = kernel_average_from_squared_distances(
                squared_distances,
                y_metric,
                bandwidth=1.0,
                exclude_self=True,
            )
            loss = np.mean((loo_prediction - y_metric) ** 2)
        else:
            raise ValueError(f"Unknown objective kind: {objective_kind}")

        centered = parameters[1:] - parameters[1:].mean()
        penalty = 0.1 * parameters[0] ** 2 + np.mean(centered**2)
        return float(loss + reg * penalty)

    starts = []
    if initial_parameters is not None:
        starts.append(np.asarray(initial_parameters, dtype=float))
    starts.extend(
        [
            np.array([0.0, 0.0, 0.0]),
            np.array([pi / 4, 0.0, -2.0]),
            np.array([-pi / 4, 0.0, -2.0]),
            np.array([pi / 2, 0.0, -2.0]),
        ]
    )
    best_result = None
    for start in starts:
        result = minimize(
            objective,
            start,
            method="Powell",
            options={"maxfev": maxfev, "xtol": 1e-3, "ftol": 1e-3},
        )
        if best_result is None or result.fun < best_result.fun:
            best_result = result

    return best_result.x, float(best_result.fun)


def learn_rotated_metric_allocation(
    x_train: np.ndarray,
    rewards_train: np.ndarray,
    objective_kind: str,
    initial_parameters: np.ndarray | None = None,
    fit_size: int = 120,
    reg: float = 0.02,
    maxfev: int = 160,
) -> tuple[np.ndarray, float]:
    fit_size = min(len(x_train), fit_size)
    x_metric = x_train[:fit_size]
    rewards_metric = rewards_train[:fit_size]

    def objective(parameters: np.ndarray) -> float:
        transformed, _, _ = rotated_metric_transform(x_metric, parameters)
        squared_distances = pairwise_squared_distances(transformed, transformed)
        if objective_kind == "decision":
            losses = np.empty(fit_size)
            for index in range(fit_size):
                weights = np.exp(-squared_distances[index] / 2.0)
                weights[index] = 0.0
                action = choose_weighted_action(weights, rewards_metric)
                losses[index] = -rewards_metric[index, action]
            loss = losses.mean()
        elif objective_kind == "prediction":
            loo_prediction = kernel_average_from_squared_distances(
                squared_distances,
                rewards_metric,
                bandwidth=1.0,
                exclude_self=True,
            )
            loss = np.mean((loo_prediction - rewards_metric) ** 2)
        else:
            raise ValueError(f"Unknown objective kind: {objective_kind}")

        centered = parameters[1:] - parameters[1:].mean()
        penalty = 0.1 * parameters[0] ** 2 + np.mean(centered**2)
        return float(loss + reg * penalty)

    starts = []
    if initial_parameters is not None:
        starts.append(np.asarray(initial_parameters, dtype=float))
    starts.extend(
        [
            np.array([0.0, 0.0, 0.0]),
            np.array([pi / 4, 0.0, -2.0]),
            np.array([-pi / 4, 0.0, -2.0]),
            np.array([pi / 2, 0.0, -2.0]),
        ]
    )
    best_result = None
    for start in starts:
        result = minimize(
            objective,
            start,
            method="Powell",
            options={"maxfev": maxfev, "xtol": 1e-3, "ftol": 1e-3},
        )
        if best_result is None or result.fun < best_result.fun:
            best_result = result

    return best_result.x, float(best_result.fun)


def learn_spectral_mixture_newsvendor(
    x_train: np.ndarray,
    y_train: np.ndarray,
    objective_kind: str,
    initial_parameters: np.ndarray | None = None,
    fit_size: int = 120,
    reg: float = 0.01,
    maxfev: int = 220,
) -> tuple[np.ndarray, float]:
    fit_size = min(len(x_train), fit_size)
    x_metric = x_train[:fit_size]
    y_metric = y_train[:fit_size]
    y_sorted, y_order = sorted_response(y_metric)
    parameter_dim = SPECTRAL_NUM_COMPONENTS * (1 + 2 * x_train.shape[1])

    def objective(parameters: np.ndarray) -> float:
        weights = spectral_mixture_weight_matrix(
            x_metric,
            x_metric,
            parameters,
            bandwidth=1.0,
            exclude_self=True,
        )
        if objective_kind == "decision":
            loo_orders = orders_from_weights(weights, y_sorted, y_order, ALPHA)
            loss = newsvendor_cost(loo_orders, y_metric).mean()
        elif objective_kind == "prediction":
            loo_prediction = kernel_average_from_weights(weights, y_metric, exclude_self=True)
            loss = np.mean((loo_prediction - y_metric) ** 2)
        else:
            raise ValueError(f"Unknown objective kind: {objective_kind}")

        mixture_weights, variances, frequencies = unpack_spectral_mixture_parameters(
            parameters,
            x_train.shape[1],
        )
        penalty = np.mean((np.log(np.clip(mixture_weights, 1e-12, None)) - np.log(mixture_weights).mean()) ** 2)
        penalty += np.mean(np.log(variances) ** 2)
        penalty += 0.1 * np.mean(frequencies**2)
        return float(loss + reg * penalty)

    starts = []
    if initial_parameters is not None:
        starts.append(np.asarray(initial_parameters, dtype=float))
    zero = np.zeros(parameter_dim)
    freq_start = np.zeros(parameter_dim)
    freq_start[-x_train.shape[1] :] = 0.5
    starts.extend([zero, freq_start])

    best_result = None
    for start in starts:
        result = minimize(
            objective,
            start,
            method="Powell",
            options={"maxfev": maxfev, "xtol": 1e-3, "ftol": 1e-3},
        )
        if best_result is None or result.fun < best_result.fun:
            best_result = result

    return np.asarray(best_result.x, dtype=float), float(best_result.fun)


def learn_spectral_mixture_allocation(
    x_train: np.ndarray,
    rewards_train: np.ndarray,
    objective_kind: str,
    initial_parameters: np.ndarray | None = None,
    fit_size: int = 120,
    reg: float = 0.005,
    maxfev: int = 220,
) -> tuple[np.ndarray, float]:
    fit_size = min(len(x_train), fit_size)
    x_metric = x_train[:fit_size]
    rewards_metric = rewards_train[:fit_size]
    parameter_dim = SPECTRAL_NUM_COMPONENTS * (1 + 2 * x_train.shape[1])

    def objective(parameters: np.ndarray) -> float:
        weights = spectral_mixture_weight_matrix(
            x_metric,
            x_metric,
            parameters,
            bandwidth=1.0,
            exclude_self=True,
        )
        if objective_kind == "decision":
            losses = np.empty(fit_size)
            for index in range(fit_size):
                action = choose_weighted_action(weights[index], rewards_metric)
                losses[index] = -rewards_metric[index, action]
            loss = losses.mean()
        elif objective_kind == "prediction":
            loo_prediction = kernel_average_from_weights(weights, rewards_metric, exclude_self=True)
            loss = np.mean((loo_prediction - rewards_metric) ** 2)
        else:
            raise ValueError(f"Unknown objective kind: {objective_kind}")

        mixture_weights, variances, frequencies = unpack_spectral_mixture_parameters(
            parameters,
            x_train.shape[1],
        )
        penalty = np.mean((np.log(np.clip(mixture_weights, 1e-12, None)) - np.log(mixture_weights).mean()) ** 2)
        penalty += np.mean(np.log(variances) ** 2)
        penalty += 0.1 * np.mean(frequencies**2)
        return float(loss + reg * penalty)

    starts = []
    if initial_parameters is not None:
        starts.append(np.asarray(initial_parameters, dtype=float))
    zero = np.zeros(parameter_dim)
    freq_start = np.zeros(parameter_dim)
    freq_start[-x_train.shape[1] :] = 0.5
    starts.extend([zero, freq_start])

    best_result = None
    for start in starts:
        result = minimize(
            objective,
            start,
            method="Powell",
            options={"maxfev": maxfev, "xtol": 1e-3, "ftol": 1e-3},
        )
        if best_result is None or result.fun < best_result.fun:
            best_result = result

    return np.asarray(best_result.x, dtype=float), float(best_result.fun)


def learn_neural_embedding_newsvendor(
    x_train: np.ndarray,
    y_train: np.ndarray,
    objective_kind: str,
    initial_parameters: np.ndarray | None = None,
    fit_size: int = 120,
    reg: float = 0.01,
    maxfev: int = 240,
) -> tuple[np.ndarray, float]:
    fit_size = min(len(x_train), fit_size)
    x_metric = x_train[:fit_size]
    y_metric = y_train[:fit_size]
    y_sorted, y_order = sorted_response(y_metric)
    parameter_dim = neural_embedding_parameter_count(x_train.shape[1], hidden_dim=NEURAL_HIDDEN_DIM)

    def objective(parameters: np.ndarray) -> float:
        transformed = neural_embedding_transform(x_metric, parameters, hidden_dim=NEURAL_HIDDEN_DIM)
        squared_distances = pairwise_squared_distances(transformed, transformed)
        if objective_kind == "decision":
            loo_orders = orders_from_squared_distances(
                squared_distances,
                y_sorted,
                y_order,
                ALPHA,
                bandwidth=1.0,
                exclude_self=True,
            )
            loss = newsvendor_cost(loo_orders, y_metric).mean()
        elif objective_kind == "prediction":
            loo_prediction = kernel_average_from_squared_distances(
                squared_distances,
                y_metric,
                bandwidth=1.0,
                exclude_self=True,
            )
            loss = np.mean((loo_prediction - y_metric) ** 2)
        else:
            raise ValueError(f"Unknown objective kind: {objective_kind}")
        return float(loss + reg * np.mean(parameters**2))

    starts = []
    if initial_parameters is not None:
        starts.append(np.asarray(initial_parameters, dtype=float))
    starts.append(neural_embedding_initial_parameters(x_train.shape[1], hidden_dim=NEURAL_HIDDEN_DIM))
    starts.append(np.full(parameter_dim, 0.05))

    best_result = None
    for start in starts:
        result = minimize(
            objective,
            start,
            method="Powell",
            options={"maxfev": maxfev, "xtol": 1e-3, "ftol": 1e-3},
        )
        if best_result is None or result.fun < best_result.fun:
            best_result = result

    return np.asarray(best_result.x, dtype=float), float(best_result.fun)


def learn_neural_embedding_allocation(
    x_train: np.ndarray,
    rewards_train: np.ndarray,
    objective_kind: str,
    initial_parameters: np.ndarray | None = None,
    fit_size: int = 120,
    reg: float = 0.005,
    maxfev: int = 240,
) -> tuple[np.ndarray, float]:
    fit_size = min(len(x_train), fit_size)
    x_metric = x_train[:fit_size]
    rewards_metric = rewards_train[:fit_size]
    parameter_dim = neural_embedding_parameter_count(x_train.shape[1], hidden_dim=NEURAL_HIDDEN_DIM)

    def objective(parameters: np.ndarray) -> float:
        transformed = neural_embedding_transform(x_metric, parameters, hidden_dim=NEURAL_HIDDEN_DIM)
        squared_distances = pairwise_squared_distances(transformed, transformed)
        if objective_kind == "decision":
            losses = np.empty(fit_size)
            for index in range(fit_size):
                weights = np.exp(-squared_distances[index] / 2.0)
                weights[index] = 0.0
                action = choose_weighted_action(weights, rewards_metric)
                losses[index] = -rewards_metric[index, action]
            loss = losses.mean()
        elif objective_kind == "prediction":
            loo_prediction = kernel_average_from_squared_distances(
                squared_distances,
                rewards_metric,
                bandwidth=1.0,
                exclude_self=True,
            )
            loss = np.mean((loo_prediction - rewards_metric) ** 2)
        else:
            raise ValueError(f"Unknown objective kind: {objective_kind}")
        return float(loss + reg * np.mean(parameters**2))

    starts = []
    if initial_parameters is not None:
        starts.append(np.asarray(initial_parameters, dtype=float))
    starts.append(neural_embedding_initial_parameters(x_train.shape[1], hidden_dim=NEURAL_HIDDEN_DIM))
    starts.append(np.full(parameter_dim, 0.05))

    best_result = None
    for start in starts:
        result = minimize(
            objective,
            start,
            method="Powell",
            options={"maxfev": maxfev, "xtol": 1e-3, "ftol": 1e-3},
        )
        if best_result is None or result.fun < best_result.fun:
            best_result = result

    return np.asarray(best_result.x, dtype=float), float(best_result.fun)


def apply_transform(
    x_train_std: np.ndarray,
    x_val_std: np.ndarray,
    x_test_std: np.ndarray,
    transform_kind: str,
    transform_payload: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    if transform_kind == "identity":
        return x_train_std, x_val_std, x_test_std, {}

    if transform_kind == "screened_diagonal":
        if transform_payload is None:
            raise ValueError("Missing scales for screened_diagonal transform")
        scales = transform_payload
        transformed_train = transform_screened_diagonal(x_train_std, scales)
        transformed_val = transform_screened_diagonal(x_val_std, scales)
        transformed_test = transform_screened_diagonal(x_test_std, scales)
        return transformed_train, transformed_val, transformed_test, {}

    if transform_kind == "rotated_2d":
        if transform_payload is None:
            raise ValueError("Missing parameters for rotated_2d transform")
        parameters = transform_payload
        transformed_train, _, scales = rotated_metric_transform(x_train_std, parameters)
        transformed_val, _, _ = rotated_metric_transform(x_val_std, parameters)
        transformed_test, _, _ = rotated_metric_transform(x_test_std, parameters)
        metadata = {
            "angle": float(parameters[0]),
            "scale_ratio": float(scales[0] / max(scales[1], 1e-12)),
            "primary_scale": float(scales[0]),
            "secondary_scale": float(scales[1]),
        }
        return transformed_train, transformed_val, transformed_test, metadata

    if transform_kind == "linear_projection":
        if transform_payload is None:
            raise ValueError("Missing projection matrix for linear_projection transform")
        projection = np.atleast_2d(transform_payload)
        transformed_train = x_train_std @ projection.T
        transformed_val = x_val_std @ projection.T
        transformed_test = x_test_std @ projection.T
        metadata = {"projected_dim": float(projection.shape[0])}
        return transformed_train, transformed_val, transformed_test, metadata

    if transform_kind == "neural_embedding":
        if transform_payload is None:
            raise ValueError("Missing parameters for neural_embedding transform")
        parameters = np.asarray(transform_payload, dtype=float)
        transformed_train = neural_embedding_transform(x_train_std, parameters, hidden_dim=NEURAL_HIDDEN_DIM)
        transformed_val = neural_embedding_transform(x_val_std, parameters, hidden_dim=NEURAL_HIDDEN_DIM)
        transformed_test = neural_embedding_transform(x_test_std, parameters, hidden_dim=NEURAL_HIDDEN_DIM)
        metadata = {
            "neural_parameter_norm": float(np.linalg.norm(parameters)),
        }
        return transformed_train, transformed_val, transformed_test, metadata

    raise ValueError(f"Unknown transform kind: {transform_kind}")


def direct_kernel_weight_matrix(
    x_query: np.ndarray,
    x_train: np.ndarray,
    kernel_kind: str,
    kernel_payload: np.ndarray | None,
    bandwidth: float,
    exclude_self: bool = False,
) -> np.ndarray:
    if kernel_kind == "spectral_mixture":
        if kernel_payload is None:
            raise ValueError("Missing parameters for spectral_mixture kernel")
        return spectral_mixture_weight_matrix(
            x_query,
            x_train,
            np.asarray(kernel_payload, dtype=float),
            bandwidth=bandwidth,
            exclude_self=exclude_self,
        )
    raise ValueError(f"Unknown direct kernel kind: {kernel_kind}")


def tune_bandwidth_newsvendor_direct_kernel(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    kernel_kind: str,
    kernel_payload: np.ndarray | None,
) -> float:
    y_sorted, y_order = sorted_response(y_train)
    candidates = bandwidth_candidates(x_train)
    best_bandwidth = float(candidates[0])
    best_cost = float("inf")

    for bandwidth in candidates:
        weights = direct_kernel_weight_matrix(
            x_val,
            x_train,
            kernel_kind,
            kernel_payload,
            float(bandwidth),
        )
        candidate_orders = orders_from_weights(weights, y_sorted, y_order, ALPHA)
        avg_cost = newsvendor_cost(candidate_orders, y_val).mean()
        if avg_cost < best_cost:
            best_cost = float(avg_cost)
            best_bandwidth = float(bandwidth)
    return best_bandwidth


def tune_bandwidth_scalar_prediction_direct_kernel(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    mean_val: np.ndarray,
    kernel_kind: str,
    kernel_payload: np.ndarray | None,
) -> float:
    candidates = bandwidth_candidates(x_train)
    best_bandwidth = float(candidates[0])
    best_mse = float("inf")

    for bandwidth in candidates:
        weights = direct_kernel_weight_matrix(
            x_val,
            x_train,
            kernel_kind,
            kernel_payload,
            float(bandwidth),
        )
        prediction = kernel_average_from_weights(weights, y_train)
        mse = float(np.mean((prediction - mean_val) ** 2))
        if mse < best_mse:
            best_mse = mse
            best_bandwidth = float(bandwidth)
    return best_bandwidth


def tune_bandwidth_allocation_direct_kernel(
    x_train: np.ndarray,
    rewards_train: np.ndarray,
    x_val: np.ndarray,
    rewards_val: np.ndarray,
    kernel_kind: str,
    kernel_payload: np.ndarray | None,
) -> float:
    candidates = bandwidth_candidates(x_train)
    best_bandwidth = float(candidates[0])
    best_cost = float("inf")

    for bandwidth in candidates:
        weights = direct_kernel_weight_matrix(
            x_val,
            x_train,
            kernel_kind,
            kernel_payload,
            float(bandwidth),
        )
        costs = []
        for row_weights, reward in zip(weights, rewards_val):
            action = choose_weighted_action(row_weights, rewards_train)
            costs.append(-reward[action])
        avg_cost = float(np.mean(costs))
        if avg_cost < best_cost:
            best_cost = avg_cost
            best_bandwidth = float(bandwidth)
    return best_bandwidth


def tune_bandwidth_vector_prediction_direct_kernel(
    x_train: np.ndarray,
    rewards_train: np.ndarray,
    x_val: np.ndarray,
    mean_rewards_val: np.ndarray,
    kernel_kind: str,
    kernel_payload: np.ndarray | None,
) -> float:
    candidates = bandwidth_candidates(x_train)
    best_bandwidth = float(candidates[0])
    best_mse = float("inf")

    for bandwidth in candidates:
        weights = direct_kernel_weight_matrix(
            x_val,
            x_train,
            kernel_kind,
            kernel_payload,
            float(bandwidth),
        )
        prediction = kernel_average_from_weights(weights, rewards_train)
        mse = float(np.mean((prediction - mean_rewards_val) ** 2))
        if mse < best_mse:
            best_mse = mse
            best_bandwidth = float(bandwidth)
    return best_bandwidth


def evaluate_newsvendor_transform(
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
) -> dict[str, float]:
    x_train_t, x_val_t, x_test_t, metadata = apply_transform(
        x_train_std,
        x_val_std,
        x_test_std,
        transform_kind,
        transform_payload,
    )

    y_sorted, y_order = sorted_response(y_train)
    decision_bandwidth = tune_bandwidth_newsvendor(x_train_t, y_train, x_val_t, y_val)
    orders = orders_from_squared_distances(
        pairwise_squared_distances(x_test_t, x_train_t),
        y_sorted,
        y_order,
        ALPHA,
        decision_bandwidth,
    )
    cost = float(newsvendor_cost(orders, y_test).mean())

    prediction_bandwidth = tune_bandwidth_scalar_prediction(
        x_train_t,
        y_train,
        x_val_t,
        mean_val,
    )
    mean_prediction = kernel_average_from_squared_distances(
        pairwise_squared_distances(x_test_t, x_train_t),
        y_train,
        prediction_bandwidth,
    )
    pred_mse = float(np.mean((mean_prediction - mean_test) ** 2))
    oracle_cost = float(newsvendor_cost(oracle_orders, y_test).mean())

    result = {
        "cost": cost,
        "regret": cost - oracle_cost,
        "decision_bandwidth": float(decision_bandwidth),
        "prediction_bandwidth": float(prediction_bandwidth),
        "pred_mse": pred_mse,
    }
    result.update(metadata)
    return result


def evaluate_allocation_transform(
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
) -> tuple[dict[str, float], np.ndarray]:
    x_train_t, x_val_t, x_test_t, metadata = apply_transform(
        x_train_std,
        x_val_std,
        x_test_std,
        transform_kind,
        transform_payload,
    )

    cost, decision_bandwidth, decisions = evaluate_allocation_kernel(
        x_train_t,
        rewards_train,
        x_val_t,
        rewards_val,
        x_test_t,
        rewards_test,
    )
    prediction_bandwidth = tune_bandwidth_vector_prediction(
        x_train_t,
        rewards_train,
        x_val_t,
        mean_rewards_val,
    )
    predicted_rewards = kernel_average_from_squared_distances(
        pairwise_squared_distances(x_test_t, x_train_t),
        rewards_train,
        prediction_bandwidth,
    )
    pred_mse = float(np.mean((predicted_rewards - mean_rewards_test) ** 2))
    oracle_cost = float(np.mean(-np.max(mean_rewards_test, axis=1)))

    result = {
        "cost": float(cost),
        "regret": float(cost - oracle_cost),
        "decision_bandwidth": float(decision_bandwidth),
        "prediction_bandwidth": float(prediction_bandwidth),
        "pred_mse": pred_mse,
    }
    result.update(metadata)
    return result, decisions


def evaluate_newsvendor_direct_kernel(
    x_train_std: np.ndarray,
    y_train: np.ndarray,
    x_val_std: np.ndarray,
    y_val: np.ndarray,
    mean_val: np.ndarray,
    x_test_std: np.ndarray,
    y_test: np.ndarray,
    mean_test: np.ndarray,
    oracle_orders: np.ndarray,
    kernel_kind: str,
    kernel_payload: np.ndarray | None,
) -> dict[str, float]:
    y_sorted, y_order = sorted_response(y_train)
    decision_bandwidth = tune_bandwidth_newsvendor_direct_kernel(
        x_train_std,
        y_train,
        x_val_std,
        y_val,
        kernel_kind,
        kernel_payload,
    )
    test_weights = direct_kernel_weight_matrix(
        x_test_std,
        x_train_std,
        kernel_kind,
        kernel_payload,
        decision_bandwidth,
    )
    orders = orders_from_weights(test_weights, y_sorted, y_order, ALPHA)
    cost = float(newsvendor_cost(orders, y_test).mean())

    prediction_bandwidth = tune_bandwidth_scalar_prediction_direct_kernel(
        x_train_std,
        y_train,
        x_val_std,
        mean_val,
        kernel_kind,
        kernel_payload,
    )
    pred_weights = direct_kernel_weight_matrix(
        x_test_std,
        x_train_std,
        kernel_kind,
        kernel_payload,
        prediction_bandwidth,
    )
    mean_prediction = kernel_average_from_weights(pred_weights, y_train)
    pred_mse = float(np.mean((mean_prediction - mean_test) ** 2))
    oracle_cost = float(newsvendor_cost(oracle_orders, y_test).mean())

    result = {
        "cost": cost,
        "regret": cost - oracle_cost,
        "decision_bandwidth": float(decision_bandwidth),
        "prediction_bandwidth": float(prediction_bandwidth),
        "pred_mse": pred_mse,
    }
    if kernel_kind == "spectral_mixture" and kernel_payload is not None:
        result.update(spectral_mixture_metadata(np.asarray(kernel_payload, dtype=float), x_train_std.shape[1]))
    return result


def evaluate_allocation_direct_kernel(
    x_train_std: np.ndarray,
    rewards_train: np.ndarray,
    x_val_std: np.ndarray,
    rewards_val: np.ndarray,
    mean_rewards_val: np.ndarray,
    x_test_std: np.ndarray,
    rewards_test: np.ndarray,
    mean_rewards_test: np.ndarray,
    kernel_kind: str,
    kernel_payload: np.ndarray | None,
) -> tuple[dict[str, float], np.ndarray]:
    decision_bandwidth = tune_bandwidth_allocation_direct_kernel(
        x_train_std,
        rewards_train,
        x_val_std,
        rewards_val,
        kernel_kind,
        kernel_payload,
    )
    test_weights = direct_kernel_weight_matrix(
        x_test_std,
        x_train_std,
        kernel_kind,
        kernel_payload,
        decision_bandwidth,
    )
    decisions = np.empty(len(x_test_std), dtype=int)
    costs = np.empty(len(x_test_std))
    for index, (row_weights, reward) in enumerate(zip(test_weights, rewards_test)):
        action = choose_weighted_action(row_weights, rewards_train)
        decisions[index] = action
        costs[index] = -reward[action]

    prediction_bandwidth = tune_bandwidth_vector_prediction_direct_kernel(
        x_train_std,
        rewards_train,
        x_val_std,
        mean_rewards_val,
        kernel_kind,
        kernel_payload,
    )
    pred_weights = direct_kernel_weight_matrix(
        x_test_std,
        x_train_std,
        kernel_kind,
        kernel_payload,
        prediction_bandwidth,
    )
    predicted_rewards = kernel_average_from_weights(pred_weights, rewards_train)
    pred_mse = float(np.mean((predicted_rewards - mean_rewards_test) ** 2))
    oracle_cost = float(np.mean(-np.max(mean_rewards_test, axis=1)))

    result = {
        "cost": float(costs.mean()),
        "regret": float(costs.mean() - oracle_cost),
        "decision_bandwidth": float(decision_bandwidth),
        "prediction_bandwidth": float(prediction_bandwidth),
        "pred_mse": pred_mse,
    }
    if kernel_kind == "spectral_mixture" and kernel_payload is not None:
        result.update(spectral_mixture_metadata(np.asarray(kernel_payload, dtype=float), x_train_std.shape[1]))
    return result, decisions


def evaluate_newsvendor_stochopt_forest(
    x_train_std: np.ndarray,
    y_train: np.ndarray,
    x_test_std: np.ndarray,
    y_test: np.ndarray,
    mean_test: np.ndarray,
    oracle_orders: np.ndarray,
    seed: int,
) -> dict[str, float]:
    forest = build_stochopt_forest(
        x_train_std,
        y_train,
        task_kind="newsvendor",
        n_trees=40,
        subsample_ratio=0.5,
        max_depth=7,
        min_leaf_size=3,
        n_thresholds=20,
        balancedness_tol=0.1,
        honest=False,
        seed=seed,
    )
    weights = stochopt_forest_weight_matrix(forest, x_test_std, len(x_train_std))
    y_sorted, y_order = sorted_response(y_train)
    orders = orders_from_weights(weights, y_sorted, y_order, ALPHA)
    cost = float(newsvendor_cost(orders, y_test).mean())
    mean_prediction = kernel_average_from_weights(weights, y_train)
    pred_mse = float(np.mean((mean_prediction - mean_test) ** 2))
    oracle_cost = float(newsvendor_cost(oracle_orders, y_test).mean())
    avg_leaf_mass = float(np.mean(np.sum(weights > 0.0, axis=1)))
    return {
        "cost": cost,
        "regret": cost - oracle_cost,
        "pred_mse": pred_mse,
        "avg_leaf_support": avg_leaf_mass,
    }


def load_official_sof_newsvendor_module() -> Any:
    global _OFFICIAL_SOF_NV_MODULE
    if _OFFICIAL_SOF_NV_MODULE is not None:
        return _OFFICIAL_SOF_NV_MODULE

    official_root = Path(__file__).resolve().parents[1] / "replication_package_MS-BDA-20-02949" / "newsvendor"
    if not official_root.exists():
        raise FileNotFoundError(f"Official SOF newsvendor code not found at {official_root}")

    if "mkl" not in sys.modules:
        mkl = types.ModuleType("mkl")
        mkl.set_num_threads = lambda *args, **kwargs: None
        sys.modules["mkl"] = mkl
    if "gurobipy" not in sys.modules:
        gurobipy = types.ModuleType("gurobipy")
        gurobipy.GRB = types.SimpleNamespace()
        sys.modules["gurobipy"] = gurobipy
    if str(official_root) not in sys.path:
        sys.path.insert(0, str(official_root))

    spec = importlib.util.spec_from_file_location(
        "_official_sof_nv_tree_utilities",
        official_root / "nv_tree_utilities.py",
    )
    if spec is None or spec.loader is None:
        raise ImportError("Could not load official SOF newsvendor module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    _OFFICIAL_SOF_NV_MODULE = module
    return module


def fit_official_newsvendor_sof(
    x_train_std: np.ndarray,
    y_train: np.ndarray,
    seed: int,
    variant: str = OFFICIAL_SOF_VARIANT,
    num_trees: int = OFFICIAL_SOF_NUM_TREES,
    subsample_ratio: float = 1.0,
    bootstrap: bool = True,
    honesty: bool = False,
    mtry: int | None = None,
    min_leaf_size: int = 10,
    max_depth: int = 100,
    n_proposals: int | None = None,
    balancedness_tol: float = 0.2,
) -> Any:
    official_nv = load_official_sof_newsvendor_module()
    y_train_matrix = y_train[:, None]
    h_list = np.array([OVERAGE_COST], dtype=float)
    b_list = np.array([UNDERAGE_COST], dtype=float)
    capacity = 1e6

    opt_solver = partial(official_nv.solve_multi_nv, h_list=h_list, b_list=b_list, C=capacity, verbose=False)
    hessian_computer = partial(official_nv.compute_hessian, h_list=h_list, b_list=b_list, C=capacity)
    gradient_computer = partial(official_nv.compute_gradient, h_list=h_list, b_list=b_list, C=capacity)
    active_constraint = partial(official_nv.search_active_constraint, C=capacity)
    update_step = partial(official_nv.compute_update_step, constraint=True)

    if variant == "approx_sol":
        crit_computer = partial(official_nv.compute_crit_approx_sol, h_list=h_list, b_list=b_list)
        impurity_computer = partial(official_nv.impurity_approx_sol, h_list=h_list, b_list=b_list, C=capacity)
    elif variant == "approx_risk":
        crit_computer = official_nv.compute_crit_approx_risk
        impurity_computer = partial(official_nv.impurity_approx_risk, h_list=h_list, b_list=b_list, C=capacity)
    else:
        raise ValueError(f"Unknown official SOF variant: {variant}")

    if mtry is None:
        mtry = x_train_std.shape[1]
    if n_proposals is None:
        n_proposals = len(x_train_std)

    model = official_nv.forest(
        opt_solver=opt_solver,
        hessian_computer=hessian_computer,
        gradient_computer=gradient_computer,
        search_active_constraint=active_constraint,
        compute_update_step=update_step,
        crit_computer=crit_computer,
        impurity_computer=impurity_computer,
        subsample_ratio=subsample_ratio,
        bootstrap=bootstrap,
        n_trees=num_trees,
        honesty=honesty,
        mtry=mtry,
        min_leaf_size=min_leaf_size,
        max_depth=max_depth,
        n_proposals=n_proposals,
        balancedness_tol=balancedness_tol,
        verbose=False,
        seed=seed,
    )
    model.fit(y_train_matrix, x_train_std, y_train_matrix, x_train_std)
    return model


def evaluate_official_newsvendor_sof(
    model: Any,
    x_train_std: np.ndarray,
    y_train: np.ndarray,
    x_test_std: np.ndarray,
    y_test: np.ndarray,
    mean_test: np.ndarray,
    oracle_orders: np.ndarray,
) -> dict[str, float | np.ndarray]:
    official_nv = load_official_sof_newsvendor_module()
    y_train_matrix = y_train[:, None]
    h_list = np.array([OVERAGE_COST], dtype=float)
    b_list = np.array([UNDERAGE_COST], dtype=float)
    capacity = 1e6

    weight_rows = np.vstack([model.get_weights(query) for query in x_test_std])
    orders = np.empty(len(x_test_std))
    for index, row_weights in enumerate(weight_rows):
        orders[index] = official_nv.solve_multi_nv(
            y_train_matrix,
            h_list=h_list,
            b_list=b_list,
            C=capacity,
            if_weight=True,
            weights=row_weights,
        )[0][0]
    cost = float(newsvendor_cost(orders, y_test).mean())
    mean_prediction = kernel_average_from_weights(weight_rows, y_train)
    pred_mse = float(np.mean((mean_prediction - mean_test) ** 2))
    oracle_cost = float(newsvendor_cost(oracle_orders, y_test).mean())
    avg_leaf_mass = float(np.mean(np.sum(weight_rows > 0.0, axis=1)))
    split_frequency = normalized_profile(model.compute_feature_split_freq(x_train_std.shape[1]))
    return {
        "cost": cost,
        "regret": cost - oracle_cost,
        "pred_mse": pred_mse,
        "avg_leaf_support": avg_leaf_mass,
        "split_frequency": split_frequency,
    }


def evaluate_official_newsvendor_forests(
    x_train_std: np.ndarray,
    y_train: np.ndarray,
    x_test_std: np.ndarray,
    y_test: np.ndarray,
    mean_test: np.ndarray,
    oracle_orders: np.ndarray,
    seed: int,
) -> dict[str, dict[str, float]]:
    return {
        "sof_apx_sol": evaluate_official_newsvendor_sof(
            fit_official_newsvendor_sof(x_train_std, y_train, seed=seed, variant="approx_sol"),
            x_train_std,
            y_train,
            x_test_std,
            y_test,
            mean_test,
            oracle_orders,
        ),
        "sof_apx_risk": evaluate_official_newsvendor_sof(
            fit_official_newsvendor_sof(x_train_std, y_train, seed=seed, variant="approx_risk"),
            x_train_std,
            y_train,
            x_test_std,
            y_test,
            mean_test,
            oracle_orders,
        ),
    }


def evaluate_allocation_stochopt_forest(
    x_train_std: np.ndarray,
    rewards_train: np.ndarray,
    x_test_std: np.ndarray,
    rewards_test: np.ndarray,
    mean_rewards_test: np.ndarray,
    seed: int,
) -> tuple[dict[str, float], np.ndarray]:
    forest = build_stochopt_forest(
        x_train_std,
        rewards_train,
        task_kind="allocation",
        n_trees=40,
        subsample_ratio=0.5,
        max_depth=7,
        min_leaf_size=3,
        n_thresholds=20,
        balancedness_tol=0.1,
        honest=False,
        seed=seed,
    )
    weights = stochopt_forest_weight_matrix(forest, x_test_std, len(x_train_std))
    decisions = np.empty(len(x_test_std), dtype=int)
    costs = np.empty(len(x_test_std))
    for index, (row_weights, reward) in enumerate(zip(weights, rewards_test)):
        action = choose_weighted_action(row_weights, rewards_train)
        decisions[index] = action
        costs[index] = -reward[action]
    predicted_rewards = kernel_average_from_weights(weights, rewards_train)
    pred_mse = float(np.mean((predicted_rewards - mean_rewards_test) ** 2))
    oracle_cost = float(np.mean(-np.max(mean_rewards_test, axis=1)))
    avg_leaf_mass = float(np.mean(np.sum(weights > 0.0, axis=1)))
    return (
        {
            "cost": float(costs.mean()),
            "regret": float(costs.mean() - oracle_cost),
            "pred_mse": pred_mse,
            "avg_leaf_support": avg_leaf_mass,
        },
        decisions,
    )


def generate_benchmark_a(
    rng: np.random.Generator,
    n: int,
    d: int,
    k: int,
    sigma: float = 4.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = rng.normal(size=(n, d))
    beta = np.zeros(d)
    magnitudes = np.linspace(6.0, 1.5, k)
    signs = np.where(np.arange(k) % 2 == 0, 1.0, -1.0)
    beta[:k] = signs * magnitudes
    mean_demand = 25.0 + x @ beta
    demand = mean_demand + sigma * rng.normal(size=n)
    return x, demand, mean_demand


def generate_benchmark_b(
    rng: np.random.Generator,
    n: int,
    angle: float = pi / 4,
    sigma_base: float = 2.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x = rng.normal(size=(n, 2))
    u = cos(angle) * x[:, 0] + sin(angle) * x[:, 1]
    v = -sin(angle) * x[:, 0] + cos(angle) * x[:, 1]
    mean_demand = 25.0 + 8.0 * np.tanh(2.0 * u)
    sigma = sigma_base * (0.8 + 1.2 * (v > 0.0) + 0.2 * np.abs(u))
    demand = mean_demand + sigma * rng.normal(size=n)
    oracle_order = mean_demand + sigma * ORACLE_SHIFT_UNIT
    return x, demand, mean_demand, oracle_order


def generate_benchmark_c(
    rng: np.random.Generator,
    n: int,
    angle: float = pi / 4,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x = rng.normal(size=(n, 2))
    u = cos(angle) * x[:, 0] + sin(angle) * x[:, 1]
    mean_reward_0 = np.where(u > 0.0, 12.0, 5.0) + 0.4 * u
    mean_reward_1 = np.where(u > 0.0, 5.0, 12.0) - 0.4 * u
    rewards = np.column_stack(
        [
            mean_reward_0 + rng.normal(scale=2.5, size=n),
            mean_reward_1 + rng.normal(scale=2.5, size=n),
        ]
    )
    mean_rewards = np.column_stack([mean_reward_0, mean_reward_1])
    return x, rewards, mean_rewards, u


def benchmark_a_seed(d: int, k: int, n_train: int, replication: int) -> int:
    return 1000 + 100 * d + 10 * k + replication + n_train


def benchmark_a_base_required_columns(d: int) -> list[str]:
    return [
        "fixed_regret",
        "pca_regret",
        "lasso_regret",
        "prediction_regret",
        "learned_regret",
        "lasso_selected_count",
        "prediction_signal_mass_share",
        "learned_signal_mass_share",
        f"lasso_selected_{d - 1}",
        f"prediction_scale_{d - 1}",
        f"learned_scale_{d - 1}",
    ]


def benchmark_a_sof_required_columns(d: int) -> list[str]:
    return [
        "sof_regret",
        "lasso_sof_regret",
        "sof_signal_split_share",
        "lasso_sof_signal_split_share",
        f"sof_split_{d - 1}",
        f"lasso_sof_split_{d - 1}",
    ]


def restore_lasso_selected_from_row(
    row: dict[str, Any] | None,
    d: int,
) -> np.ndarray:
    if row is None:
        return np.array([], dtype=int)
    columns = [f"lasso_selected_{j}" for j in range(d)]
    if not row_has_columns(row, columns):
        return np.array([], dtype=int)
    mask = np.array([float(row[column]) > 0.5 for column in columns], dtype=bool)
    return np.flatnonzero(mask)


def run_benchmark_a_replication(
    d: int,
    k: int,
    n_train: int,
    replication: int,
    existing_row: dict[str, Any] | None = None,
) -> dict[str, float | int | str]:
    seed = benchmark_a_seed(d, k, n_train, replication)
    rng = np.random.default_rng(seed)
    n_val = max(100, n_train // 2)
    x_train, y_train, _ = generate_benchmark_a(rng, n_train, d=d, k=k)
    x_val, y_val, mean_val = generate_benchmark_a(rng, n_val, d=d, k=k)
    x_test, y_test, mean_test = generate_benchmark_a(rng, 2000, d=d, k=k)

    x_train_std, x_val_std = standardize(x_train, x_val)
    _, x_test_std = standardize(x_train, x_test)
    oracle_orders = mean_test + 4.0 * ORACLE_SHIFT_UNIT

    base_ready = row_has_columns(existing_row, benchmark_a_base_required_columns(d))
    if base_ready:
        row: dict[str, float | int | str] = dict(existing_row)
        lasso_selected = restore_lasso_selected_from_row(existing_row, d)
        if len(lasso_selected) == 0:
            lasso_selected, _, _ = select_lasso_features(x_train_std, y_train, x_val_std, y_val)
    else:
        pca_projection = fit_pca_projection(x_train_std, n_components=k)
        lasso_selected, lasso_alpha, lasso_selection_mse = select_lasso_features(
            x_train_std,
            y_train,
            x_val_std,
            y_val,
        )
        lasso_projection = projection_from_selected_features(d, lasso_selected)
        selected = select_linear_features(x_train_std, y_train, top_k=min(10, d))
        _, prediction_scales, prediction_objective = learn_screened_diagonal_metric(
            x_train_std,
            y_train,
            objective_kind="prediction",
            selected_features=selected,
            fit_size=min(80, n_train),
            top_k=min(10, d),
            reg=0.02,
            maxfev=90,
        )
        _, decision_scales, decision_objective = learn_screened_diagonal_metric(
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

        global_order = float(np.quantile(y_train, ALPHA))
        global_cost = float(newsvendor_cost(np.full(len(x_test), global_order), y_test).mean())
        global_pred_mse = float(np.mean((np.full(len(x_test), y_train.mean()) - mean_test) ** 2))

        point_design = np.column_stack([np.ones(len(x_train)), x_train])
        point_beta, *_ = np.linalg.lstsq(point_design, y_train, rcond=None)
        point_prediction = np.column_stack([np.ones(len(x_test)), x_test]) @ point_beta
        point_cost = float(newsvendor_cost(point_prediction, y_test).mean())
        point_pred_mse = float(np.mean((point_prediction - mean_test) ** 2))

        fixed_metrics = evaluate_newsvendor_transform(
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
        pca_metrics = evaluate_newsvendor_transform(
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
            transform_payload=pca_projection,
        )
        lasso_metrics = evaluate_newsvendor_transform(
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
        )
        prediction_metrics = evaluate_newsvendor_transform(
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
        )
        learned_metrics = evaluate_newsvendor_transform(
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
            transform_payload=decision_scales,
        )
        oracle_cost = float(newsvendor_cost(oracle_orders, y_test).mean())

        row = {
            "benchmark": "A",
            "d": d,
            "k": k,
            "n_train": n_train,
            "replication": replication,
            "prediction_metric_objective": prediction_objective,
            "decision_metric_objective": decision_objective,
            "global_cost": global_cost,
            "global_regret": global_cost - oracle_cost,
            "global_pred_mse": global_pred_mse,
            "point_cost": point_cost,
            "point_regret": point_cost - oracle_cost,
            "point_pred_mse": point_pred_mse,
            "fixed_cost": fixed_metrics["cost"],
            "fixed_regret": fixed_metrics["regret"],
            "fixed_pred_mse": fixed_metrics["pred_mse"],
            "fixed_bandwidth": fixed_metrics["decision_bandwidth"],
            "pca_cost": pca_metrics["cost"],
            "pca_regret": pca_metrics["regret"],
            "pca_pred_mse": pca_metrics["pred_mse"],
            "pca_bandwidth": pca_metrics["decision_bandwidth"],
            "pca_projected_dim": pca_metrics["projected_dim"],
            "lasso_cost": lasso_metrics["cost"],
            "lasso_regret": lasso_metrics["regret"],
            "lasso_pred_mse": lasso_metrics["pred_mse"],
            "lasso_bandwidth": lasso_metrics["decision_bandwidth"],
            "lasso_projected_dim": lasso_metrics["projected_dim"],
            "lasso_alpha": lasso_alpha,
            "lasso_selection_mse": lasso_selection_mse,
            "prediction_cost": prediction_metrics["cost"],
            "prediction_regret": prediction_metrics["regret"],
            "prediction_pred_mse": prediction_metrics["pred_mse"],
            "prediction_bandwidth": prediction_metrics["decision_bandwidth"],
            "learned_cost": learned_metrics["cost"],
            "learned_regret": learned_metrics["regret"],
            "learned_pred_mse": learned_metrics["pred_mse"],
            "learned_bandwidth": learned_metrics["decision_bandwidth"],
            "oracle_cost": oracle_cost,
            "lasso_selected_count": int(len(lasso_selected)),
            "lasso_signal_share": float(np.mean(lasso_selected < k)),
            "lasso_signal_recall": float(np.mean(np.isin(np.arange(k), lasso_selected))),
            "prediction_signal_mass_share": float(prediction_scales[:k].sum() / prediction_scales.sum()),
            "learned_signal_mass_share": float(decision_scales[:k].sum() / decision_scales.sum()),
        }
        row["prediction_signal_to_nuisance_scale"] = float(
            prediction_scales[:k].mean() / max(prediction_scales[k:].mean(), 1e-12)
        ) if k < d else float(prediction_scales[:k].mean())
        row["learned_signal_to_nuisance_scale"] = float(
            decision_scales[:k].mean() / max(decision_scales[k:].mean(), 1e-12)
        ) if k < d else float(decision_scales[:k].mean())

        for feature_index in range(d):
            row[f"lasso_selected_{feature_index}"] = float(feature_index in lasso_selected)
            row[f"prediction_scale_{feature_index}"] = float(prediction_scales[feature_index])
            row[f"learned_scale_{feature_index}"] = float(decision_scales[feature_index])

    sof_model = fit_official_newsvendor_sof(
        x_train_std,
        y_train,
        seed=50_000 + seed,
        variant=OFFICIAL_SOF_VARIANT,
    )
    sof_metrics = evaluate_official_newsvendor_sof(
        sof_model,
        x_train_std,
        y_train,
        x_test_std,
        y_test,
        mean_test,
        oracle_orders,
    )

    lasso_sof_model = fit_official_newsvendor_sof(
        x_train_std[:, lasso_selected],
        y_train,
        seed=70_000 + seed,
        variant=OFFICIAL_SOF_VARIANT,
    )
    lasso_sof_metrics = evaluate_official_newsvendor_sof(
        lasso_sof_model,
        x_train_std[:, lasso_selected],
        y_train,
        x_test_std[:, lasso_selected],
        y_test,
        mean_test,
        oracle_orders,
    )

    sof_profile = np.asarray(sof_metrics["split_frequency"], dtype=float)
    lasso_sof_profile = lift_profile_to_dimension(
        lasso_selected,
        np.asarray(lasso_sof_metrics["split_frequency"], dtype=float),
        d,
    )
    row.update(
        {
            "sof_cost": float(sof_metrics["cost"]),
            "sof_regret": float(sof_metrics["regret"]),
            "sof_pred_mse": float(sof_metrics["pred_mse"]),
            "sof_avg_leaf_support": float(sof_metrics["avg_leaf_support"]),
            "lasso_sof_cost": float(lasso_sof_metrics["cost"]),
            "lasso_sof_regret": float(lasso_sof_metrics["regret"]),
            "lasso_sof_pred_mse": float(lasso_sof_metrics["pred_mse"]),
            "lasso_sof_avg_leaf_support": float(lasso_sof_metrics["avg_leaf_support"]),
            "sof_signal_split_share": float(sof_profile[:k].sum()),
            "lasso_sof_signal_split_share": float(lasso_sof_profile[:k].sum()),
        }
    )
    for feature_index in range(d):
        row[f"sof_split_{feature_index}"] = float(sof_profile[feature_index])
        row[f"lasso_sof_split_{feature_index}"] = float(lasso_sof_profile[feature_index])

    return row


def run_benchmark_a(
    existing: pd.DataFrame | None = None,
    cache_path: Path | None = None,
) -> pd.DataFrame:
    existing_lookup = build_existing_row_lookup(existing, ("d", "k", "n_train", "replication"))
    rows_by_key: dict[tuple[int, int, int, int], dict[str, float | int | str]] = {
        key: dict(row) for key, row in existing_lookup.items()
    }
    pending_tasks: list[tuple[int, int, int, int, dict[str, Any] | None]] = []

    for d in [5, 10, 20, 50, 100]:
        for k in [1, 3, 5]:
            if k > d:
                continue
            for n_train in [100, 200, 400, 800]:
                for replication in range(6):
                    key = (d, k, n_train, replication)
                    existing_row = existing_lookup.get(key)
                    if not row_has_columns(existing_row, benchmark_a_base_required_columns(d) + benchmark_a_sof_required_columns(d)):
                        pending_tasks.append((d, k, n_train, replication, existing_row))

    if pending_tasks:
        computed_rows = Parallel(n_jobs=BENCHMARK_PARALLEL_JOBS, verbose=10)(
            delayed(run_benchmark_a_replication)(
                d,
                k,
                n_train,
                replication,
                existing_row,
            )
            for d, k, n_train, replication, existing_row in pending_tasks
        )
        for row in computed_rows:
            key = (int(row["d"]), int(row["k"]), int(row["n_train"]), int(row["replication"]))
            rows_by_key[key] = dict(row)
        if cache_path is not None:
            (
                pd.DataFrame(rows_by_key.values())
                .sort_values(["d", "k", "n_train", "replication"])
                .to_csv(cache_path, index=False)
            )

    return pd.DataFrame(rows_by_key.values()).sort_values(["d", "k", "n_train", "replication"]).reset_index(drop=True)


def benchmark_b_base_required_columns() -> list[str]:
    return [
        "fixed_regret",
        "lasso_regret",
        "prediction_regret",
        "learned_regret",
        "spectral_regret",
        "neural_regret",
        "lasso_selected_count",
        "prediction_angle",
        "learned_scale_ratio",
    ]


def benchmark_b_sof_required_columns() -> list[str]:
    return [
        "sof_regret",
        "sof_avg_leaf_support",
        "lasso_sof_regret",
        "lasso_sof_avg_leaf_support",
    ]


def run_benchmark_b(existing: pd.DataFrame | None = None) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    existing_lookup = build_existing_row_lookup(existing, ("replication",))

    for replication in range(20):
        existing_row = existing_lookup.get((replication,))
        seed = 2000 + replication
        rng = np.random.default_rng(seed)
        x_train, y_train, _, _ = generate_benchmark_b(rng, 250)
        x_val, y_val, mean_val, _ = generate_benchmark_b(rng, 150)
        x_test, y_test, mean_test, oracle_orders = generate_benchmark_b(rng, 2000)

        x_train_std, x_val_std = standardize(x_train, x_val)
        _, x_test_std = standardize(x_train, x_test)

        if row_has_columns(existing_row, benchmark_b_base_required_columns() + benchmark_b_sof_required_columns()):
            rows.append(dict(existing_row))
            continue

        lasso_selected, lasso_alpha, lasso_selection_mse = select_lasso_features(
            x_train_std,
            y_train,
            x_val_std,
            y_val,
        )

        if row_has_columns(existing_row, benchmark_b_base_required_columns()):
            row = dict(existing_row)
        else:
            lasso_projection = projection_from_selected_features(x_train_std.shape[1], lasso_selected)

            prediction_parameters, prediction_objective = learn_rotated_metric_newsvendor(
                x_train_std,
                y_train,
                objective_kind="prediction",
                fit_size=120,
                reg=0.02,
                maxfev=160,
            )
            decision_parameters, decision_objective = learn_rotated_metric_newsvendor(
                x_train_std,
                y_train,
                objective_kind="decision",
                initial_parameters=prediction_parameters,
                fit_size=120,
                reg=0.02,
                maxfev=160,
            )
            spectral_parameters, spectral_objective = learn_spectral_mixture_newsvendor(
                x_train_std,
                y_train,
                objective_kind="decision",
                fit_size=120,
                reg=0.01,
                maxfev=220,
            )
            neural_prediction_parameters, neural_prediction_objective = learn_neural_embedding_newsvendor(
                x_train_std,
                y_train,
                objective_kind="prediction",
                fit_size=120,
                reg=0.01,
                maxfev=220,
            )
            neural_parameters, neural_objective = learn_neural_embedding_newsvendor(
                x_train_std,
                y_train,
                objective_kind="decision",
                initial_parameters=neural_prediction_parameters,
                fit_size=120,
                reg=0.01,
                maxfev=220,
            )

            oracle_cost = float(newsvendor_cost(oracle_orders, y_test).mean())
            global_order = float(np.quantile(y_train, ALPHA))
            global_cost = float(newsvendor_cost(np.full(len(x_test), global_order), y_test).mean())
            global_pred_mse = float(np.mean((np.full(len(x_test), y_train.mean()) - mean_test) ** 2))

            point_design = np.column_stack([np.ones(len(x_train)), x_train])
            point_beta, *_ = np.linalg.lstsq(point_design, y_train, rcond=None)
            point_prediction = np.column_stack([np.ones(len(x_test)), x_test]) @ point_beta
            point_cost = float(newsvendor_cost(point_prediction, y_test).mean())
            point_pred_mse = float(np.mean((point_prediction - mean_test) ** 2))

            fixed_metrics = evaluate_newsvendor_transform(
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
            lasso_metrics = evaluate_newsvendor_transform(
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
            )
            prediction_metrics = evaluate_newsvendor_transform(
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
                transform_payload=prediction_parameters,
            )
            learned_metrics = evaluate_newsvendor_transform(
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
            spectral_metrics = evaluate_newsvendor_direct_kernel(
                x_train_std,
                y_train,
                x_val_std,
                y_val,
                mean_val,
                x_test_std,
                y_test,
                mean_test,
                oracle_orders,
                kernel_kind="spectral_mixture",
                kernel_payload=spectral_parameters,
            )
            neural_metrics = evaluate_newsvendor_transform(
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

            row = {
                "benchmark": "B",
                "replication": replication,
                "prediction_metric_objective": prediction_objective,
                "decision_metric_objective": decision_objective,
                "spectral_metric_objective": spectral_objective,
                "neural_prediction_objective": neural_prediction_objective,
                "neural_metric_objective": neural_objective,
                "global_cost": global_cost,
                "global_regret": global_cost - oracle_cost,
                "global_pred_mse": global_pred_mse,
                "point_cost": point_cost,
                "point_regret": point_cost - oracle_cost,
                "point_pred_mse": point_pred_mse,
                "fixed_cost": fixed_metrics["cost"],
                "fixed_regret": fixed_metrics["regret"],
                "fixed_pred_mse": fixed_metrics["pred_mse"],
                "fixed_bandwidth": fixed_metrics["decision_bandwidth"],
                "lasso_cost": lasso_metrics["cost"],
                "lasso_regret": lasso_metrics["regret"],
                "lasso_pred_mse": lasso_metrics["pred_mse"],
                "lasso_bandwidth": lasso_metrics["decision_bandwidth"],
                "lasso_projected_dim": lasso_metrics["projected_dim"],
                "lasso_selected_count": int(len(lasso_selected)),
                "lasso_alpha": lasso_alpha,
                "lasso_selection_mse": lasso_selection_mse,
                "prediction_cost": prediction_metrics["cost"],
                "prediction_regret": prediction_metrics["regret"],
                "prediction_pred_mse": prediction_metrics["pred_mse"],
                "prediction_bandwidth": prediction_metrics["decision_bandwidth"],
                "learned_cost": learned_metrics["cost"],
                "learned_regret": learned_metrics["regret"],
                "learned_pred_mse": learned_metrics["pred_mse"],
                "learned_bandwidth": learned_metrics["decision_bandwidth"],
                "spectral_cost": spectral_metrics["cost"],
                "spectral_regret": spectral_metrics["regret"],
                "spectral_pred_mse": spectral_metrics["pred_mse"],
                "spectral_bandwidth": spectral_metrics["decision_bandwidth"],
                "spectral_dominant_weight": spectral_metrics["spectral_dominant_weight"],
                "spectral_avg_frequency_norm": spectral_metrics["spectral_avg_frequency_norm"],
                "neural_cost": neural_metrics["cost"],
                "neural_regret": neural_metrics["regret"],
                "neural_pred_mse": neural_metrics["pred_mse"],
                "neural_bandwidth": neural_metrics["decision_bandwidth"],
                "neural_parameter_norm": neural_metrics["neural_parameter_norm"],
                "oracle_cost": oracle_cost,
                "prediction_angle": prediction_metrics["angle"],
                "learned_angle": learned_metrics["angle"],
                "prediction_scale_ratio": prediction_metrics["scale_ratio"],
                "learned_scale_ratio": learned_metrics["scale_ratio"],
            }

        if not row_has_columns(row, ["sof_regret", "sof_cost", "sof_pred_mse", "sof_avg_leaf_support"]):
            sof_model = fit_official_newsvendor_sof(
                x_train_std,
                y_train,
                seed=90_000 + replication,
                variant=OFFICIAL_SOF_VARIANT,
            )
            sof_metrics = evaluate_official_newsvendor_sof(
                sof_model,
                x_train_std,
                y_train,
                x_test_std,
                y_test,
                mean_test,
                oracle_orders,
            )
            row.update(
                {
                    "sof_cost": float(sof_metrics["cost"]),
                    "sof_regret": float(sof_metrics["regret"]),
                    "sof_pred_mse": float(sof_metrics["pred_mse"]),
                    "sof_avg_leaf_support": float(sof_metrics["avg_leaf_support"]),
                }
            )

        if not row_has_columns(row, ["lasso_sof_regret", "lasso_sof_cost", "lasso_sof_pred_mse", "lasso_sof_avg_leaf_support"]):
            lasso_sof_model = fit_official_newsvendor_sof(
                x_train_std[:, lasso_selected],
                y_train,
                seed=95_000 + replication,
                variant=OFFICIAL_SOF_VARIANT,
            )
            lasso_sof_metrics = evaluate_official_newsvendor_sof(
                lasso_sof_model,
                x_train_std[:, lasso_selected],
                y_train,
                x_test_std[:, lasso_selected],
                y_test,
                mean_test,
                oracle_orders,
            )
            row.update(
                {
                    "lasso_sof_cost": float(lasso_sof_metrics["cost"]),
                    "lasso_sof_regret": float(lasso_sof_metrics["regret"]),
                    "lasso_sof_pred_mse": float(lasso_sof_metrics["pred_mse"]),
                    "lasso_sof_avg_leaf_support": float(lasso_sof_metrics["avg_leaf_support"]),
                }
            )
        rows.append(row)

    return pd.DataFrame(rows).sort_values(["replication"]).reset_index(drop=True)


def benchmark_c_base_required_columns() -> list[str]:
    return [
        "fixed_regret",
        "prediction_regret",
        "learned_regret",
        "spectral_regret",
        "neural_regret",
        "prediction_angle",
        "learned_scale_ratio",
    ]


def benchmark_c_lasso_sof_required_columns() -> list[str]:
    return [
        "lasso_sof_regret",
        "lasso_sof_cost",
        "lasso_sof_pred_mse",
        "lasso_sof_avg_leaf_support",
        "lasso_sof_selected_count",
    ]


def run_benchmark_c(
    existing: pd.DataFrame | None = None,
    existing_boundary: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, float | int | str]] = []
    boundary_curve: pd.DataFrame | None = None
    existing_lookup = build_existing_row_lookup(existing, ("replication",))
    boundary_has_lasso_sof = existing_boundary is not None and "lasso_sof_choose_project_0" in existing_boundary.columns

    for replication in range(20):
        existing_row = existing_lookup.get((replication,))
        if (
            row_has_columns(existing_row, benchmark_c_base_required_columns() + benchmark_c_lasso_sof_required_columns())
            and not (replication == 0 and not boundary_has_lasso_sof)
        ):
            rows.append(dict(existing_row))
            if replication == 0 and existing_boundary is not None:
                boundary_curve = existing_boundary.copy()
            continue

        seed = 3000 + replication
        rng = np.random.default_rng(seed)
        x_train, rewards_train, _, _ = generate_benchmark_c(rng, 300)
        x_val, rewards_val, mean_rewards_val, _ = generate_benchmark_c(rng, 150)
        x_test, rewards_test, mean_rewards_test, signal_test = generate_benchmark_c(rng, 3000)

        x_train_std, x_val_std = standardize(x_train, x_val)
        _, x_test_std = standardize(x_train, x_test)

        lasso_selected, lasso_alpha, lasso_selection_mse = select_multitarget_lasso_features(
            x_train_std,
            rewards_train,
            x_val_std,
            rewards_val,
        )
        lasso_sof_metrics, lasso_sof_decisions = evaluate_allocation_stochopt_forest(
            x_train_std[:, lasso_selected],
            rewards_train,
            x_test_std[:, lasso_selected],
            rewards_test,
            mean_rewards_test,
            seed=110_000 + replication,
        )

        need_full_row = not row_has_columns(existing_row, benchmark_c_base_required_columns())
        need_boundary_curve = replication == 0 and not boundary_has_lasso_sof

        if need_full_row or need_boundary_curve:
            prediction_parameters, prediction_objective = learn_rotated_metric_allocation(
                x_train_std,
                rewards_train,
                objective_kind="prediction",
                fit_size=120,
                reg=0.005,
                maxfev=220,
            )
            decision_parameters, decision_objective = learn_rotated_metric_allocation(
                x_train_std,
                rewards_train,
                objective_kind="decision",
                initial_parameters=prediction_parameters,
                fit_size=120,
                reg=0.005,
                maxfev=220,
            )
            spectral_parameters, spectral_objective = learn_spectral_mixture_allocation(
                x_train_std,
                rewards_train,
                objective_kind="decision",
                fit_size=120,
                reg=0.005,
                maxfev=220,
            )
            neural_prediction_parameters, neural_prediction_objective = learn_neural_embedding_allocation(
                x_train_std,
                rewards_train,
                objective_kind="prediction",
                fit_size=120,
                reg=0.005,
                maxfev=220,
            )
            neural_parameters, neural_objective = learn_neural_embedding_allocation(
                x_train_std,
                rewards_train,
                objective_kind="decision",
                initial_parameters=neural_prediction_parameters,
                fit_size=120,
                reg=0.005,
                maxfev=220,
            )

            oracle_cost = float(np.mean(-np.max(mean_rewards_test, axis=1)))
            global_mean_rewards = rewards_train.mean(axis=0)
            global_action = int(np.argmax(global_mean_rewards))
            global_cost = float(np.mean(-rewards_test[:, global_action]))
            global_pred_mse = float(np.mean((np.tile(global_mean_rewards, (len(x_test), 1)) - mean_rewards_test) ** 2))

            point_design = np.column_stack([np.ones(len(x_train)), x_train])
            beta_0, *_ = np.linalg.lstsq(point_design, rewards_train[:, 0], rcond=None)
            beta_1, *_ = np.linalg.lstsq(point_design, rewards_train[:, 1], rcond=None)
            point_test_design = np.column_stack([np.ones(len(x_test)), x_test])
            point_reward_0 = point_test_design @ beta_0
            point_reward_1 = point_test_design @ beta_1
            point_action = (point_reward_1 > point_reward_0).astype(int)
            point_cost = float(np.mean(-rewards_test[np.arange(len(x_test)), point_action]))
            point_pred_rewards = np.column_stack([point_reward_0, point_reward_1])
            point_pred_mse = float(np.mean((point_pred_rewards - mean_rewards_test) ** 2))

            fixed_metrics, fixed_decisions = evaluate_allocation_transform(
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
            prediction_metrics, prediction_decisions = evaluate_allocation_transform(
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
            )
            learned_metrics, learned_decisions = evaluate_allocation_transform(
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
            spectral_metrics, spectral_decisions = evaluate_allocation_direct_kernel(
                x_train_std,
                rewards_train,
                x_val_std,
                rewards_val,
                mean_rewards_val,
                x_test_std,
                rewards_test,
                mean_rewards_test,
                kernel_kind="spectral_mixture",
                kernel_payload=spectral_parameters,
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
            row = {
                "benchmark": "C",
                "replication": replication,
                "prediction_metric_objective": prediction_objective,
                "decision_metric_objective": decision_objective,
                "spectral_metric_objective": spectral_objective,
                "neural_prediction_objective": neural_prediction_objective,
                "neural_metric_objective": neural_objective,
                "global_cost": global_cost,
                "global_regret": global_cost - oracle_cost,
                "global_pred_mse": global_pred_mse,
                "point_cost": point_cost,
                "point_regret": point_cost - oracle_cost,
                "point_pred_mse": point_pred_mse,
                "fixed_cost": fixed_metrics["cost"],
                "fixed_regret": fixed_metrics["regret"],
                "fixed_pred_mse": fixed_metrics["pred_mse"],
                "fixed_bandwidth": fixed_metrics["decision_bandwidth"],
                "prediction_cost": prediction_metrics["cost"],
                "prediction_regret": prediction_metrics["regret"],
                "prediction_pred_mse": prediction_metrics["pred_mse"],
                "prediction_bandwidth": prediction_metrics["decision_bandwidth"],
                "learned_cost": learned_metrics["cost"],
                "learned_regret": learned_metrics["regret"],
                "learned_pred_mse": learned_metrics["pred_mse"],
                "learned_bandwidth": learned_metrics["decision_bandwidth"],
                "spectral_cost": spectral_metrics["cost"],
                "spectral_regret": spectral_metrics["regret"],
                "spectral_pred_mse": spectral_metrics["pred_mse"],
                "spectral_bandwidth": spectral_metrics["decision_bandwidth"],
                "spectral_dominant_weight": spectral_metrics["spectral_dominant_weight"],
                "spectral_avg_frequency_norm": spectral_metrics["spectral_avg_frequency_norm"],
                "neural_cost": neural_metrics["cost"],
                "neural_regret": neural_metrics["regret"],
                "neural_pred_mse": neural_metrics["pred_mse"],
                "neural_bandwidth": neural_metrics["decision_bandwidth"],
                "neural_parameter_norm": neural_metrics["neural_parameter_norm"],
                "oracle_cost": oracle_cost,
                "prediction_angle": prediction_metrics["angle"],
                "learned_angle": learned_metrics["angle"],
                "prediction_scale_ratio": prediction_metrics["scale_ratio"],
                "learned_scale_ratio": learned_metrics["scale_ratio"],
            }
        else:
            row = dict(existing_row)

        if replication == 0:
            oracle_action = np.argmax(mean_rewards_test, axis=1)
            bins = np.linspace(signal_test.min(), signal_test.max(), 25)
            curve_rows = []
            for left, right in zip(bins[:-1], bins[1:]):
                mask = (signal_test >= left) & (signal_test < right)
                if not mask.any():
                    continue
                curve_rows.append(
                    {
                        "u_center": float((left + right) / 2.0),
                        "oracle_choose_project_0": float(np.mean(oracle_action[mask] == 0)),
                        "point_choose_project_0": float(np.mean(point_action[mask] == 0)),
                        "fixed_choose_project_0": float(np.mean(fixed_decisions[mask] == 0)),
                        "prediction_choose_project_0": float(np.mean(prediction_decisions[mask] == 0)),
                        "learned_choose_project_0": float(np.mean(learned_decisions[mask] == 0)),
                        "spectral_choose_project_0": float(np.mean(spectral_decisions[mask] == 0)),
                        "neural_choose_project_0": float(np.mean(neural_decisions[mask] == 0)),
                        "lasso_sof_choose_project_0": float(np.mean(lasso_sof_decisions[mask] == 0)),
                    }
                )
            boundary_curve = pd.DataFrame(curve_rows)

        row.update(
            {
                "lasso_sof_cost": float(lasso_sof_metrics["cost"]),
                "lasso_sof_regret": float(lasso_sof_metrics["regret"]),
                "lasso_sof_pred_mse": float(lasso_sof_metrics["pred_mse"]),
                "lasso_sof_avg_leaf_support": float(lasso_sof_metrics["avg_leaf_support"]),
                "lasso_sof_selected_count": int(len(lasso_selected)),
                "lasso_sof_alpha": lasso_alpha,
                "lasso_sof_selection_mse": lasso_selection_mse,
            }
        )
        rows.append(row)

    return pd.DataFrame(rows), boundary_curve if boundary_curve is not None else pd.DataFrame()


def benchmark_a_exponents(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    grouped = frame.groupby(["d", "k"])
    for (d, k), group in grouped:
        summary = (
            group.groupby("n_train")[
                [
                    "fixed_regret",
                    "pca_regret",
                    "lasso_regret",
                    "prediction_regret",
                    "learned_regret",
                    "sof_regret",
                    "lasso_sof_regret",
                ]
            ]
            .mean()
            .reset_index()
        )
        log_n = np.log(summary["n_train"].to_numpy())
        for method in [
            "fixed_regret",
            "pca_regret",
            "lasso_regret",
            "prediction_regret",
            "learned_regret",
            "sof_regret",
            "lasso_sof_regret",
        ]:
            log_regret = np.log(np.clip(summary[method].to_numpy(), 1e-8, None))
            slope, intercept = np.polyfit(log_n, log_regret, 1)
            rows.append(
                {
                    "d": d,
                    "k": k,
                    "method": method.replace("_regret", ""),
                    "slope": float(slope),
                    "effective_exponent": float(-slope),
                    "intercept": float(intercept),
                }
            )
    return pd.DataFrame(rows)


def benchmark_a_k3_curves(frame: pd.DataFrame) -> pd.DataFrame:
    subset = frame[(frame["k"] == 3) & (frame["d"].isin([5, 20, 100]))]
    summary = (
        subset.groupby(["d", "n_train"])[
            [
                "fixed_regret",
                "pca_regret",
                "lasso_regret",
                "prediction_regret",
                "learned_regret",
                "sof_regret",
                "lasso_sof_regret",
            ]
        ]
        .mean()
        .reset_index()
        .sort_values(["d", "n_train"])
    )
    return summary


def benchmark_a_importance_profile(
    frame: pd.DataFrame,
    d: int = 20,
    k: int = 3,
    n_train: int = 400,
) -> pd.DataFrame:
    subset = frame[(frame["d"] == d) & (frame["k"] == k) & (frame["n_train"] == n_train)]
    if subset.empty:
        return pd.DataFrame()

    feature_rows = []
    lasso_cols = [f"lasso_selected_{j}" for j in range(d)]
    prediction_cols = [f"prediction_scale_{j}" for j in range(d)]
    learned_cols = [f"learned_scale_{j}" for j in range(d)]
    sof_cols = [f"sof_split_{j}" for j in range(d)]
    lasso_sof_cols = [f"lasso_sof_split_{j}" for j in range(d)]
    lasso_profile = subset[lasso_cols].fillna(0.0).mean().to_numpy(dtype=float)
    prediction_profile = subset[prediction_cols].mean().to_numpy(dtype=float)
    learned_profile = subset[learned_cols].mean().to_numpy(dtype=float)
    sof_profile = subset[sof_cols].mean().to_numpy(dtype=float)
    lasso_sof_profile = subset[lasso_sof_cols].mean().to_numpy(dtype=float)
    fixed_profile = np.ones(d, dtype=float)
    if lasso_profile.sum() <= 0.0:
        lasso_profile = np.ones(d, dtype=float)

    profiles = {
        "Fixed kernel": normalized_profile(fixed_profile),
        "LASSO + kernel": normalized_profile(lasso_profile),
        "Pred.-trained repr.": normalized_profile(prediction_profile),
        "Decision-trained repr.": normalized_profile(learned_profile),
        "SOF": normalized_profile(sof_profile),
        "LASSO->SOF": normalized_profile(lasso_sof_profile),
    }
    for method, profile in profiles.items():
        for feature_index, value in enumerate(profile):
            feature_rows.append(
                {
                    "method": method,
                    "feature_index": feature_index,
                    "importance_share": float(value),
                    "is_signal": feature_index < k,
                    "d": d,
                    "k": k,
                    "n_train": n_train,
                }
            )
    return pd.DataFrame(feature_rows)
