"""A small second-order gradient boosting regressor implemented with NumPy.

This educational implementation supports squared-error regression, numeric
features, greedy decision-tree splits, and L1/L2 leaf regularization.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


def _soft_threshold(value: float, strength: float) -> float:
    """Shrink a value toward zero by ``strength`` (the L1 penalty)."""
    return float(np.sign(value) * max(abs(value) - strength, 0.0))


@dataclass
class Node:
    """One node in a regression tree."""
    value: float
    feature: Optional[int] = None
    threshold: Optional[float] = None
    left: Optional["Node"] = None
    right: Optional["Node"] = None

    @property
    def is_leaf(self) -> bool:
        return self.feature is None


class DecisionTree:
    """Greedy tree fitted to gradient and Hessian statistics."""

    def __init__(
        self,
        max_depth: int = 2,
        min_samples_split: int = 2,
        reg_lambda: float = 1.0,
        reg_alpha: float = 0.0,
        gamma: float = 0.0,
        colsample_bytree: float = 1.0,
        random_state: Optional[int] = None,
    ) -> None:
        if max_depth < 0:
            raise ValueError("max_depth must be non-negative")
        if min_samples_split < 2:
            raise ValueError("min_samples_split must be at least 2")
        if reg_lambda < 0 or reg_alpha < 0 or gamma < 0:
            raise ValueError("regularization strengths and gamma must be non-negative")
        if not 0 < colsample_bytree <= 1:
            raise ValueError("colsample_bytree must be in (0, 1]")
        self.max_depth = int(max_depth)
        self.min_samples_split = int(min_samples_split)
        self.reg_lambda = float(reg_lambda)
        self.reg_alpha = float(reg_alpha)
        self.gamma = float(gamma)
        self.colsample_bytree = float(colsample_bytree)
        self.random_state = random_state
        self.root: Optional[Node] = None
        self.n_features_in_: Optional[int] = None

    def _score(self, gradients: np.ndarray, hessians: np.ndarray) -> float:
        grad_sum = float(np.sum(gradients))
        hess_sum = float(np.sum(hessians))
        denom = hess_sum + self.reg_lambda
        if denom <= 0:
            return 0.0
        shrunk = _soft_threshold(grad_sum, self.reg_alpha)
        return shrunk * shrunk / denom

    def _leaf_value(self, gradients: np.ndarray, hessians: np.ndarray) -> float:
        grad_sum = float(np.sum(gradients))
        hess_sum = float(np.sum(hessians))
        denom = hess_sum + self.reg_lambda
        if denom <= 0:
            return 0.0
        return -_soft_threshold(grad_sum, self.reg_alpha) / denom

    def fit(self, X: np.ndarray, gradients: np.ndarray, hessians: np.ndarray) -> "DecisionTree":
        X = np.asarray(X, dtype=float)
        gradients = np.asarray(gradients, dtype=float).reshape(-1)
        hessians = np.asarray(hessians, dtype=float).reshape(-1)
        if X.ndim != 2 or X.shape[0] == 0:
            raise ValueError("X must be a non-empty two-dimensional array")
        if len(gradients) != len(X) or len(hessians) != len(X):
            raise ValueError("X, gradients, and hessians must have the same row count")
        if not np.all(np.isfinite(X)) or not np.all(np.isfinite(gradients)) or not np.all(np.isfinite(hessians)):
            raise ValueError("inputs must contain only finite values")
        if np.any(hessians < 0):
            raise ValueError("hessians must be non-negative")
        self.n_features_in_ = X.shape[1]
        rng = np.random.default_rng(self.random_state)
        feature_count = max(1, int(np.ceil(self.n_features_in_ * self.colsample_bytree)))
        chosen_features = np.sort(rng.choice(self.n_features_in_, feature_count, replace=False))

        def grow(rows: np.ndarray, depth: int) -> Node:
            g = gradients[rows]
            h = hessians[rows]
            leaf = Node(value=self._leaf_value(g, h))
            if depth >= self.max_depth or len(rows) < self.min_samples_split or len(rows) < 2:
                return leaf
            parent_score = self._score(g, h)
            best_gain = self.gamma
            best_split = None
            for feature in chosen_features:
                values = X[rows, feature]
                unique_values = np.unique(values)
                if len(unique_values) < 2:
                    continue
                thresholds = (unique_values[:-1] + unique_values[1:]) / 2.0
                for threshold in thresholds:
                    left_mask = values <= threshold
                    if not np.any(left_mask) or np.all(left_mask):
                        continue
                    left_rows = rows[left_mask]
                    right_rows = rows[~left_mask]
                    gain = 0.5 * (
                        self._score(gradients[left_rows], hessians[left_rows])
                        + self._score(gradients[right_rows], hessians[right_rows])
                        - parent_score
                    )
                    if gain > best_gain + 1e-15:
                        best_gain = gain
                        best_split = (int(feature), float(threshold), left_rows, right_rows)
            if best_split is None:
                return leaf
            feature, threshold, left_rows, right_rows = best_split
            return Node(
                value=leaf.value,
                feature=feature,
                threshold=threshold,
                left=grow(left_rows, depth + 1),
                right=grow(right_rows, depth + 1),
            )

        self.root = grow(np.arange(len(X)), 0)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.root is None or self.n_features_in_ is None:
            raise RuntimeError("Call fit before using predict")
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be a two-dimensional array")
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"X must have exactly {self.n_features_in_} features")
        result = np.empty(X.shape[0], dtype=float)
        for row_number, row in enumerate(X):
            node = self.root
            while not node.is_leaf:
                assert node.feature is not None and node.threshold is not None
                node = node.left if row[node.feature] <= node.threshold else node.right
                assert node is not None
            result[row_number] = node.value
        return result


class GradientBoostedRegressor:
    """Gradient-boosted regression using second-order tree approximations."""

    def __init__(
        self,
        n_estimators: int = 20,
        learning_rate: float = 0.1,
        max_depth: int = 2,
        min_samples_split: int = 2,
        reg_lambda: float = 1.0,
        reg_alpha: float = 0.0,
        gamma: float = 0.0,
        colsample_bytree: float = 1.0,
        random_state: Optional[int] = 42,
    ) -> None:
        if n_estimators < 1:
            raise ValueError("n_estimators must be at least 1")
        if not 0 < learning_rate <= 1:
            raise ValueError("learning_rate must be in (0, 1]")
        self.n_estimators = int(n_estimators)
        self.learning_rate = float(learning_rate)
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.reg_lambda = reg_lambda
        self.reg_alpha = reg_alpha
        self.gamma = gamma
        self.colsample_bytree = colsample_bytree
        self.random_state = random_state
        self.base_score_: Optional[float] = None
        self.n_features_in_: Optional[int] = None
        self.trees_: list[DecisionTree] = []
        self.loss_history_: list[float] = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GradientBoostedRegressor":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)
        if X.ndim != 2:
            raise ValueError("X must be a two-dimensional array")
        if X.shape[0] == 0 or X.shape[0] != len(y):
            raise ValueError("X and y must contain the same non-zero number of rows")
        if X.shape[1] == 0:
            raise ValueError("X must contain at least one feature")
        if not np.all(np.isfinite(X)) or not np.all(np.isfinite(y)):
            raise ValueError("X and y must contain finite values")
        self.n_features_in_ = X.shape[1]
        self.base_score_ = float(np.mean(y))
        current = np.full(len(y), self.base_score_, dtype=float)
        self.trees_ = []
        self.loss_history_ = [float(0.5 * np.mean((y - current) ** 2))]
        for iteration in range(self.n_estimators):
            gradients = current - y
            hessians = np.ones_like(y)
            tree = DecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                reg_lambda=self.reg_lambda,
                reg_alpha=self.reg_alpha,
                gamma=self.gamma,
                colsample_bytree=self.colsample_bytree,
                random_state=None if self.random_state is None else self.random_state + iteration,
            ).fit(X, gradients, hessians)
            current += self.learning_rate * tree.predict(X)
            self.trees_.append(tree)
            self.loss_history_.append(float(0.5 * np.mean((y - current) ** 2)))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.base_score_ is None or self.n_features_in_ is None:
            raise RuntimeError("Call fit before using predict")
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be a two-dimensional array")
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"X must have exactly {self.n_features_in_} features")
        predictions = np.full(X.shape[0], self.base_score_, dtype=float)
        for tree in self.trees_:
            predictions += self.learning_rate * tree.predict(X)
        return predictions
