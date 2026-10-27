# Technical Report: Custom Gradient Boosting Optimizer

## 1. Objective

The goal is to implement a compact gradient-boosted regressor from first principles. Each new tree models the current prediction errors, and its output is added to the existing prediction with a learning-rate multiplier.

## 2. Loss, gradient, and Hessian

For target `y_i` and current prediction `p_i`, use half squared error:

`L_i = 0.5 * (y_i - p_i)^2`

Differentiating with respect to the prediction gives:

`g_i = p_i - y_i`

The second derivative is constant:

`h_i = 1`

The gradient indicates the local direction of change. The Hessian describes local curvature. For squared error, the Hessian is one for every observation.

## 3. Regularized leaf weight

For a candidate leaf, define `G = sum(g_i)` and `H = sum(h_i)`. The local second-order objective for leaf weight `w` is approximated by:

`Q(w) = G*w + 0.5*(H + lambda)*w^2 + alpha*abs(w)`

Here `lambda` is the L2 penalty and `alpha` is the L1 penalty. The L1 term creates a soft-threshold operation:

`S(G, alpha) = sign(G) * max(abs(G) - alpha, 0)`

The minimizing leaf value is:

`w* = -S(G, alpha) / (H + lambda)`

L2 regularization enlarges the denominator, shrinking leaf values. L1 regularization can set small aggregate gradients to zero. The implementation guards against non-positive denominators.

## 4. Split selection

For a node, the implementation calculates the regularized score:

`Score = S(G, alpha)^2 / (H + lambda)`

For a candidate threshold, the gain is:

`Gain = 0.5 * (Score_left + Score_right - Score_parent)`

Each feature's unique sorted values define candidate thresholds at adjacent-value midpoints. The split with the largest gain is chosen, provided that its gain is greater than `gamma`. `max_depth` limits tree depth, while `min_samples_split` prevents splitting very small nodes. Optional feature subsampling can reduce the features considered by each tree.

## 5. Boosting procedure

1. Initialize predictions to the mean of the target values.
2. Compute gradients as `prediction - target` and Hessians as ones.
3. Grow a tree from those gradient and Hessian values.
4. Update predictions using `prediction += learning_rate * tree_prediction`.
5. Repeat for the requested number of estimators.

The loss history records half mean squared error before training and after each boosting round.

## 6. Validation and tests

The test suite checks soft-threshold behavior, the analytical leaf-weight calculation, prediction shape and finiteness, repeatability with a fixed random seed, errors for prediction before fitting, mismatched row counts, incorrect feature counts, and invalid learning rates.

Run `python -m pytest -q` to execute the tests.

## 7. Data note and limitations

The assignment description supplies five explicit target values and also names `sin(x1) * cos(x2)`. These do not agree for every point. This project uses the explicit targets in the table and flags the discrepancy for confirmation. The implementation is intended for learning and small demonstrations, not as a replacement for established gradient-boosting libraries.
