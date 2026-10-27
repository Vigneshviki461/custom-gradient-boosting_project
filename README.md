# Custom Gradient Boosting Optimizer

## What this project does

This project implements a small gradient-boosted regression model using Python and NumPy. It does not call XGBoost, LightGBM, or scikit-learn to build trees. The tree split and leaf calculations are implemented in this project.

## Files

- `gradient_boosting.py` — tree nodes, split search, regularized leaf values, and boosting loop.
- `test_gradient_boosting.py` — unit tests for the core calculations and input checks.
- `requirements.txt` — required Python packages.
- `technical_report.md` — algorithm explanation and mathematical derivation.

## Assignment sample data

| Point | x1 | x2 | target |
|---|---:|---:|---:|
| P1 | 0.1 | 0.2 | 0.19 |
| P2 | 0.5 | 0.8 | 0.65 |
| P3 | 1.2 | 0.3 | 0.28 |
| P4 | 1.5 | 1.1 | 0.95 |
| P5 | 0.8 | 1.5 | 0.05 |

The task also mentions `f(x1, x2) = sin(x1) * cos(x2)`, but that formula does not reproduce all the listed targets. The code and tests use the five explicit target values. If the assessor expects formula-generated targets, ask for clarification rather than silently mixing the two definitions.

## Run locally

Use Python 3.10 or newer. In a terminal opened inside this folder:

```bash
python -m pip install -r requirements.txt
python -m pytest -q
```

A simple example:

```python
import numpy as np
from gradient_boosting import GradientBoostedRegressor

X = np.array([[0.1, 0.2], [0.5, 0.8], [1.2, 0.3], [1.5, 1.1], [0.8, 1.5]])
y = np.array([0.19, 0.65, 0.28, 0.95, 0.05])

model = GradientBoostedRegressor(n_estimators=10, learning_rate=0.1, max_depth=2)
model.fit(X, y)
print(model.predict(X))
print(model.loss_history_)
```

## Important limitations

This is an educational implementation intended for a tiny synthetic dataset. It supports numeric features and squared-error regression. It does not implement missing-value routing, categorical variables, parallel training, or early stopping. Five observations are not enough to establish generalization performance; use a larger dataset and a held-out test set for real model evaluation.
