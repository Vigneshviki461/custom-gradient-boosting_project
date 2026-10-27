
import numpy as np
from gradient_boosting import GradientBoostedRegressor

X = np.array([
    [0.1, 0.2],
    [0.5, 0.8],
    [1.2, 0.3],
    [1.5, 1.1],
    [0.8, 1.5],
])

y = np.array([0.19, 0.65, 0.28, 0.95, 0.05])

model = GradientBoostedRegressor(
    n_estimators=10,
    learning_rate=0.1,
    max_depth=2,
)

model.fit(X, y)
predictions = model.predict(X)

print("CUSTOM GRADIENT BOOSTING")
print("------------------------")
print("Actual values:", y)
print("Predictions:", np.round(predictions, 4))
print("Loss history:", np.round(model.loss_history_, 6))
print("Number of trees:", len(model.trees_))
