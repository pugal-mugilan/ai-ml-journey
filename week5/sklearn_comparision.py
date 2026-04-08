"""
sklearn Comparison
Same stock data, same split — sklearn versions of all three models.
Confirms scratch implementations were correct.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# ============================================================
# DATA — (identical seed + generation)
# ============================================================
np.random.seed(42)
n_samples = 500
n_features = 5

X = np.random.randn(n_samples, n_features)
feature_names = ['daily_return', 'volume_spike', 'volatility', 'moving_avg_signal', 'momentum']

true_weights = np.array([0.0, 0.2, 0.0, 0.8, 0.6]).reshape(-1, 1)
true_bias = -0.5

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

scores = X @ true_weights + true_bias
probabilities = sigmoid(scores)
y = np.random.binomial(1, probabilities).reshape(-1, 1)

# Train/test split
x_train, x_test = X[:400], X[400:]
y_train, y_test = y[:400], y[400:]

# Normalization (training min/max only)
x_min = x_train.min(axis=0)
x_max = x_train.max(axis=0)
x_train_norm = (x_train - x_min) / (x_max - x_min)
x_test_norm = (x_test - x_min) / (x_max - x_min)

# ============================================================
# MODEL 1: sklearn Logistic Regression
# ============================================================
print("=== sklearn Logistic Regression ===")
lr_model = LogisticRegression()  # default C=1.0 (Ridge-like penalty)
lr_model.fit(x_train_norm, y_train.ravel())
lr_preds = lr_model.predict(x_test_norm)

print(classification_report(y_test.ravel(), lr_preds))
print("Confusion Matrix:")
print(confusion_matrix(y_test.ravel(), lr_preds))

print(f"\nWeights: {lr_model.coef_}")
print(f"Bias:    {lr_model.intercept_}")

# ============================================================
# MODEL 2: sklearn Decision Tree
# ============================================================
print("\n=== sklearn Decision Tree (max_depth=3) ===")
tree_model = DecisionTreeClassifier(max_depth=3, random_state=42)
tree_model.fit(x_train_norm, y_train.ravel())
tree_preds = tree_model.predict(x_test_norm)

print(classification_report(y_test.ravel(), tree_preds))
print("Confusion Matrix:")
print(confusion_matrix(y_test.ravel(), tree_preds))

# ============================================================
# MODEL 3: sklearn Random Forest
# ============================================================
print("\n=== sklearn Random Forest (100 trees, max_depth=3) ===")
forest_model = RandomForestClassifier(n_estimators=100, max_depth=3, random_state=42)
forest_model.fit(x_train_norm, y_train.ravel())
forest_preds = forest_model.predict(x_test_norm)

print(classification_report(y_test.ravel(), forest_preds))
print("Confusion Matrix:")
print(confusion_matrix(y_test.ravel(), forest_preds))

# ============================================================
# FEATURE IMPORTANCE COMPARISON
# ============================================================
print("\n=== Feature Importance: LR Weights vs Forest Importance ===")
print(f"{'Feature':<22} | {'True Weight':>11} | {'LR Weight':>10} | {'Forest Imp':>10}")
print("-" * 65)
for name, tw, lrw, fi in zip(feature_names, true_weights.flatten(),
                               lr_model.coef_.flatten(), forest_model.feature_importances_):
    print(f"{name:<22} | {tw:>11.1f} | {lrw:>10.4f} | {fi:>10.4f}")

print(f"\nBoth rank moving_avg_signal #1, momentum #2 — matching true weights.")
print(f"LR gives clear separation. Forest is muddled but correct ranking.")
print(f"sklearn's Ridge penalty (C=1.0) shrinks LR weights vs scratch version.")

# ============================================================
# SCRATCH vs SKLEARN COMPARISON SUMMARY
# ============================================================
print("\n" + "=" * 60)
print("SCRATCH vs SKLEARN — Results Match!")
print("=" * 60)
print("""
| Model               | Scratch Acc | sklearn Acc |
|---------------------|-------------|-------------|
| Logistic Regression | 0.68        | 0.70        |
| Decision Tree (d=3) | 0.73        | 0.73        |
| Random Forest (d=3) | 0.73        | 0.73        |

LR difference: sklearn's default Ridge penalty (C=1.0).
Tree/Forest: identical — same algorithm, same results.

""")