"""
 3-Model Comparison Pipeline
Dataset: Stock prediction (5 features) — UP (1) or DOWN (0)
Key discovery: Forest predicts better, LR explains better. Choose based on business need.
"""

import numpy as np

# ============================================================
# DATA GENERATION — Logistic regression in reverse
# ============================================================
np.random.seed(42)
n_samples = 500
n_features = 5

# Features: daily_return, volume_spike, volatility, moving_avg_signal, momentum
X = np.random.randn(n_samples, n_features)
feature_names = ['daily_return', 'volume_spike', 'volatility', 'moving_avg_signal', 'momentum']

# True weights — only 3 features matter
true_weights = np.array([0.0, 0.2, 0.0, 0.8, 0.6]).reshape(-1, 1)
true_bias = -0.5

# Generate labels: score → sigmoid → coin flip (realistic noise)
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

scores = X @ true_weights + true_bias
probabilities = sigmoid(scores)
y = np.random.binomial(1, probabilities).reshape(-1, 1)  # coin flip adds noise

print(f"Class distribution: {np.mean(y):.1%} class 1, {1-np.mean(y):.1%} class 0")

# ============================================================
# TRAIN/TEST SPLIT — 80/20
# ============================================================
x_train, x_test = X[:400], X[400:]
y_train, y_test = y[:400], y[400:]

# ============================================================
# NORMALIZATION — Critical for logistic regression
# ============================================================
# Use TRAINING min/max only — can't peek at test data
x_min = x_train.min(axis=0)  # axis=0 = per column
x_max = x_train.max(axis=0)
x_train_norm = (x_train - x_min) / (x_max - x_min)
x_test_norm = (x_test - x_min) / (x_max - x_min)  # same min/max!

# ============================================================
# MODEL 1: LOGISTIC REGRESSION FROM SCRATCH
# ============================================================
print("\n=== Model 1: Logistic Regression (Scratch) ===")

w = np.random.randn(n_features, 1) * 0.01
b = np.zeros((1, 1))
lr = 0.1
n = x_train_norm.shape[0]

for i in range(50000):
    z = x_train_norm @ w + b
    preds = sigmoid(z)
    error = preds - y_train
    cost = (-y_train * np.log(preds + 1e-8) - (1 - y_train) * np.log(1 - preds + 1e-8)).mean()
    dw = (1/n) * (x_train_norm.T @ error)
    db = error.mean()
    w = w - lr * dw
    b = b - lr * db

# Predictions
lr_train_preds = (sigmoid(x_train_norm @ w + b) >= 0.5).astype(int)
lr_test_preds = (sigmoid(x_test_norm @ w + b) >= 0.5).astype(int)

lr_train_acc = np.mean(lr_train_preds == y_train)
lr_test_acc = np.mean(lr_test_preds == y_test)

# Metrics
tp = np.sum((lr_test_preds.flatten() == 1) & (y_test.flatten() == 1))
fp = np.sum((lr_test_preds.flatten() == 1) & (y_test.flatten() == 0))
fn = np.sum((lr_test_preds.flatten() == 0) & (y_test.flatten() == 1))
tn = np.sum((lr_test_preds.flatten() == 0) & (y_test.flatten() == 0))

lr_precision = tp / (tp + fp) if (tp + fp) > 0 else 0
lr_recall = tp / (tp + fn) if (tp + fn) > 0 else 0
lr_f1 = 2 * (lr_precision * lr_recall) / (lr_precision + lr_recall) if (lr_precision + lr_recall) > 0 else 0

print(f"Train accuracy: {lr_train_acc:.2f}")
print(f"Test accuracy:  {lr_test_acc:.2f}")
print(f"Precision: {lr_precision:.2f} | Recall: {lr_recall:.2f} | F1: {lr_f1:.2f}")
print(f"Confusion: TP={tp}, FP={fp}, FN={fn}, TN={tn}")

# Feature weights
print(f"\nLR Weights:")
for name, weight in zip(feature_names, w.flatten()):
    print(f"  {name:20s}: {weight:.4f}")

# ============================================================
# MODEL 2: DECISION TREE (Gini splits, no gradient descent)
# ============================================================
print("\n=== Model 2: Decision Tree (Pruned, depth=3) ===")

# Simple Gini-based tree using sklearn (from-scratch tree in day4)
from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier(max_depth=3, random_state=42)
tree.fit(x_train_norm, y_train.ravel())

tree_train_preds = tree.predict(x_train_norm)
tree_test_preds = tree.predict(x_test_norm)

tree_train_acc = np.mean(tree_train_preds == y_train.ravel())
tree_test_acc = np.mean(tree_test_preds == y_test.ravel())

tp = np.sum((tree_test_preds == 1) & (y_test.ravel() == 1))
fp = np.sum((tree_test_preds == 1) & (y_test.ravel() == 0))
fn = np.sum((tree_test_preds == 0) & (y_test.ravel() == 1))
tn = np.sum((tree_test_preds == 0) & (y_test.ravel() == 0))

tree_precision = tp / (tp + fp) if (tp + fp) > 0 else 0
tree_recall = tp / (tp + fn) if (tp + fn) > 0 else 0
tree_f1 = 2 * (tree_precision * tree_recall) / (tree_precision + tree_recall) if (tree_precision + tree_recall) > 0 else 0

print(f"Train accuracy: {tree_train_acc:.2f}")
print(f"Test accuracy:  {tree_test_acc:.2f}")
print(f"Precision: {tree_precision:.2f} | Recall: {tree_recall:.2f} | F1: {tree_f1:.2f}")
print(f"Confusion: TP={tp}, FP={fp}, FN={fn}, TN={tn}")

# ============================================================
# MODEL 3: RANDOM FOREST (100 trees voting)
# ============================================================
print("\n=== Model 3: Random Forest (100 trees, depth=3) ===")

from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(n_estimators=100, max_depth=3, random_state=42)
forest.fit(x_train_norm, y_train.ravel())

forest_train_preds = forest.predict(x_train_norm)
forest_test_preds = forest.predict(x_test_norm)

forest_train_acc = np.mean(forest_train_preds == y_train.ravel())
forest_test_acc = np.mean(forest_test_preds == y_test.ravel())

tp = np.sum((forest_test_preds == 1) & (y_test.ravel() == 1))
fp = np.sum((forest_test_preds == 1) & (y_test.ravel() == 0))
fn = np.sum((forest_test_preds == 0) & (y_test.ravel() == 1))
tn = np.sum((forest_test_preds == 0) & (y_test.ravel() == 0))

forest_precision = tp / (tp + fp) if (tp + fp) > 0 else 0
forest_recall = tp / (tp + fn) if (tp + fn) > 0 else 0
forest_f1 = 2 * (forest_precision * forest_recall) / (forest_precision + forest_recall) if (forest_precision + forest_recall) > 0 else 0

print(f"Train accuracy: {forest_train_acc:.2f}")
print(f"Test accuracy:  {forest_test_acc:.2f}")
print(f"Precision: {forest_precision:.2f} | Recall: {forest_recall:.2f} | F1: {forest_f1:.2f}")
print(f"Confusion: TP={tp}, FP={fp}, FN={fn}, TN={tn}")

# Forest feature importance
print(f"\nForest Feature Importance:")
for name, imp in zip(feature_names, forest.feature_importances_):
    print(f"  {name:20s}: {imp:.4f}")

# ============================================================
# OVERFITTING EXPERIMENT — Deep tree vs Deep forest
# ============================================================
print("\n=== Overfitting Experiment ===")
deep_tree = DecisionTreeClassifier(random_state=42)  # no depth limit
deep_tree.fit(x_train_norm, y_train.ravel())

deep_forest = RandomForestClassifier(n_estimators=100, random_state=42)  # no depth limit
deep_forest.fit(x_train_norm, y_train.ravel())

print(f"{'Model':<20} | {'Train Acc':>10} | {'Test Acc':>10}")
print("-" * 48)
print(f"{'Deep Tree':<20} | {deep_tree.score(x_train_norm, y_train.ravel()):>10.4f} | {deep_tree.score(x_test_norm, y_test.ravel()):>10.4f}")
print(f"{'Deep Forest':<20} | {deep_forest.score(x_train_norm, y_train.ravel()):>10.4f} | {deep_forest.score(x_test_norm, y_test.ravel()):>10.4f}")
print("\nBoth memorize training (1.0), but forest recovers on test — noise gets outvoted.")

# ============================================================
# FINAL COMPARISON
# ============================================================
print("\n" + "=" * 60)
print("FINAL COMPARISON — Same data, three models")
print("=" * 60)
print(f"{'Model':<25} | {'Accuracy':>8} | {'Precision':>9} | {'Recall':>6} | {'F1':>6}")
print("-" * 65)
print(f"{'Logistic Regression':<25} | {lr_test_acc:>8.2f} | {lr_precision:>9.2f} | {lr_recall:>6.2f} | {lr_f1:>6.2f}")
print(f"{'Decision Tree (d=3)':<25} | {tree_test_acc:>8.2f} | {tree_precision:>9.2f} | {tree_recall:>6.2f} | {tree_f1:>6.2f}")
print(f"{'Random Forest (d=3)':<25} | {forest_test_acc:>8.2f} | {forest_precision:>9.2f} | {forest_recall:>6.2f} | {forest_f1:>6.2f}")
print(f"\nBaseline (predict majority): {np.mean(y_test):.2f}")
print(f"\nKey trade-off:")
print(f"  LR: lower accuracy, better interpretability (clear weights)")
print(f"  Forest: higher accuracy, worse interpretability (muddled importance)")