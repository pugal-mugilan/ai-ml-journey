"""
 Decision Trees + Random Forests
Key concepts: Gini impurity, recursive splitting, overfitting, pruning, bagging, ensemble voting
A completely different paradigm — NO gradient descent
"""

import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ============================================================
# DATA — Same tumor data 
# ============================================================
np.random.seed(42)
benign_sizes = np.random.normal(2, 0.8, 50)
malignant_sizes = np.random.normal(5, 0.8, 50)
X = np.concatenate([benign_sizes, malignant_sizes]).reshape(-1, 1)
y = np.concatenate([np.zeros(50), np.ones(50)])

# Train/test split
X_train, X_test = X[:80], X[80:]
y_train, y_test = y[:80], y[80:]

# ============================================================
# GINI IMPURITY — "How mixed is this group?"
# ============================================================
def gini(labels):
    """Gini = 1 - p1² - p2². 0 = pure, 0.5 = max mess."""
    if len(labels) == 0:
        return 0
    p1 = np.mean(labels == 1)
    p0 = 1 - p1
    return 1 - p1**2 - p0**2

# Demonstrate Gini values
print("=== Gini Impurity Examples ===")
print(f"50/50 mix:  Gini = {gini(np.array([0,0,0,0,0,1,1,1,1,1])):.3f}  (maximum mess)")
print(f"96/4 split: Gini = {gini(np.array([0]*96 + [1]*4)):.3f}  (nearly pure)")
print(f"100/0 pure: Gini = {gini(np.array([0]*100)):.3f}  (perfectly pure)")

# ============================================================
# FINDING BEST SPLIT — Try every threshold, pick lowest weighted Gini
# ============================================================
def find_best_split(X_col, y):
    """Try every midpoint between sorted values, return best split."""
    sorted_vals = np.unique(X_col)
    best_gini = float('inf')
    best_threshold = None

    for i in range(len(sorted_vals) - 1):
        threshold = (sorted_vals[i] + sorted_vals[i + 1]) / 2
        left_mask = X_col <= threshold
        right_mask = ~left_mask

        left_gini = gini(y[left_mask])
        right_gini = gini(y[right_mask])

        n_left = left_mask.sum()
        n_right = right_mask.sum()
        n_total = len(y)

        weighted = (n_left / n_total * left_gini) + (n_right / n_total * right_gini)

        if weighted < best_gini:
            best_gini = weighted
            best_threshold = threshold

    return best_threshold, best_gini

threshold, weighted_gini = find_best_split(X_train.flatten(), y_train)
print(f"\n=== Best Split ===")
print(f"Split at: {threshold:.2f} cm | Weighted Gini: {weighted_gini:.4f}")

# ============================================================
# DECISION TREE — Overfit vs Pruned (using sklearn)
# ============================================================
print("\n=== Decision Tree: Overfit vs Pruned ===")

# Overfit tree — no limits
overfit_tree = DecisionTreeClassifier(random_state=42)
overfit_tree.fit(X_train, y_train)
print(f"Overfit Tree:")
print(f"  Train accuracy: {overfit_tree.score(X_train, y_train):.3f}")
print(f"  Test accuracy:  {overfit_tree.score(X_test, y_test):.3f}")
print(f"  Depth: {overfit_tree.get_depth()}, Leaves: {overfit_tree.get_n_leaves()}")

# Pruned tree — max_depth=2
pruned_tree = DecisionTreeClassifier(max_depth=2, random_state=42)
pruned_tree.fit(X_train, y_train)
print(f"\nPruned Tree (max_depth=2):")
print(f"  Train accuracy: {pruned_tree.score(X_train, y_train):.3f}")
print(f"  Test accuracy:  {pruned_tree.score(X_test, y_test):.3f}")
print(f"  Depth: {pruned_tree.get_depth()}, Leaves: {pruned_tree.get_n_leaves()}")

# Show the questions the pruned tree asks
print(f"\nPruned Tree Questions:")
print(export_text(pruned_tree, feature_names=['tumor_size']))

# ============================================================
# RANDOM FOREST — 100 trees voting
# ============================================================
print("=== Random Forest (100 trees, max_depth=2) ===")
forest = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=42)
forest.fit(X_train, y_train)
print(f"Train accuracy: {forest.score(X_train, y_train):.3f}")
print(f"Test accuracy:  {forest.score(X_test, y_test):.3f}")

# ============================================================
# COMPARISON — All three models
# ============================================================
print("\n=== Model Comparison ===")
print(f"{'Model':<25} | {'Train Acc':>10} | {'Test Acc':>10}")
print("-" * 52)
print(f"{'Overfit Tree':<25} | {overfit_tree.score(X_train, y_train):>10.3f} | {overfit_tree.score(X_test, y_test):>10.3f}")
print(f"{'Pruned Tree (depth=2)':<25} | {pruned_tree.score(X_train, y_train):>10.3f} | {pruned_tree.score(X_test, y_test):>10.3f}")
print(f"{'Random Forest':<25} | {forest.score(X_train, y_train):>10.3f} | {forest.score(X_test, y_test):>10.3f}")

print("\nKey takeaway: Overfit tree has perfect training (1.0) but worse test accuracy.")
print("Pruned tree sacrifices some training accuracy for better generalization.")
print("Random forest: 100 trees vote → noise gets outvoted.")

# ============================================================
# TWO PARADIGMS
# ============================================================
print("\n=== Two ML Paradigms ===")
print("Optimization-based: Linear Reg → Logistic Reg → Neural Nets")
print("  How: gradient descent — nudge weights to minimize cost")
print("  Fix overfitting: regularization (Ridge/Lasso)")
print()
print("Tree-based: Decision Tree → Random Forest → XGBoost")
print("  How: recursive splitting — ask best yes/no questions")
print("  Fix overfitting: pruning (max_depth) + ensembles")