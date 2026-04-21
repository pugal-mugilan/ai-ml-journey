"""
Week 6 Day 6 — Full ML Pipeline
Load → EDA → PCA → Isolation Forest → Gradient Boosting → Compare

Dataset: 5-feature synthetic stock prediction data (500 samples, binary UP/DOWN)
Same dataset used throughout Weeks 5 and 6 for direct model comparison.
"""

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# ----------------------------------------------------------------------
# STEP 1 — Load, scale, split
# ----------------------------------------------------------------------

np.random.seed(42)
n_samples = 500

# Synthetic 5-feature stock dataset (matches Week 5/6 setup)
# 3 informative features + 2 noise features
X = np.random.randn(n_samples, 5)

# Label: a linear rule on the first 3 features + a bit of noise
# (This is why logistic regression wins on this data — it's linearly generated.)
logits = 1.2 * X[:, 0] - 0.8 * X[:, 1] + 0.5 * X[:, 2] + 0.3 * np.random.randn(n_samples)
y = (logits > 0).astype(int)

# Split FIRST, then scale (fit on train only — no leakage)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=100, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)      # use TRAIN params on test

print(f"Train shape: {X_train_scaled.shape}, Test shape: {X_test_scaled.shape}")
print(f"Class balance (train): {y_train.mean():.2f}")  # ~0.5 expected
print(f"Class balance (test):  {y_test.mean():.2f}")

# ----------------------------------------------------------------------
# STEP 2 — PCA for exploration
# ----------------------------------------------------------------------
# Goal: check if the data has obvious 2D structure.
# Also captures how much variance the top components hold — if top-2 only
# hold ~40% (independent features), PCA can't compress this data meaningfully.

from sklearn.decomposition import PCA

pca_full = PCA(n_components=5)        # fit all components to see the variance profile
pca_full.fit(X_train_scaled)

print("\n--- PCA variance profile ---")
for i, ratio in enumerate(pca_full.explained_variance_ratio_):
    print(f"PC{i+1}: {ratio:.3f} ({ratio*100:.1f}%)")
print(f"PC1+PC2 total: {pca_full.explained_variance_ratio_[:2].sum()*100:.1f}%")

# Project train and test to 2D for the "with PCA" pipeline branch
pca_2d = PCA(n_components=2, random_state=42)
X_train_pca = pca_2d.fit_transform(X_train_scaled)
X_test_pca = pca_2d.transform(X_test_scaled)    # again: train params on test

print(f"Projected shape: {X_train_pca.shape}")

# ----------------------------------------------------------------------
# STEP 3 — Isolation Forest for outlier flagging
# ----------------------------------------------------------------------
# No labels needed. Flags rows that are "easy to isolate" with random splits.
# Fit on train only. Apply to train → decide which rows to drop for training.

from sklearn.ensemble import IsolationForest

iso = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
iso.fit(X_train_scaled)

train_flags = iso.predict(X_train_scaled)    # +1 = normal, -1 = anomaly
n_flagged = (train_flags == -1).sum()
print(f"\n--- Isolation Forest ---")
print(f"Flagged {n_flagged} rows as anomalies ({n_flagged / len(y_train) * 100:.1f}%)")

# Cleaned training set: drop flagged rows
keep_mask = train_flags == 1
X_train_clean = X_train_scaled[keep_mask]
y_train_clean = y_train[keep_mask]
print(f"Cleaned train shape: {X_train_clean.shape}")

# ----------------------------------------------------------------------
# STEP 4 — Train + evaluate all models
# ----------------------------------------------------------------------
# Full comparison:
# - Logistic Regression (W5) — optimization-based baseline
# - Random Forest (W5) — bagging
# - Gradient Boosting on raw features (W6) — the hero
# - Gradient Boosting on PCA features — does compression help?
# - Gradient Boosting on cleaned data — does outlier removal help?

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def evaluate(name, model, X_tr, y_tr, X_te, y_te):
    model.fit(X_tr, y_tr)
    preds = model.predict(X_te)
    return {
        "model": name,
        "accuracy":  accuracy_score(y_te, preds),
        "precision": precision_score(y_te, preds),
        "recall":    recall_score(y_te, preds),
        "f1":        f1_score(y_te, preds),
        "preds": preds,
    }

results = []

# Baseline: majority class (tells us what "no model" looks like)
majority_acc = max(y_test.mean(), 1 - y_test.mean())
print(f"\nMajority-class baseline: {majority_acc:.3f}")

# Model 1 — Logistic Regression on raw features
results.append(evaluate(
    "Logistic Regression",
    LogisticRegression(max_iter=1000, random_state=42),
    X_train_scaled, y_train, X_test_scaled, y_test,
))

# Model 2 — Random Forest on raw features
results.append(evaluate(
    "Random Forest",
    RandomForestClassifier(n_estimators=100, random_state=42),
    X_train_scaled, y_train, X_test_scaled, y_test,
))

# Model 3 — Gradient Boosting on raw features
results.append(evaluate(
    "Gradient Boosting (raw)",
    GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42),
    X_train_scaled, y_train, X_test_scaled, y_test,
))

# Model 4 — Gradient Boosting on PCA-reduced features
results.append(evaluate(
    "Gradient Boosting (PCA-2)",
    GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42),
    X_train_pca, y_train, X_test_pca, y_test,
))

# Model 5 — Gradient Boosting on outlier-cleaned data
results.append(evaluate(
    "Gradient Boosting (cleaned)",
    GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42),
    X_train_clean, y_train_clean, X_test_scaled, y_test,
))

# ----------------------------------------------------------------------
# STEP 5 — Print comparison table
# ----------------------------------------------------------------------
print("\n" + "=" * 70)
# ----------------------------------------------------------------------
# STEP 6 — Export the winning model for Week 7 deployment
# ----------------------------------------------------------------------
# We picked GB (cleaned) — highest accuracy (0.910), represents the full pipeline.
# We need to save BOTH the model AND the scaler — inference requires the
# same scaling that training used.

import joblib

# Train one clean copy of GB on the cleaned+scaled training data
# (The one inside evaluate() is scoped to that function — we need a fresh handle here)
gb_final = GradientBoostingClassifier(
    n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42
)
gb_final.fit(X_train_clean, y_train_clean)

joblib.dump(gb_final, "model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("\n--- Exported for deployment ---")
print("Saved: model.pkl, scaler.pkl")

# Sanity check — load them back and verify one prediction matches
loaded_model = joblib.load("model.pkl")
loaded_scaler = joblib.load("scaler.pkl")
sanity_pred = loaded_model.predict(X_test_scaled[:1])
original_pred = gb_final.predict(X_test_scaled[:1])
assert sanity_pred == original_pred, "Sanity check FAILED"
print(f"Sanity check passed: both predicted {sanity_pred[0]}")

print(f"{'Model':<32}{'Acc':>8}{'Prec':>8}{'Rec':>8}{'F1':>8}")
print("-" * 70)
for r in results:
    print(f"{r['model']:<32}{r['accuracy']:>8.3f}{r['precision']:>8.3f}"
          f"{r['recall']:>8.3f}{r['f1']:>8.3f}")
print("=" * 70)


# Pull predictions out by position in the results list
lr_preds         = results[0]["preds"]
rf_preds         = results[1]["preds"]
gb_raw_preds     = results[2]["preds"]
gb_cleaned_preds = results[4]["preds"]

print("\n--- Convergence diagnostic ---")
print(f"LR vs GB-cleaned disagreements: {(lr_preds != gb_cleaned_preds).sum()} / {len(y_test)}")
print(f"RF vs GB-raw disagreements:     {(rf_preds != gb_raw_preds).sum()} / {len(y_test)}")