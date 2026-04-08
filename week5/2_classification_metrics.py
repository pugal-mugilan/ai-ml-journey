"""
Classification Metrics
Key concepts: Confusion matrix, accuracy paradox, precision, recall, F1, threshold tuning
Applied to: logistic regression tumor model
"""

import numpy as np

# ============================================================
# REUSE trained model (rebuild quickly)
# ============================================================
np.random.seed(42)
benign_sizes = np.random.normal(2, 0.8, 50)
malignant_sizes = np.random.normal(5, 0.8, 50)
X = np.concatenate([benign_sizes, malignant_sizes]).reshape(-1, 1)
y = np.concatenate([np.zeros(50), np.ones(50)]).reshape(-1, 1)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Train
w = np.random.randn(1, 1)
b = np.random.randn(1, 1)
lr = 0.1
n = X.shape[0]

for i in range(50000):
    z = X @ w + b
    preds = sigmoid(z)
    error = preds - y
    dw = (1/n) * (X.T @ error)
    db = error.mean()
    w = w - lr * dw
    b = b - lr * db

# ============================================================
# CONFUSION MATRIX — The 2x2 grid
# ============================================================
probabilities = sigmoid(X @ w + b).flatten()
threshold = 0.5
predicted = (probabilities >= threshold).astype(int)
actual = y.flatten().astype(int)

TP = np.sum((predicted == 1) & (actual == 1))
FP = np.sum((predicted == 1) & (actual == 0))
FN = np.sum((predicted == 0) & (actual == 1))
TN = np.sum((predicted == 0) & (actual == 0))

print("=== Confusion Matrix ===")
print(f"                  Actual Pos  Actual Neg")
print(f"  Predicted Pos      TP={TP:3d}      FP={FP:3d}")
print(f"  Predicted Neg      FN={FN:3d}      TN={TN:3d}")

# ============================================================
# METRICS — Each tells you something different
# ============================================================
accuracy = (TP + TN) / (TP + TN + FP + FN)
precision = TP / (TP + FP) if (TP + FP) > 0 else 0
recall = TP / (TP + FN) if (TP + FN) > 0 else 0
f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

print(f"\n=== Metrics at threshold = {threshold} ===")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}  (Of my 'malignant' predictions, how many were right?)")
print(f"Recall:    {recall:.4f}  (Of all actual malignant cases, how many did I catch?)")
print(f"F1 Score:  {f1:.4f}  (Harmonic mean — only good when BOTH are good)")

# ============================================================
# ACCURACY PARADOX — When accuracy lies
# ============================================================
print("\n=== Accuracy Paradox Demo ===")
print("Scenario: 950 healthy, 50 cancer. Model predicts 'healthy' for everyone.")
tp, fp, fn, tn = 0, 0, 50, 950
acc = (tp + tn) / (tp + tn + fp + fn)
rec = tp / (tp + fn) if (tp + fn) > 0 else 0
print(f"Accuracy: {acc:.1%}  (looks great!)")
print(f"Recall:   {rec:.1%}  (missed EVERY cancer patient)")

# ============================================================
# THRESHOLD SWEEP — See precision/recall trade off
# ============================================================
print("\n=== Threshold Sweep ===")
print(f"{'Threshold':>10} | {'Precision':>10} | {'Recall':>10} | {'F1':>10}")
print("-" * 50)

for t in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    pred_t = (probabilities >= t).astype(int)
    tp_t = np.sum((pred_t == 1) & (actual == 1))
    fp_t = np.sum((pred_t == 1) & (actual == 0))
    fn_t = np.sum((pred_t == 0) & (actual == 1))

    prec_t = tp_t / (tp_t + fp_t) if (tp_t + fp_t) > 0 else 0
    rec_t = tp_t / (tp_t + fn_t) if (tp_t + fn_t) > 0 else 0
    f1_t = 2 * (prec_t * rec_t) / (prec_t + rec_t) if (prec_t + rec_t) > 0 else 0

    print(f"{t:>10.1f} | {prec_t:>10.4f} | {rec_t:>10.4f} | {f1_t:>10.4f}")

print("\nLower threshold → recall↑ precision↓ (wider net)")
print("Higher threshold → precision↑ recall↓ (stricter filter)")

# ============================================================
# WORKED EXAMPLE — Spam Filter
# ============================================================
print("\n=== Spam Filter Example ===")
tp, fp, fn, tn = 35, 10, 5, 150
total = tp + fp + fn + tn
acc = (tp + tn) / total
prec = tp / (tp + fp)
rec = tp / (tp + fn)
f1 = 2 * (prec * rec) / (prec + rec)

print(f"200 emails: 160 real, 40 spam")
print(f"TP={tp} (spam caught), FP={fp} (real flagged), FN={fn} (spam missed), TN={tn} (real safe)")
print(f"Accuracy:  {acc:.1%}  (looks great)")
print(f"Precision: {prec:.1%}  (1 in 5 'spam' flags was real email — problem!)")
print(f"Recall:    {rec:.1%}  (caught most spam)")
print(f"F1:        {f1:.1%}")
print(f"→ For spam filter: raise threshold to improve precision")