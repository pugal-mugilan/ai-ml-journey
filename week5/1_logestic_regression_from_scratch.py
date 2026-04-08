"""
 Logistic Regression from Scratch
Dataset: Tumor size → Malignant (1) or Benign (0)
Key discovery: Only 3 things change from linear regression — sigmoid, cross-entropy, 1/n
"""

import numpy as np

# ============================================================
# DATA — Two clusters, no hidden formula
# ============================================================
np.random.seed(42)
benign_sizes = np.random.normal(2, 0.8, 50)      # cluster centered at 2 cm
malignant_sizes = np.random.normal(5, 0.8, 50)    # cluster centered at 5 cm

X = np.concatenate([benign_sizes, malignant_sizes]).reshape(-1, 1)  # (100, 1)
y = np.concatenate([np.zeros(50), np.ones(50)]).reshape(-1, 1)      # (100, 1)

# ============================================================
# SIGMOID — The squisher (0 to 1)
# ============================================================
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# ============================================================
# TRAINING — Same loop as Week 4, three things changed
# ============================================================
w = np.random.randn(1, 1)
b = np.random.randn(1, 1)
learning_rate = 0.1
n = X.shape[0]
iterations = 50000

for i in range(iterations):
    # Step 1: Linear computation
    z = X @ w + b

    # Step 2: Sigmoid (NEW — squish to 0-1)
    predictions = sigmoid(z)

    # Step 3: Error (order matters: pred - y, NOT y - pred)
    error = predictions - y

    # Step 4: Cost — Cross-entropy (NEW — replaces MSE)
    cost = (-y * np.log(predictions) - (1 - y) * np.log(1 - predictions)).mean()

    # Step 5: Derivatives (1/n instead of 2/n — no squaring in cross-entropy)
    dw = (1/n) * (X.T @ error)
    db = error.mean()

    # Step 6: Update
    w = w - learning_rate * dw
    b = b - learning_rate * db

    if i % 10000 == 0:
        print(f"Iteration {i:5d} | Cost: {cost:.4f} | w: {w[0,0]:.4f} | b: {b[0,0]:.4f}")

# ============================================================
# DECISION BOUNDARY — Where sigmoid = 0.5 (z = 0)
# ============================================================
boundary = -b[0, 0] / w[0, 0]
print(f"\nDecision boundary: {boundary:.2f} cm")
print(f"(Cluster centers: benign=2, malignant=5, midpoint=3.5)")

# ============================================================
# VERIFICATION — Test predictions on new data
# ============================================================
test_tumors = np.array([1.0, 2.5, 3.5, 4.5, 6.0]).reshape(-1, 1)
test_probs = sigmoid(test_tumors @ w + b)

print(f"\n{'Tumor Size':>12} | {'Probability':>11} | Prediction")
print("-" * 45)
for size, prob in zip(test_tumors.flatten(), test_probs.flatten()):
    label = "Malignant" if prob >= 0.5 else "Benign"
    print(f"{size:>10.1f} cm | {prob:>11.4f} | {label}")