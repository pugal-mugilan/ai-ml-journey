"""
Multiple Feature Linear Regression (Matrix Form)
========================================================
Extend from single feature to multiple features using matrix multiplication.

Key concepts:
- y = X @ w + b (matrix form replaces individual dot products)
- Data matrix X: rows = samples, columns = features
- X.T @ error computes ALL derivatives at once
- Broadcasting bug: (100,) vs (100,1) shape mismatch
- Learned weights reveal feature importance automatically
"""

import numpy as np

# ============================================================
# Step 1: Generate fake data with known weights
# ============================================================
# True relationship: price = 50*sqft + 3*bedroom + 10*age + 10000 + noise
# Known answer: w = [50, 3, 10], b = 10000

sqft = np.random.uniform(2500, 5000, 100)
bedroom = np.random.uniform(1, 5, 100)
age = np.random.uniform(2, 20, 100)
noise = np.random.normal(0, 5000, 100)

actual_prices = 50 * sqft + 3 * bedroom + 10 * age + noise + 10000

# ============================================================
# Step 2: Normalize ALL features and target
# ============================================================
# Rule: if features aren't on the same scale, normalize them ALL.
# Even "small" numbers like bedroom (1-5) vs age (2-20) are 4x apart.
# The model cares about relative scale, not absolute size.

scaled_sqft = (sqft - sqft.min()) / (sqft.max() - sqft.min())
scaled_bedroom = (bedroom - bedroom.min()) / (bedroom.max() - bedroom.min())
scaled_age = (age - age.min()) / (age.max() - age.min())

scaled_actual_prices = (actual_prices - actual_prices.min()) / (actual_prices.max() - actual_prices.min())
# CRITICAL: reshape to (100,1) to avoid broadcasting bug
# (100,) - (100,1) would give (100,100) — every combination instead of paired subtraction
scaled_actual_prices = scaled_actual_prices.reshape(-1, 1)

# ============================================================
# Step 3: Build data matrix and weight vector
# ============================================================
# X shape: (100, 3) — 100 houses, 3 features each
# w shape: (3, 1) — one weight per feature

X = np.column_stack([scaled_sqft, scaled_bedroom, scaled_age])  # (100, 3)
w = np.random.randn(3, 1)  # (3, 1)
b = np.random.randn()
learning_rate = 0.01
n = X.shape[0]  # 100 — don't hardcode

# ============================================================
# Step 4: Training loop (matrix version)
# ============================================================
# Key shape checks:
#   X @ w       = (100,3) @ (3,1) = (100,1) — 100 predictions
#   X.T @ error = (3,100) @ (100,1) = (3,1) — 3 derivatives

for i in range(10000):
    predicted_prices = X @ w + b               # (100,1)
    error = predicted_prices - scaled_actual_prices  # (100,1)

    cost = (error ** 2).mean()

    # Matrix derivative: all 3 derivatives computed in one line
    # X.T rows = feature columns, dotted with error = each feature's derivative
    dw = (2 / n) * (X.T @ error)   # (3,1)
    db = (2 * error).mean()

    w = w - learning_rate * dw
    b = b - learning_rate * db

    if i % 2000 == 0:
        print(f"Iteration {i:5d} | Cost: {cost:.6f}")

# ============================================================
# Step 5: Results — weights reveal feature importance
# ============================================================
feature_names = ['sqft', 'bedroom', 'age']
print("\nLearned weights (normalized space):")
for name, weight in zip(feature_names, w.flatten()):
    print(f"  {name:10s}: {weight:.4f}")
print(f"  {'bias':10s}: {b:.4f}")

print("\nInterpretation:")
print("Sqft weight dominates because 50*sqft contributes ₹125,000-250,000")
print("Bedroom (3*bedroom = ₹3-15) and age (10*age = ₹20-200) are noise by comparison")
print("The model figured this out on its own — nobody told it sqft matters most.")

# ============================================================
# Step 6: Verify predictions
# ============================================================
scaled_predictions = X @ w + b
price_min = actual_prices.min()
price_max = actual_prices.max()
real_predictions = scaled_predictions.flatten() * (price_max - price_min) + price_min

errors = real_predictions - actual_prices
print(f"\nVerification:")
print(f"Mean error: ₹{errors.mean():.2f}")
print(f"Avg absolute error: ₹{np.abs(errors).mean():.2f}")
print(f"Price range: ₹{price_min:.0f} to ₹{price_max:.0f}")
print(f"Error as % of max price: {np.abs(errors).mean() / price_max * 100:.1f}%")