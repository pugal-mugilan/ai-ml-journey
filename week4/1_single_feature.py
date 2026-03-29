"""
Single Feature Linear Regression from Scratch
=====================================================
Predict house prices from square footage using gradient descent.

Key concepts:
- y = w * x + b (the model)
- Cost function (MSE) = mean of (prediction - actual)²
- Partial derivatives: dw and db
- Min-max normalization to prevent exploding gradients
- The bias term (b) lets the line shift up/down instead of forcing through origin
"""

import numpy as np

# ============================================================
# Step 1: Generate fake data (simulating reality)
# ============================================================
# True relationship: price = 50 * sqft + 10000 + noise
# We KNOW the answer: w=50, b=10000
# This is our "unit test" — verify the algorithm finds these values

sqft = np.random.uniform(2500, 5000, 100)
noise = np.random.normal(0, 5000, 100)
actual_prices = 50 * sqft + noise + 10000  # true formula

# ============================================================
# Step 2: Normalize (preparing for training)
# ============================================================
# Without normalization, raw sqft (2500-5000) makes gradients explode.
# Learning rate 0.1 would blow cost to infinity.
# Min-max scales everything to 0-1 range.

scaled_sqft = (sqft - sqft.min()) / (sqft.max() - sqft.min())
scaled_prices = (actual_prices - actual_prices.min()) / (actual_prices.max() - actual_prices.min())

# ============================================================
# Step 3: Initialize parameters (random guess)
# ============================================================
w = np.random.randn()  # random weight
b = np.random.randn()  # random bias
learning_rate = 0.01
n = len(scaled_sqft)

# ============================================================
# Step 4: Training loop (gradient descent)
# ============================================================
# 1. Predict → 2. Measure error → 3. Find direction → 4. Update → Repeat

for i in range(10000):
    # Predict
    predictions = w * scaled_sqft + b

    # Error
    error = predictions - scaled_prices

    # Cost (MSE)
    cost = (error ** 2).mean()

    # Derivatives (chain rule from Week 3 Day 5)
    # dw: outer' × inner' where inner derivative w.r.t. w = x
    # db: outer' × inner' where inner derivative w.r.t. b = 1
    dw = (2 * error * scaled_sqft).mean()
    db = (2 * error).mean()

    # Update (subtract because we want to go DOWNHILL)
    w = w - learning_rate * dw
    b = b - learning_rate * db

    if i % 2000 == 0:
        print(f"Iteration {i:5d} | Cost: {cost:.6f} | w: {w:.4f} | b: {b:.4f}")

print(f"\nFinal: w = {w:.4f}, b = {b:.4f}")
print(f"These are in normalized space (0-1 range), not real-world units.")

# ============================================================
# Step 5: Verify — reverse normalization and check predictions
# ============================================================
scaled_predictions = w * scaled_sqft + b
real_predictions = scaled_predictions * (actual_prices.max() - actual_prices.min()) + actual_prices.min()

errors = real_predictions - actual_prices
print(f"\nVerification:")
print(f"Mean error: ₹{errors.mean():.2f} (should be near 0)")
print(f"Avg absolute error: ₹{np.abs(errors).mean():.2f}")
print(f"Price range: ₹{actual_prices.min():.0f} to ₹{actual_prices.max():.0f}")