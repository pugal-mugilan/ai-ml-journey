"""
Regularization — Ridge (L2) and Lasso (L1) from Scratch
================================================================
Fix multicollinearity by penalizing large weights.

Key concepts:
- Multicollinearity: correlated features → det ≈ 0 → unstable weights
- Ridge (L2): adds λ × sum(w²) to cost → shrinks weights, never kills them
  - Derivative adds λ × 2w (force weakens near zero — rubber band)
- Lasso (L1): adds λ × sum(|w|) to cost → pushes weights to exactly zero
  - Derivative adds λ × sign(w) (constant force — same push at all sizes)
- Gradient descent Lasso oscillates (can't land at zero) — sklearn uses coordinate descent
"""

import numpy as np

# ============================================================
# Step 1: Generate data WITH multicollinearity
# ============================================================
# rooms ≈ 2 × bedrooms — these two features carry the same information
# This makes X.T @ X nearly singular (det ≈ 0)

sqft = np.random.uniform(2500, 5000, 100)
bedroom = np.random.uniform(1, 5, 100)
rooms = 2 * bedroom + np.random.normal(0, 0.3, 100)  # correlated with bedroom!
age = np.random.uniform(2, 20, 100)
noise = np.random.normal(0, 5000, 100)

actual_prices = 50 * sqft + 3 * bedroom + 10 * age + 5 * rooms + noise + 10000

# ============================================================
# Step 2: Normalize everything
# ============================================================
scaled_sqft = (sqft - sqft.min()) / (sqft.max() - sqft.min())
scaled_bedroom = (bedroom - bedroom.min()) / (bedroom.max() - bedroom.min())
scaled_rooms = (rooms - rooms.min()) / (rooms.max() - rooms.min())
scaled_age = (age - age.min()) / (age.max() - age.min())

scaled_actual_prices = (actual_prices - actual_prices.min()) / (actual_prices.max() - actual_prices.min())
scaled_actual_prices = scaled_actual_prices.reshape(-1, 1)

X = np.column_stack([scaled_sqft, scaled_bedroom, scaled_rooms, scaled_age])
n = X.shape[0]
learning_rate = 0.01
lam = 0.1  # λ — can't use 'lambda' in Python (reserved keyword)

# ============================================================
# Step 3: Demonstrate multicollinearity — weights are unstable
# ============================================================
print("=" * 60)
print("MULTICOLLINEARITY DEMO: Run plain regression twice")
print("Same data, different random starting weights → different final weights")
print("=" * 60)

for run in range(2):
    w = np.random.randn(4, 1)
    b = np.random.randn()

    for i in range(10000):
        predicted = X @ w + b
        error = predicted - scaled_actual_prices
        dw = (2 / n) * (X.T @ error)
        db = (2 * error).mean()
        w = w - learning_rate * dw
        b = b - learning_rate * db

    print(f"\nRun {run + 1} weights: {w.flatten()}")
    print(f"  sqft={w[0,0]:.3f}  bedroom={w[1,0]:.3f}  rooms={w[2,0]:.3f}  age={w[3,0]:.3f}")

print("\nNotice: sqft and age are stable, but bedroom and rooms swing wildly.")
print("Predictions are similar both times — only weight STABILITY is broken.\n")

# ============================================================
# Step 4: Ridge Regression (L2) — shrink weights
# ============================================================
print("=" * 60)
print("RIDGE REGRESSION (L2)")
print("cost = MSE + λ × sum(w²)")
print("derivative adds: λ × 2w (weakens near zero)")
print("=" * 60)

w = np.random.randn(4, 1)
b = np.random.randn()

for i in range(10000):
    predicted = X @ w + b
    error = predicted - scaled_actual_prices

    cost = (error ** 2).mean() + lam * np.sum(w ** 2)

    # Ridge derivative: original + penalty
    # λ × 2w acts like a rubber band — bigger weight = stronger pull toward zero
    dw = (2 / n) * (X.T @ error) + lam * 2 * w
    db = (2 * error).mean()

    w = w - learning_rate * dw
    b = b - learning_rate * db

    if i % 2000 == 0:
        print(f"Iteration {i:5d} | Cost: {cost:.6f} | Weights: {w.flatten()}")

print(f"\nRidge final weights: {w.flatten()}")
print("All weights are small but NONE are exactly zero — Ridge shrinks, doesn't kill.\n")

# ============================================================
# Step 5: Lasso Regression (L1) — kill useless weights
# ============================================================
print("=" * 60)
print("LASSO REGRESSION (L1)")
print("cost = MSE + λ × sum(|w|)")
print("derivative adds: λ × sign(w) (constant force at all sizes)")
print("=" * 60)

w = np.random.randn(4, 1)
b = np.random.randn()

for i in range(10000):
    predicted = X @ w + b
    error = predicted - scaled_actual_prices

    cost = (error ** 2).mean() + lam * np.sum(np.abs(w))

    # Lasso derivative: sign(w) gives +1 or -1 regardless of weight size
    # Same force whether w=1000 or w=0.001 — pushes all the way to zero
    dw = (2 / n) * (X.T @ error) + lam * np.sign(w)
    db = (2 * error).mean()

    w = w - learning_rate * dw
    b = b - learning_rate * db

    if i % 2000 == 0:
        print(f"Iteration {i:5d} | Cost: {cost:.6f} | Weights: {w.flatten()}")

print(f"\nLasso final weights: {w.flatten()}")
print("Weights are very small but NOT exactly zero — gradient descent oscillates.")
print("sklearn's Lasso uses coordinate descent which CAN land at exactly zero.")

# ============================================================
# Step 6: Compare Ridge vs Lasso
# ============================================================
print("\n" + "=" * 60)
print("RIDGE vs LASSO SUMMARY")
print("=" * 60)
print("""
Ridge (L2):
  Penalty: λ × sum(w²)
  Force:   λ × 2w — weakens near zero (rubber band)
  Result:  Shrinks ALL weights, keeps all features
  Use:     All features somewhat useful

Lasso (L1):
  Penalty: λ × sum(|w|)
  Force:   λ × sign(w) — constant at all sizes
  Result:  Kills useless weights to exactly zero (feature selection)
  Use:     Some features are useless, want automatic feature selection
""")