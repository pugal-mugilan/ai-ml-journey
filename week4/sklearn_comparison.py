"""
sklearn Comparison — From Scratch vs Production
=======================================================
Compare hand-built gradient descent models with sklearn's implementations.

Key discoveries:
- sklearn's LinearRegression uses the Normal Equation (one shot, no loop)
  w = inverse(X.T @ X) @ X.T @ y
- Predictions match even when multicollinear weights differ
- sklearn's Lasso uses coordinate descent — CAN land weights at exactly zero
- Our gradient descent Lasso oscillates — weights never reach exactly zero
- Verification: reverse normalization, check error as % of price range
"""

import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso

# ============================================================
# Step 1: Generate data with multicollinearity
# ============================================================
sqft = np.random.uniform(2500, 5000, 100)
bedroom = np.random.uniform(1, 5, 100)
rooms = 2 * bedroom + np.random.normal(0, 0.3, 100)  # correlated with bedroom
age = np.random.uniform(2, 20, 100)
noise = np.random.normal(0, 5000, 100)

actual_prices = 50 * sqft + 3 * bedroom + 10 * age + 5 * rooms + noise + 10000

# ============================================================
# Step 2: Normalize
# ============================================================
scaled_sqft = (sqft - sqft.min()) / (sqft.max() - sqft.min())
scaled_bedroom = (bedroom - bedroom.min()) / (bedroom.max() - bedroom.min())
scaled_rooms = (rooms - rooms.min()) / (rooms.max() - rooms.min())
scaled_age = (age - age.min()) / (age.max() - age.min())

scaled_actual_prices = (actual_prices - actual_prices.min()) / (actual_prices.max() - actual_prices.min())
scaled_actual_prices = scaled_actual_prices.reshape(-1, 1)

X = np.column_stack([scaled_sqft, scaled_bedroom, scaled_rooms, scaled_age])
n = X.shape[0]

# ============================================================
# Step 3: From-scratch linear regression (gradient descent)
# ============================================================
print("=" * 60)
print("LINEAR REGRESSION: From Scratch vs sklearn")
print("=" * 60)

w = np.random.randn(4, 1)
b = np.random.randn()
learning_rate = 0.01

for i in range(10000):
    predicted = X @ w + b
    error = predicted - scaled_actual_prices
    cost = (error ** 2).mean()
    dw = (2 / n) * (X.T @ error)
    db = (2 * error).mean()
    w = w - learning_rate * dw
    b = b - learning_rate * db

# sklearn LinearRegression (uses Normal Equation — one shot, no loop)
model = LinearRegression()
model.fit(X, scaled_actual_prices)

feature_names = ['sqft', 'bedroom', 'rooms', 'age']
print(f"\n{'Feature':<10} {'From Scratch':>15} {'sklearn':>15}")
print("-" * 42)
for name, scratch_w, sklearn_w in zip(feature_names, w.flatten(), model.coef_.flatten()):
    print(f"{name:<10} {scratch_w:>15.4f} {sklearn_w:>15.4f}")
print(f"{'bias':<10} {b:>15.4f} {model.intercept_[0]:>15.4f}")

# Check prediction difference
scratch_pred = (X @ w + b).flatten()
sklearn_pred = model.predict(X).flatten()
print(f"\nMax prediction difference: {np.abs(scratch_pred - sklearn_pred).max():.6f}")
print("(Should be tiny — both methods find similar predictions)")
print("(Weights differ for correlated features because of multicollinearity)")

# ============================================================
# Step 4: Ridge comparison
# ============================================================
print("\n" + "=" * 60)
print("RIDGE REGRESSION: From Scratch vs sklearn")
print("=" * 60)

lam = 0.1
w_ridge = np.random.randn(4, 1)
b_ridge = np.random.randn()

for i in range(10000):
    predicted = X @ w_ridge + b_ridge
    error = predicted - scaled_actual_prices
    cost = (error ** 2).mean() + lam * np.sum(w_ridge ** 2)
    dw = (2 / n) * (X.T @ error) + lam * 2 * w_ridge
    db = (2 * error).mean()
    w_ridge = w_ridge - learning_rate * dw
    b_ridge = b_ridge - learning_rate * db

ridge_model = Ridge(alpha=0.1)
ridge_model.fit(X, scaled_actual_prices)

print(f"\n{'Feature':<10} {'From Scratch':>15} {'sklearn':>15}")
print("-" * 42)
for name, scratch_w, sklearn_w in zip(feature_names, w_ridge.flatten(), ridge_model.coef_.flatten()):
    print(f"{name:<10} {scratch_w:>15.4f} {sklearn_w:>15.4f}")

# ============================================================
# Step 5: Lasso comparison — the big reveal
# ============================================================
print("\n" + "=" * 60)
print("LASSO REGRESSION: From Scratch vs sklearn")
print("=" * 60)

w_lasso = np.random.randn(4, 1)
b_lasso = np.random.randn()

for i in range(10000):
    predicted = X @ w_lasso + b_lasso
    error = predicted - scaled_actual_prices
    cost = (error ** 2).mean() + lam * np.sum(np.abs(w_lasso))
    dw = (2 / n) * (X.T @ error) + lam * np.sign(w_lasso)
    db = (2 * error).mean()
    w_lasso = w_lasso - learning_rate * dw
    b_lasso = b_lasso - learning_rate * db

# Try multiple alpha values to see feature selection in action
print("\nsklearn Lasso at different alpha values:")
for alpha in [0.1, 0.01, 0.001]:
    lasso_model = Lasso(alpha=alpha)
    lasso_model.fit(X, scaled_actual_prices)
    print(f"  alpha={alpha:<5} → weights: {lasso_model.coef_}")

print(f"\nFrom scratch Lasso weights: {w_lasso.flatten()}")
print("\nKey observation:")
print("- sklearn Lasso kills features to EXACTLY zero (coordinate descent)")
print("- From-scratch Lasso gets close but oscillates (gradient descent limitation)")
print("- alpha=0.1 too aggressive (kills everything)")
print("- alpha=0.001 just right (kills only redundant/useless features)")

# ============================================================
# Step 6: Verify — which model actually predicts well?
# ============================================================
print("\n" + "=" * 60)
print("MODEL VERIFICATION: Reverse normalization, check real errors")
print("=" * 60)

price_min = actual_prices.min()
price_max = actual_prices.max()

# sklearn LinearRegression
sklearn_scaled_pred = model.predict(X)
sklearn_real_pred = sklearn_scaled_pred.flatten() * (price_max - price_min) + price_min
sklearn_errors = sklearn_real_pred - actual_prices

# From scratch
scratch_scaled_pred = (X @ w + b).flatten()
scratch_real_pred = scratch_scaled_pred * (price_max - price_min) + price_min
scratch_errors = scratch_real_pred - actual_prices

print(f"\n{'Metric':<25} {'From Scratch':>15} {'sklearn':>15}")
print("-" * 57)
print(f"{'Mean error (₹)':<25} {scratch_errors.mean():>15.2f} {sklearn_errors.mean():>15.2f}")
print(f"{'Avg absolute error (₹)':<25} {np.abs(scratch_errors).mean():>15.2f} {np.abs(sklearn_errors).mean():>15.2f}")
print(f"{'Max error (₹)':<25} {np.abs(scratch_errors).max():>15.2f} {np.abs(sklearn_errors).max():>15.2f}")
print(f"{'Error as % of max price':<25} {np.abs(scratch_errors).mean()/price_max*100:>14.1f}% {np.abs(sklearn_errors).mean()/price_max*100:>14.1f}%")

print(f"\nPrice range: ₹{price_min:.0f} to ₹{price_max:.0f}")
print(f"Noise std: 5000 → expected irreducible error ≈ ₹4,000")
print(f"sklearn's error ≈ noise level = model found the true pattern")