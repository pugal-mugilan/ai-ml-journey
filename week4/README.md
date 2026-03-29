# Linear Regression — From Math to Production

The **compression anchor** of the entire ML journey. Every future algorithm is a variation of what's in this folder:
- Logistic regression = this + sigmoid
- Neural networks = this + layers
- XGBoost = this + boosting

## What's Here

| File                         | What It Covers |
|------------------------------|---------------|
| `1_single_feature.py`        | Single feature regression from scratch: `y = w*x + b`, gradient descent with bias, min-max normalization |
| `2_matrix_form.py`           | Multiple features using matrix multiplication: `y = X@w + b`, `dw = (2/n) * X.T @ error` |
| `3_regularization.py`        | Ridge (L2) and Lasso (L1) from scratch — fixing multicollinearity with weight penalties |
| `sklearn_comparison.py` | From-scratch vs sklearn: LinearRegression, Ridge, Lasso side by side |

## The Journey


### Single feature from scratch
Built `y = w*x + b` with gradient descent. Discovered the exploding gradient problem — raw sqft values (2500-5000) made gradients blow up. Fix: min-max normalization scales everything to 0-1.

### Matrix form (the power move)
Extended to multiple features. Instead of computing one derivative per feature manually, `dw = (2/n) * (X.T @ error)` computes ALL derivatives at once. Whether you have 3 features or 1000, it's always two lines of code.

### The 4 assumptions (health checks)
Learned the four assumptions every linear regression depends on: linearity, independence of errors, homoscedasticity, normality of residuals. All checked via residual plots. Key insight: the "screaming voter" problem — big-error samples dominate the gradient before `mean()` collapses everything into one number.

### Multicollinearity + Regularization
Created correlated features (rooms ≈ 2 × bedrooms) and watched weights become unstable. Implemented both fixes from scratch:
- **Ridge (L2):** Adds `λ × 2w` to derivative — rubber band that weakens near zero, shrinks but never kills
- **Lasso (L1):** Adds `λ × sign(w)` to derivative — constant force that pushes all the way to zero (feature selection)

### sklearn comparison + verification
Compared from-scratch results with sklearn. Key findings:
- sklearn's `LinearRegression` uses the **Normal Equation** — one-shot computation, no iteration
- Predictions match even when multicollinear weights differ wildly
- sklearn's Lasso kills features to **exactly zero** (coordinate descent), our gradient descent version oscillates
- Verified models by reversing normalization — sklearn's error matched the noise level (irreducible error)

## Key Concepts

**Normal Equation:** `w = inverse(X.T @ X) @ X.T @ y` — requires a square matrix, which is why `X.T @ X` is used (makes any shape square). Breaks when det ≈ 0 (multicollinearity).

**Multicollinearity breaks weight stability, not predictions.** Two different runs give wildly different weights for correlated features, but nearly identical predictions. The model can't decide how to split credit.

**Ridge vs Lasso:**
- Ridge: penalty = `λ × sum(w²)`, force weakens near zero → shrinks all weights
- Lasso: penalty = `λ × sum(|w|)`, constant force → kills useless weights to zero
- Lambda too high → kills everything. Lambda too low → no effect. Finding the right lambda matters.

**Model verification checklist:**
1. Mean error near ₹0? → No systematic bias
2. Avg error < 5% of price range? → Model is working
3. Avg error ≈ noise level? → Model found the true pattern (only randomness remains)

## How to Run

```bash
python 1_single_feature.py
python 2_matrix_form.py
python 3_regularization.py
python sklearn_comparison.py
```

Requires: `numpy`, `scikit-learn`