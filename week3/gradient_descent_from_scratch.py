"""
Gradient Descent from Scratch
Week 3 - AI/ML Journey

What this does:
- There's a mystery function that turns inputs into outputs (y ≈ 3x)
- We don't know the parameter (slope = 3) — gradient descent figures it out
- It starts with a random guess, measures how wrong it is (cost),
  checks which direction to adjust (derivative), and nudges the parameter
- Repeat until cost stops dropping — the algorithm "learned" the pattern

This is the same core algorithm behind every neural network in the world,
just with one parameter instead of millions.
"""

import numpy as np

# ============================================================
# STEP 1: Generate fake data (our "mystery function")
# ============================================================
# We secretly set slope = 3, but gradient descent doesn't know this.
# It has to discover it from the data alone.

np.random.seed(42)  # For reproducibility
x = np.random.rand(100)          # 100 random inputs between 0 and 1
true_slope = 3
noise = np.random.randn(100) * 0.2  # Small random noise (like real-world messiness)
y = true_slope * x + noise          # The "real" outputs

print("=" * 50)
print("GRADIENT DESCENT FROM SCRATCH")
print("=" * 50)
print(f"True slope (hidden from algorithm): {true_slope}")
print(f"Sample data: x={x[0]:.3f} → y={y[0]:.3f}")
print()

# ============================================================
# STEP 2: Start with a random guess
# ============================================================
w = np.random.randn()  # Could be anything — 0.5, -1.2, etc.
print(f"Starting guess for slope: {w:.4f}")
print(f"Goal: get close to {true_slope}")
print()

# ============================================================
# STEP 3: Run gradient descent
# ============================================================
# The loop:
#   1. Predict: predicted_y = w * x
#   2. Cost: how wrong am I? (Mean Squared Error)
#   3. Derivative: which direction should w move?
#   4. Update: w = w - learning_rate * derivative

learning_rate = 0.1
num_steps = 100
cost_history = []  # Track cost at each step to see it drop

print(f"Learning rate: {learning_rate}")
print(f"Running {num_steps} steps...")
print("-" * 50)

for i in range(num_steps):
    # 1. Predict
    predicted_y = w * x

    # 2. Cost (Mean Squared Error)
    # Same logic as variance from stats: square errors to prevent
    # cancellation and punish big mistakes more
    cost = ((predicted_y - y) ** 2).mean()
    cost_history.append(cost)

    # 3. Derivative (chain rule: outer' × inner')
    # Cost = mean of (w*x - y)²
    # Derivative with respect to w = mean of 2*(w*x - y)*x
    derivative = (2 * (predicted_y - y) * x).mean()

    # 4. Update — subtract because we want to go DOWNHILL
    # Positive derivative = cost increasing = move w down
    # Negative derivative = cost decreasing = move w up
    w = w - learning_rate * derivative

    # Print progress every 10 steps
    if (i + 1) % 10 == 0:
        print(f"Step {i+1:3d} | w = {w:.4f} | cost = {cost:.6f} | derivative = {derivative:.6f}")

print("-" * 50)
print(f"Final w: {w:.4f}")
print(f"True slope: {true_slope}")
print(f"How close: {abs(true_slope - w):.4f} off")
print()

# ============================================================
# STEP 4: Compare learning rates
# ============================================================
# Same algorithm, different step sizes — see the effect

print("=" * 50)
print("LEARNING RATE COMPARISON")
print("=" * 50)

for lr in [0.0001, 0.1, 1.0]:
    w_test = 0.5  # Same starting point for fair comparison
    for _ in range(100):
        pred = w_test * x
        deriv = (2 * (pred - y) * x).mean()
        w_test = w_test - lr * deriv

    cost_final = ((w_test * x - y) ** 2).mean()
    print(f"lr = {lr:<6} → final w = {w_test:.4f} | final cost = {cost_final:.6f}")

print()
print("Too small (0.0001): barely moves in 100 steps")
print("Just right (0.1):   reaches close to 3 steadily")
print("Large (1.0):        converges fast (but can overshoot on harder problems)")