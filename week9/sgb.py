import numpy as np

# Loss function: L(x, y) = x² + 10y²
def loss(x, y):
    return x**2 + 10 * y**2

# Gradients (derivatives)
def gradient(x, y):
    return 2 * x, 20 * y  # dL/dx, dL/dy

# Starting point and settings
x_start, y_start = -2.5, 1.2
lr = 0.09
beta = 0.5
steps = 30

# ---- Plain SGD ----
x, y = x_start, y_start
print("=== Plain SGD ===")
for i in range(steps):
    gx, gy = gradient(x, y)
    x = x - lr * gx
    y = y - lr * gy
    print(f"Step {i+1:2d} | x={x:+.4f}, y={y:+.4f} | loss={loss(x, y):.4f}")

print()

# ---- SGD + Momentum ----
x, y = x_start, y_start
vx, vy = 0.0, 0.0  # velocity starts at zero — ball is sitting still
print("=== SGD + Momentum (β=0.9) ===")
for i in range(steps):
    gx, gy = gradient(x, y)
    vx = beta * vx + gx        # accumulate: 90% old direction + new gradient
    vy = beta * vy + gy
    x = x - lr * vx             # step using velocity, not raw gradient
    y = y - lr * vy
    print(f"Step {i+1:2d} | x={x:+.4f}, y={y:+.4f} | loss={loss(x, y):.4f}")