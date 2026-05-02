"""
numpy_nn.py — 2-Layer Neural Network from Scratch (NumPy only)

Solves the XOR problem using:
- Forward pass: matrix multiplication + ReLU + sigmoid
- Backward pass: manually-derived gradients via chain rule
- Update: vanilla gradient descent

Architecture: Input(2) → Hidden(8, ReLU) → Output(1, Sigmoid)

Why 8 hidden neurons instead of 4?
With random * 0.1 initialization, 4 neurons often ALL land in the dead
ReLU zone (Z always negative → gradient always 0 → permanently stuck).
8 neurons = more chances for survivors. PyTorch's Kaiming initialization
solves this at 4 neurons — see pytorch_nn.py.
"""

import numpy as np
import matplotlib.pyplot as plt


# ── Data ──────────────────────────────────────────────────────────────
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])   # 4 XOR inputs
Y = np.array([[0], [1], [1], [0]])                 # XOR targets


# ── Forward Pass ──────────────────────────────────────────────────────
def forward(X, W1, b1, W2, b2):
    Z1 = X @ W1 + b1              # (4,2) @ (2,8) = (4,8)  — pre-activation, hidden
    A1 = np.maximum(0, Z1)        # (4,8)                   — ReLU activation
    Z2 = A1 @ W2 + b2             # (4,8) @ (8,1) = (4,1)  — pre-activation, output
    A2 = 1 / (1 + np.exp(-Z2))    # (4,1)                   — sigmoid activation
    return Z1, A1, Z2, A2


# ── Loss: Binary Cross-Entropy ────────────────────────────────────────
def compute_loss(Y, A2):
    m = Y.shape[0]
    A2 = np.clip(A2, 1e-8, 1 - 1e-8)   # prevent log(0)
    loss = -np.mean(Y * np.log(A2) + (1 - Y) * np.log(1 - A2))
    return loss


# ── Backward Pass (chain rule, manually derived) ──────────────────────
def backward(X, Y, Z1, A1, Z2, A2, W2):
    m = Y.shape[0]

    # Output layer gradients
    dZ2 = A2 - Y                                        # (4,1)  — prediction minus target
    dW2 = (1 / m) * A1.T @ dZ2                          # (8,1)  — how much to blame W2
    db2 = (1 / m) * np.sum(dZ2, axis=0, keepdims=True)  # (1,1)  — how much to blame b2

    # Hidden layer gradients (blame flows backward through W2, gated by ReLU)
    dA1 = dZ2 @ W2.T                                    # (4,8)  — error passed back
    dZ1 = dA1 * (Z1 > 0)                                # (4,8)  — ReLU gate: 0 where Z1 was negative
    dW1 = (1 / m) * X.T @ dZ1                            # (2,8)  — how much to blame W1
    db1 = (1 / m) * np.sum(dZ1, axis=0, keepdims=True)  # (1,8)  — how much to blame b1

    return dW1, db1, dW2, db2


# ── Initialize Weights ────────────────────────────────────────────────
np.random.seed(42)
W1 = np.random.randn(2, 8) * 0.1    # small random values
b1 = np.zeros((1, 8))
W2 = np.random.randn(8, 1) * 0.1
b2 = np.zeros((1, 1))

# ── Hyperparameters ───────────────────────────────────────────────────
lr = 1.0
epochs = 5000

# ── Training Loop ─────────────────────────────────────────────────────
losses = []

for epoch in range(epochs):
    # 1. Forward
    Z1, A1, Z2, A2 = forward(X, W1, b1, W2, b2)

    # 2. Loss
    loss = compute_loss(Y, A2)
    losses.append(loss)

    # 3. Backward
    dW1, db1, dW2, db2 = backward(X, Y, Z1, A1, Z2, A2, W2)

    # 4. Update (gradient descent)
    W1 -= lr * dW1
    b1 -= lr * db1
    W2 -= lr * dW2
    b2 -= lr * db2

    if epoch % 500 == 0:
        print(f"Epoch {epoch:4d} | Loss: {loss:.4f}")

# ── Results ───────────────────────────────────────────────────────────
print(f"\nFinal loss: {losses[-1]:.6f}")
print(f"Predictions: {np.round(A2.flatten(), 2)}")
print(f"Targets:     {Y.flatten()}")
accuracy = np.mean((A2 > 0.5).astype(int) == Y) * 100
print(f"Accuracy:    {accuracy:.0f}%")