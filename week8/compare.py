"""
compare.py — Side-by-Side: NumPy NN vs PyTorch NN on XOR

Runs both implementations, collects their loss curves,
and plots them on one chart.

Usage: python compare.py
Outputs: loss_comparison.png + printed accuracies
"""

import numpy as np
import matplotlib.pyplot as plt


# =====================================================================
# 1. NumPy Neural Network (8 hidden, random * 0.1 init)
# =====================================================================
def run_numpy_nn():
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    Y = np.array([[0], [1], [1], [0]])

    np.random.seed(42)
    W1 = np.random.randn(2, 8) * 0.1
    b1 = np.zeros((1, 8))
    W2 = np.random.randn(8, 1) * 0.1
    b2 = np.zeros((1, 1))

    lr, epochs = 1.0, 5000
    losses = []

    for epoch in range(epochs):
        # Forward
        Z1 = X @ W1 + b1
        A1 = np.maximum(0, Z1)
        Z2 = A1 @ W2 + b2
        A2 = 1 / (1 + np.exp(-Z2))

        # Loss (BCE)
        A2c = np.clip(A2, 1e-8, 1 - 1e-8)
        loss = -np.mean(Y * np.log(A2c) + (1 - Y) * np.log(1 - A2c))
        losses.append(loss)

        # Backward
        m = Y.shape[0]
        dZ2 = A2 - Y
        dW2 = (1 / m) * A1.T @ dZ2
        db2 = (1 / m) * np.sum(dZ2, axis=0, keepdims=True)
        dA1 = dZ2 @ W2.T
        dZ1 = dA1 * (Z1 > 0)
        dW1 = (1 / m) * X.T @ dZ1
        db1 = (1 / m) * np.sum(dZ1, axis=0, keepdims=True)

        # Update
        W1 -= lr * dW1; b1 -= lr * db1
        W2 -= lr * dW2; b2 -= lr * db2

    accuracy = np.mean((A2 > 0.5).astype(int) == Y) * 100
    return losses, accuracy


# =====================================================================
# 2. PyTorch Neural Network (4 hidden, Kaiming init)
# =====================================================================
def run_pytorch_nn():
    import torch
    import torch.nn as nn

    torch.manual_seed(42)

    X = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
    y = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32)

    model = nn.Sequential(
        nn.Linear(2, 4),
        nn.ReLU(),
        nn.Linear(4, 1),
        nn.Sigmoid()
    )

    loss_fn = nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1.0)

    losses = []
    for epoch in range(5000):
        predictions = model(X)
        loss = loss_fn(predictions, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    with torch.no_grad():
        final_preds = model(X)
        accuracy = ((final_preds > 0.5).float() == y).float().mean().item() * 100

    return losses, accuracy


# =====================================================================
# 3. Run Both + Compare
# =====================================================================
if __name__ == "__main__":
    print("=" * 50)
    print("NumPy NN  (8 hidden, manual backprop, random*0.1)")
    print("=" * 50)
    np_losses, np_acc = run_numpy_nn()
    print(f"  Final loss: {np_losses[-1]:.6f}")
    print(f"  Accuracy:   {np_acc:.0f}%")

    print()
    print("=" * 50)
    print("PyTorch NN (4 hidden, autograd, Kaiming init)")
    print("=" * 50)
    pt_losses, pt_acc = run_pytorch_nn()
    print(f"  Final loss: {pt_losses[-1]:.6f}")
    print(f"  Accuracy:   {pt_acc:.0f}%")

    # ── Plot ──────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(np_losses, label=f"NumPy (8 hidden, manual backprop) — {np_acc:.0f}%",
            linewidth=1.8, color="#2563eb")
    ax.plot(pt_losses, label=f"PyTorch (4 hidden, autograd) — {pt_acc:.0f}%",
            linewidth=1.8, linestyle="--", color="#f97316")
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Loss (BCE)", fontsize=12)
    ax.set_title("XOR Training: NumPy vs PyTorch", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("loss_comparison.png", dpi=150)
    print(f"\nPlot saved → loss_comparison.png")