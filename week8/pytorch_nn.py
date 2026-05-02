"""
pytorch_nn.py — 2-Layer Neural Network in PyTorch

Same XOR problem as numpy_nn.py, same training approach.
PyTorch automates: gradient computation (autograd), weight
initialization (Kaiming), parameter updates (optimizer.step).

Architecture: Input(2) → Hidden(4, ReLU) → Output(1, Sigmoid)

Why 4 hidden neurons here but 8 in NumPy?
PyTorch's nn.Linear uses Kaiming initialization, which picks the
random scale based on layer size. Fewer neurons start in the dead
ReLU zone, so 4 is enough. The NumPy version used random * 0.1,
which needed 8 neurons to guarantee enough survivors.
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt


# ── Data ──────────────────────────────────────────────────────────────
X = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
y = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32)


# ── Model ─────────────────────────────────────────────────────────────
model = nn.Sequential(
    nn.Linear(2, 4),     # hidden layer: W1 (2,4) + b1 (4,)
    nn.ReLU(),           # activation
    nn.Linear(4, 1),     # output layer: W2 (4,1) + b2 (1,)
    nn.Sigmoid()         # output activation
)

# ── Loss + Optimizer ──────────────────────────────────────────────────
loss_fn = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1.0)

# ── Hyperparameters ───────────────────────────────────────────────────
epochs = 5000

# ── Training Loop ─────────────────────────────────────────────────────
losses = []

for epoch in range(epochs):
    predictions = model(X)                # forward pass  (replaces 4 lines)
    loss = loss_fn(predictions, y)        # compute loss   (replaces 2 lines)

    optimizer.zero_grad()                 # reset gradients (PyTorch accumulates)
    loss.backward()                       # backward pass  (replaces 7 lines)
    optimizer.step()                      # update weights (replaces 4 lines)

    losses.append(loss.item())

    if epoch % 500 == 0:
        print(f"Epoch {epoch:4d} | Loss: {loss.item():.4f}")

# ── Results ───────────────────────────────────────────────────────────
with torch.no_grad():
    final_preds = model(X)
    accuracy = ((final_preds > 0.5).float() == y).float().mean() * 100
    print(f"\nFinal loss: {losses[-1]:.6f}")
    print(f"Predictions: {final_preds.flatten().round().tolist()}")
    print(f"Targets:     {y.flatten().tolist()}")
    print(f"Accuracy:    {accuracy:.0f}%")