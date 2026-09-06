import numpy as np
import matplotlib.pyplot as plt

# === Loss and gradient (same as D1) ===
def loss(x, y):
    return x**2 + 10 * y**2

def grad(x, y):
    return np.array([2*x, 20*y])

# === Three optimizers ===
def sgd(lr, steps):
    pos = np.array([-2.5, 1.2])
    path = [pos.copy()]
    for _ in range(steps):
        g = grad(*pos)
        pos -= lr * g
        path.append(pos.copy())
    return np.array(path)

def sgd_momentum(lr, beta, steps):
    pos = np.array([-2.5, 1.2])
    v = np.array([0.0, 0.0])
    path = [pos.copy()]
    for _ in range(steps):
        g = grad(*pos)
        v = beta * v + g
        pos -= lr * v
        path.append(pos.copy())
    return np.array(path)

def adam(lr, steps, beta1=0.9, beta2=0.999, eps=1e-8):
    pos = np.array([-2.5, 1.2])
    m = np.array([0.0, 0.0])    # first moment (momentum)
    v = np.array([0.0, 0.0])    # second moment (squared gradients)
    path = [pos.copy()]
    for t in range(1, steps + 1):
        g = grad(*pos)
        m = beta1 * m + (1 - beta1) * g          # weighted avg of gradients
        v = beta2 * v + (1 - beta2) * g**2        # weighted avg of squared gradients
        m_hat = m / (1 - beta1**t)                # bias correction
        v_hat = v / (1 - beta2**t)                # bias correction
        pos -= lr * m_hat / (np.sqrt(v_hat) + eps)  # the update
        path.append(pos.copy())
    return np.array(path)

# === Run all three ===
lr = 0.09
steps = 30
path_sgd = sgd(lr, steps)
path_mom = sgd_momentum(lr, 0.5, steps)
path_adam = adam(lr, steps)
path_adam_tuned = adam(0.01, 1000)

# === Contour plot ===
fig, ax = plt.subplots(1, 1, figsize=(10, 6))
x_grid = np.linspace(-3.2, 3.2, 200)
y_grid = np.linspace(-1.8, 1.8, 200)
X, Y = np.meshgrid(x_grid, y_grid)
Z = loss(X, Y)
#
# ax.contour(X, Y, Z, levels=[0.1, 0.5, 1, 2, 4, 8, 12, 18, 25, 35], cmap='Blues', alpha=0.6)
# ax.plot(*path_sgd.T, 'o-', color='red', markersize=3, label=f'SGD (loss={loss(*path_sgd[-1]):.4f})')
# ax.plot(*path_mom.T, 'o-', color='blue', markersize=3, label=f'Momentum β=0.5 (loss={loss(*path_mom[-1]):.4f})')
# ax.plot(*path_adam.T, 'o-', color='green', markersize=3, label=f'Adam (loss={loss(*path_adam[-1]):.4f})')
#
# ax.plot(0, 0, 'k*', markersize=15)
# ax.set_xlabel('x'); ax.set_ylabel('y')
# ax.set_title('SGD vs Momentum vs Adam on L(x,y) = x² + 10y²')
# ax.legend()
# plt.tight_layout()
# plt.savefig('adam_comparison.png', dpi=150)

print(f"\nFinal losses after {steps} steps:")
print(f"  SGD:           {loss(*path_sgd[-1]):.6f}")
print(f"  Momentum β=0.5: {loss(*path_mom[-1]):.6f}")
print(f"  Adam:          {loss(*path_adam[-1]):.6f}")
print(f"Adam lr=0.01: {loss(*path_adam_tuned[-1]):.6f}")
