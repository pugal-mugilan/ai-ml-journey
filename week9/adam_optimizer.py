import numpy as np
import matplotlib.pyplot as plt


# --- Toy loss: L(x, y) = x² + 10y² ---
def loss(x, y):
    return x ** 2 + 10 * y ** 2


def grad(x, y):
    return np.array([2 * x, 20 * y])


# --- Adam optimizer with optional LR schedule ---
def adam_run(lr_schedule, steps=200, beta1=0.9, beta2=0.999, eps=1e-8):
    w = np.array([-2.5, 1.2])  # starting point
    m = np.zeros(2)
    v = np.zeros(2)

    losses = []
    path = [w.copy()]

    for t in range(1, steps + 1):
        g = grad(w[0], w[1])
        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * g ** 2
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)

        lr = lr_schedule(t)  # <-- this is the only difference
        w = w - lr * m_hat / (np.sqrt(v_hat) + eps)

        losses.append(loss(w[0], w[1]))
        path.append(w.copy())

    return losses, np.array(path)


# --- Schedule functions ---
steps = 200


# 1. Constant LR
def constant_lr(t):
    return 0.01


# 2. Cosine annealing
def cosine_lr(t):
    lr_max = 0.05  # start high
    lr_min = 0.0001
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + np.cos(np.pi * (t - 1) / (steps - 1)))


# 3. Warmup + cosine
warmup_steps = 10


def warmup_cosine_lr(t):
    lr_max = 0.05
    lr_min = 0.0001
    if t <= warmup_steps:
        return lr_min + (lr_max - lr_min) * (t / warmup_steps)
    else:
        remaining = steps - warmup_steps
        elapsed = t - warmup_steps
        return lr_min + 0.5 * (lr_max - lr_min) * (1 + np.cos(np.pi * elapsed / remaining))


# --- Run all three ---
losses_const, path_const = adam_run(constant_lr, steps)
losses_cosine, path_cosine = adam_run(cosine_lr, steps)
losses_warmup, path_warmup = adam_run(warmup_cosine_lr, steps)

# --- Plot 1: Loss vs Step ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax1 = axes[0]
ax1.plot(losses_const, color='gray', linewidth=2, label=f'Constant lr=0.01 (final: {losses_const[-1]:.6f})')
ax1.plot(losses_cosine, color='teal', linewidth=2, label=f'Cosine lr=0.05→0.0001 (final: {losses_cosine[-1]:.6f})')
ax1.plot(losses_warmup, color='coral', linewidth=2, label=f'Warmup+cosine (final: {losses_warmup[-1]:.6f})')
ax1.set_xlabel('Step')
ax1.set_ylabel('Loss')
ax1.set_title('Loss vs Step')
ax1.legend(fontsize=9)
ax1.set_yscale('log')
ax1.grid(alpha=0.3)

# --- Plot 2: Paths on contour ---
ax2 = axes[1]
x_range = np.linspace(-3, 3, 200)
y_range = np.linspace(-1.5, 1.5, 200)
X, Y = np.meshgrid(x_range, y_range)
Z = X ** 2 + 10 * Y ** 2

ax2.contour(X, Y, Z, levels=30, cmap='Blues', alpha=0.5)
ax2.plot(path_const[:, 0], path_const[:, 1], 'o-', color='gray', markersize=2, linewidth=1, alpha=0.7, label='Constant')
ax2.plot(path_cosine[:, 0], path_cosine[:, 1], 'o-', color='teal', markersize=2, linewidth=1, alpha=0.7, label='Cosine')
ax2.plot(path_warmup[:, 0], path_warmup[:, 1], 'o-', color='coral', markersize=2, linewidth=1, alpha=0.7,
         label='Warmup+cosine')
ax2.plot(-2.5, 1.2, 'k*', markersize=15, label='Start')
ax2.plot(0, 0, 'rx', markersize=12, markeredgewidth=3, label='Minimum')
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_title('Optimizer Paths on Loss Surface')
ax2.legend(fontsize=9)

plt.tight_layout()
plt.savefig('adam_schedule_comparison.png', dpi=150)

# --- Print final losses ---
print(f"Constant lr=0.01 final loss:    {losses_const[-1]:.8f}")
print(f"Cosine final loss:              {losses_cosine[-1]:.8f}")
print(f"Warmup+cosine final loss:       {losses_warmup[-1]:.8f}")