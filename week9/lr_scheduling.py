import numpy as np
import matplotlib.pyplot as plt

# --- Setup ---
T = 100
lr_max = 0.05
lr_min = 0.0001
epochs = np.arange(1, T + 1)
gamma = 0.5
step_size = 30
warmup_epochs = 5

# --- Schedule 1: Step Decay ---
step_decay_lr = lr_max * (gamma ** (epochs // step_size))

# --- Schedule 2: Cosine Annealing ---
cosine_lr = lr_min + 0.5 * (lr_max - lr_min) * (1 + np.cos(np.pi * (epochs - 1) / (T - 1)))

# --- Schedule 3: Warmup + Cosine ---
warmup_cosine_lr = np.zeros(T)
for i, t in enumerate(epochs):
    if t <= warmup_epochs:
        # Linear ramp from lr_min to lr_max
        warmup_cosine_lr[i] = lr_min + (lr_max - lr_min) * (t / warmup_epochs)
    else:
        # Cosine decay over remaining epochs
        remaining = T - warmup_epochs
        elapsed = t - warmup_epochs
        warmup_cosine_lr[i] = lr_min + 0.5 * (lr_max - lr_min) * (1 + np.cos(np.pi * elapsed / remaining))

# --- Plot ---
plt.figure(figsize=(10, 5))
plt.plot(epochs, step_decay_lr, color='purple', linewidth=2, label='Step decay (γ=0.5, every 30)')
plt.plot(epochs, cosine_lr, color='teal', linewidth=2, label='Cosine annealing')
plt.plot(epochs, warmup_cosine_lr, color='coral', linewidth=2, label='Warmup (5) + cosine')
plt.axhline(y=lr_max, color='gray', linewidth=1, linestyle='--', alpha=0.5, label='Constant LR (baseline)')

plt.xlabel('Epoch')
plt.ylabel('Learning Rate')
plt.title('LR Schedules Compared')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('lr_schedules.png', dpi=150)
