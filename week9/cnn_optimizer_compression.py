import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import time

# ============================================================
# Stage 1: Data Pipeline
# ============================================================

# Step 1: Define preprocessing
# ToTensor() converts pixels from 0-255 integers to 0.0-1.0 floats
# Normalize() shifts range from [0, 1] to [-1, 1] (centered at zero)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5,), std=(0.5,))
])

# Step 2: Download and load Fashion-MNIST
# train=True  → 60,000 training images
# train=False → 10,000 test images
# transform=  → apply our preprocessing to every image automatically
train_dataset = datasets.FashionMNIST(
    root='./data', train=True, download=True, transform=transform
)
test_dataset = datasets.FashionMNIST(
    root='./data', train=False, download=True, transform=transform
)

# Step 3: Create DataLoaders — serve images in batches of 128
# shuffle=True for training (stochastic in SGD)
# shuffle=False for test (same order every time = reproducible results)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

# The 10 clothing categories
class_names = [
    'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
]

# Step 4: Sanity check — look at the data before training
fig, axes = plt.subplots(2, 8, figsize=(14, 4))
for i, ax in enumerate(axes.flat):
    image, label = train_dataset[i]    # image shape: (1, 28, 28)
    ax.imshow(image.squeeze(), cmap='gray')  # squeeze: (1,28,28) → (28,28)
    ax.set_title(class_names[label], fontsize=9)
    ax.axis('off')

plt.suptitle('Fashion-MNIST — First 16 Samples', fontsize=13)
plt.tight_layout()
plt.savefig('fashion_mnist_samples.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: fashion_mnist_samples.png")

# Step 5: Print stats so we know the pipeline works
print(f"\nDataset stats:")
print(f"  Training samples: {len(train_dataset)}")
print(f"  Test samples:     {len(test_dataset)}")
print(f"  Batch size:       128")
print(f"  Training batches: {len(train_loader)}")
print(f"  Image shape:      {train_dataset[0][0].shape}")

# Quick check: grab one batch and print its shape
images, labels = next(iter(train_loader))
print(f"\nOne batch:")
print(f"  Images shape: {images.shape}")   # should be (128, 1, 28, 28)
print(f"  Labels shape: {labels.shape}")   # should be (128,)
print(f"  Pixel range:  [{images.min():.1f}, {images.max():.1f}]")  # should be [-1.0, 1.0]


# ============================================================
# Stage 2: CNN Model
# ============================================================

class FashionCNN(nn.Module):
    def __init__(self):
        super().__init__()

        # --- Section 1: Conv blocks (feature extractor) ---

        # Block 1: detect simple patterns (edges, textures)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        # in_channels=1   → grayscale image (1 channel)
        # out_channels=16  → learn 16 different filters (16 types of patterns)
        # kernel_size=3    → each filter is 3×3 pixels
        # padding=1        → pad edges with zeros so output stays 28×28

        # Block 2: detect complex patterns (combinations of edges)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        # in_channels=16   → takes the 16 feature maps from block 1
        # out_channels=32   → learn 32 filters on top of those

        # Pooling: shrink spatial size by half (keeps patterns, drops exact positions)
        self.pool = nn.MaxPool2d(kernel_size=2)

        # --- Section 2: FC head (classifier) ---

        # After 2 conv+pool blocks: 32 channels × 7 × 7 pixels = 1568 numbers
        self.fc1 = nn.Linear(32 * 7 * 7, 64)   # compress 1568 → 64
        self.fc2 = nn.Linear(64, 10)            # 64 → 10 class scores

        self.relu = nn.ReLU()

    def forward(self, x):
        # x shape: (batch_size, 1, 28, 28)

        # Block 1: conv → relu → pool
        x = self.conv1(x)      # (batch, 1, 28, 28) → (batch, 16, 28, 28)
        x = self.relu(x)       # nonlinearity — without this, layers collapse
        x = self.pool(x)       # (batch, 16, 28, 28) → (batch, 16, 14, 14)

        # Block 2: conv → relu → pool
        x = self.conv2(x)      # (batch, 16, 14, 14) → (batch, 32, 14, 14)
        x = self.relu(x)
        x = self.pool(x)       # (batch, 32, 14, 14) → (batch, 32, 7, 7)

        # Flatten: reshape 2D grid into 1D vector for FC layer
        x = x.view(x.size(0), -1)  # (batch, 32, 7, 7) → (batch, 1568)

        # Classify
        x = self.fc1(x)        # (batch, 1568) → (batch, 64)
        x = self.relu(x)
        x = self.fc2(x)        # (batch, 64) → (batch, 10)

        return x


# Sanity check: create model, pass one batch through, count parameters
model = FashionCNN()
output = model(images)   # images from Stage 1's batch
print(f"\nModel check:")
print(f"  Input shape:  {images.shape}")
print(f"  Output shape: {output.shape}")   # should be (128, 10)

total_params = sum(p.numel() for p in model.parameters())
print(f"  Total parameters: {total_params:,}")


# ============================================================
# Stage 3: Training + Evaluation Functions
# ============================================================

def train_one_epoch(model, loader, optimizer, loss_fn):
    """Train for one epoch. Returns average loss."""
    model.train()
    total_loss = 0

    for images, labels in loader:
        # Same 5-step loop from Week 8
        outputs = model(images)           # 1. Forward pass
        loss = loss_fn(outputs, labels)    # 2. Compute loss
        loss.backward()                    # 3. Backward pass (gradients)
        optimizer.step()                   # 4. Update weights
        optimizer.zero_grad()              # 5. Reset gradients

        total_loss += loss.item()

    return total_loss / len(loader)


def evaluate(model, loader):
    """Test accuracy. No gradients needed."""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            outputs = model(images)
            _, predicted = outputs.max(1)  # highest score = predicted class
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    return correct / total


# ============================================================
# Stage 4: Run the Comparison — 3 optimizers, same model
# ============================================================

EPOCHS = 10
SEED = 42

# Three optimizers to compare
optimizer_configs = {
    'SGD (lr=0.01, momentum=0.9)': lambda params: optim.SGD(params, lr=0.01, momentum=0.9),
    'Adam (lr=1e-3)':              lambda params: optim.Adam(params, lr=1e-3),
    'AdamW (lr=1e-3, wd=1e-2)':   lambda params: optim.AdamW(params, lr=1e-3, weight_decay=1e-2),
}

# Store results for plotting
all_results = {}

for name, make_optimizer in optimizer_configs.items():
    print(f"\n{'='*50}")
    print(f"Training with: {name}")
    print(f"{'='*50}")

    # Fixed seed → identical starting weights for fair comparison
    torch.manual_seed(SEED)

    # Fresh model each run (same random init thanks to seed)
    model = FashionCNN()
    optimizer = make_optimizer(model.parameters())
    loss_fn = nn.CrossEntropyLoss()

    train_losses = []
    test_accs = []
    start_time = time.time()

    for epoch in range(EPOCHS):
        loss = train_one_epoch(model, train_loader, optimizer, loss_fn)
        acc = evaluate(model, test_loader)

        train_losses.append(loss)
        test_accs.append(acc)

        print(f"  Epoch {epoch+1:2d}/{EPOCHS} — Loss: {loss:.4f} — Test Acc: {acc:.4f}")

    elapsed = time.time() - start_time
    print(f"  Time: {elapsed:.1f}s — Final test accuracy: {test_accs[-1]:.4f}")

    all_results[name] = {
        'train_losses': train_losses,
        'test_accs': test_accs,
        'time': elapsed
    }


# ============================================================
# Stage 5: Plot the Results
# ============================================================

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

colors = ['#e74c3c', '#2ecc71', '#3498db']

for i, (name, result) in enumerate(all_results.items()):
    epochs_range = range(1, EPOCHS + 1)

    # Left panel: training loss
    ax1.plot(epochs_range, result['train_losses'],
             label=name, color=colors[i], linewidth=2)

    # Right panel: test accuracy
    ax2.plot(epochs_range, result['test_accs'],
             label=name, color=colors[i], linewidth=2)

ax1.set_xlabel('Epoch')
ax1.set_ylabel('Training Loss')
ax1.set_title('Training Loss per Epoch')
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.set_xlabel('Epoch')
ax2.set_ylabel('Test Accuracy')
ax2.set_title('Test Accuracy per Epoch')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.suptitle('Optimizer Comparison on Fashion-MNIST CNN', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('optimizer_comparison.png', dpi=150, bbox_inches='tight')
plt.close()

print(f"\n{'='*50}")
print("SUMMARY")
print(f"{'='*50}")
for name, result in all_results.items():
    print(f"  {name}")
    print(f"    Final test accuracy: {result['test_accs'][-1]:.4f}")
    print(f"    Time: {result['time']:.1f}s")
print(f"\nSaved: optimizer_comparison.png")