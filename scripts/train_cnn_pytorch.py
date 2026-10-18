"""Same small CNN as train_cnn_keras.py, with an explicit PyTorch training loop (synthetic 28x28 data)."""
import numpy as np
import torch
from torch import nn


def make_data(n_per_class=200, seed=0):
    """Class k = a bright horizontal bar at a class-specific row, plus noise. 10 classes, no download."""
    rng = np.random.default_rng(seed)
    x = rng.normal(0.0, 0.2, (10 * n_per_class, 1, 28, 28)).astype("float32")
    y = np.repeat(np.arange(10), n_per_class)
    for i, k in enumerate(y):
        x[i, 0, 2 + 2 * k:4 + 2 * k, 4:24] += 1.0
    order = rng.permutation(len(y))
    return x[order], y[order].astype("int64")


model = nn.Sequential(
    nn.Conv2d(1, 16, 3), nn.ReLU(), nn.MaxPool2d(2),
    nn.Conv2d(16, 32, 3), nn.ReLU(), nn.MaxPool2d(2),
    nn.Flatten(), nn.Linear(32 * 5 * 5, 10),
)


def main():
    x, y = make_data()
    x, y = torch.from_numpy(x), torch.from_numpy(y)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    # excerpt: train_pytorch
    for epoch in range(3):
        for i in range(0, len(x), 64):
            xb, yb = x[i:i + 64], y[i:i + 64]
            optimizer.zero_grad()           # 1. clear old gradients
            loss = loss_fn(model(xb), yb)   # 2. forward pass + loss
            loss.backward()                 # 3. backpropagation
            optimizer.step()                # 4. update the weights
        accuracy = (model(x).argmax(1) == y).float().mean().item()
        print(f"epoch {epoch + 1}: loss {loss.item():.4f}  accuracy {accuracy:.3f}")
    # endexcerpt


if __name__ == "__main__":
    main()
