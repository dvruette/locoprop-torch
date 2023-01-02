import time

import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader

from locoprop import LocoLayer


"""
Train model with standard BP
Dataset: MNIST
Model: Deep autoencoder (784, 1000, 500, 250, 30, 250, 500, 1000, 784)
Loss: Binary cross-entropy
Optimizer: Adam
"""

class LocoNet(nn.Sequential):
    def __init__(
        self,
        input_dim=784,
        output_dim=784,
        hidden_dims=[1000, 500, 250, 30, 250, 500, 1000],
        activation_cls=nn.Tanh,
    ):
        super().__init__()
        self.add_module(
            "input", LocoLayer(nn.Linear(input_dim, hidden_dims[0]), activation_cls())
        )
        for i, (d_in, d_out) in enumerate(zip(hidden_dims[:-1], hidden_dims[1:])):
            self.add_module(
                f"stage{i}", LocoLayer(nn.Linear(d_in, d_out), activation_cls())
            )
        self.add_module(
            "output",
            LocoLayer(
                nn.Linear(hidden_dims[-1], output_dim, bias=False),
                nn.Sigmoid(),
                implicit=True,
            ),
        )


def get_dataloaders(batch_size=128):
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    train_ds = torchvision.datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )
    val_ds = torchvision.datasets.MNIST(
        root="./data", train=False, download=True, transform=transform
    )
    train_dl = DataLoader(train_ds, batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size, shuffle=False)
    return train_dl, val_dl


def loss_fn(logits, y):
    return (
        F.binary_cross_entropy_with_logits(logits, y, reduction="none")
        .sum(dim=-1)
        .mean()
    )


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(1)

    epochs = 100
    lr = 1e-3
    batch_size = 1000

    model = LocoNet().to(device)
    train_dl, val_dl = get_dataloaders(batch_size)
    optimizer = torch.optim.Adam(model.parameters(), lr)

    time_log = []
    train_log = []
    val_log = []

    # compute initial losses
    with torch.no_grad():
        model.train()
        losses = []
        for i, (x, _) in zip(range(32), train_dl):
            x = x.view(x.size(0), -1).to(device)

            logits = model(x)
            loss = loss_fn(logits, x)
            losses.append(loss)
        train_loss = torch.stack(losses).mean().item()
        train_log.append(train_loss)

    model.eval()
    losses = []
    for x, _ in val_dl:
        x = x.view(x.size(0), -1).to(device)

        logits = model(x)
        loss = loss_fn(logits, x)
        losses.append(loss)
    val_loss = torch.stack(losses).mean().item()
    val_log.append(val_loss)
    print(f"Initial: {train_loss=:.2f} | {val_loss=:.2f}")

    time_log.append(0)
    start_time = time.time()
    for i in tqdm.trange(epochs, desc="Epoch"):
        model.train()
        losses = []
        for x, _ in train_dl:
            x = x.view(x.size(0), -1).to(device)

            logits = model(x)
            loss = loss_fn(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss)
        train_loss = torch.stack(losses).mean().item()
        train_log.append(train_loss)

    with torch.no_grad():
        model.eval()
        losses = []
        for x, _ in val_dl:
            x = x.view(x.size(0), -1).to(device)

            logits = model(x)
            loss = loss_fn(logits, x)
            losses.append(loss)
        val_loss = torch.stack(losses).mean().item()
        val_log.append(val_loss)
    time_log.append(time.time() - start_time)

    print(f"Epoch {i + 1:02d}: {train_loss=:.4f} | {val_loss=:.4f}")


if __name__ == "__main__":
    main()
