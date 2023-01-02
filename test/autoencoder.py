import time

import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader

from locoprop import LocoLayer, LocopropTrainer


"""
Dataset: MNIST
Model: Deep autoencoder (784, 1000, 500, 250, 30, 250, 500, 1000, 784)
Loss: Binary cross-entropy
Optimizer: LocoProp (RMSprop)
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

    model = LocoNet().to(device)
    train_dl, test_dl = get_dataloaders(1000)
    trainer = LocopropTrainer(model, loss_fn)

    train_losses, test_losses, times = [], [], []

    with torch.no_grad():
        train_ls = []
        for xs, _ in train_dl:
            xs = xs.view(xs.size(0), -1).cuda()
            logits = model(xs)
            loss = loss_fn(logits, xs)
            train_ls.append(loss.item() * xs.size(0))
        train_loss = sum(train_ls) / len(train_dl.dataset)
        train_losses.append(train_loss)

        model.eval()
        test_ls = []
        for xs, _ in test_dl:
            xs = xs.view(xs.size(0), -1).cuda()
            logits = model(xs)
            loss = loss_fn(logits, xs)
            test_ls.append(loss.item() * xs.size(0))
        test_loss = sum(test_ls) / len(test_dl.dataset)
        test_losses.append(test_loss)
        times.append(0)

    start_time = time.time()

    for epoch in tqdm.trange(50, desc="Epochs"):
        model.train()
        train_ls = []
        for xs, _ in train_dl:
            xs = xs.view(xs.size(0), -1).to(device)

            loss = trainer.step(xs, xs)
            train_ls.append(loss.detach().item() * xs.size(0))

        train_loss = sum(train_ls) / len(train_dl.dataset)
        train_losses.append(train_loss)

        with torch.no_grad():
            model.eval()
            test_ls = []
            for xs, _ in test_dl:
                xs = xs.view(xs.size(0), -1).cuda()
                logits = model(xs)
                loss = loss_fn(logits, xs)
                test_ls.append(loss.item() * xs.size(0))
            test_loss = sum(test_ls) / len(test_dl.dataset)
            test_losses.append(test_loss)
        times.append(time.time() - start_time)
        print(f"Epoch {epoch+1:2d}: {train_loss=:.2f} | {test_loss=:.2f}")
