import time

import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader

from locoprop import LocoLayer, LocopropTrainer


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class LocoConvNet(nn.Sequential):
    def __init__(
        self,
        classes=10,
        kernel_size=5,
        stride=1,
        padding=2,
        activation_cls=nn.Tanh,
    ):
        super().__init__()
        self.add_module(
            "stage_0",
            LocoLayer(nn.Conv2d(1, 16, kernel_size, stride, padding), activation_cls()),
        )
        self.add_module("pool_0", nn.MaxPool2d(2))
        self.add_module(
            "stage_1",
            LocoLayer(
                nn.Conv2d(16, 32, kernel_size, stride, padding), activation_cls()
            ),
        )
        self.add_module("pool_1", nn.MaxPool2d(2))
        self.add_module("flatten", Flatten())
        self.add_module(
            "output",
            LocoLayer(
                nn.Linear(32 * 7 * 7, classes, bias=False),
                nn.Softmax(dim=-1),
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


def loss_fn(logits, ys):
    return F.cross_entropy(logits, ys)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(1)

    model = LocoConvNet().to(device)
    train_dl, test_dl = get_dataloaders(1000)
    trainer = LocopropTrainer(
        model,
        loss_fn,
        learning_rate=100,
        local_iterations=5,
        optimizer_hparams=dict(lr=2e-5, eps=1e-5, momentum=0.999, alpha=0.9),
    )

    train_losses, test_losses, test_accs, times = [], [], [], []

    with torch.no_grad():
        train_ls = []
        for xs, ys in train_dl:
            xs, ys = xs.to(device), ys.to(device)
            logits = model(xs)
            loss = loss_fn(logits, ys)
            train_ls.append(loss.item() * xs.size(0))
        train_loss = sum(train_ls) / len(train_dl.dataset)
        train_losses.append(train_loss)

        model.eval()
        test_ls = []
        for xs, ys in test_dl:
            xs, ys = xs.to(device), ys.to(device)
            logits = model(xs)
            loss = loss_fn(logits, ys)
            test_ls.append(loss.item() * xs.size(0))
        test_loss = sum(test_ls) / len(test_dl.dataset)
        test_losses.append(test_loss)
    times.append(0)

    start_time = time.time()

    for epoch in tqdm.trange(50, desc="Epochs"):
        model.train()
        train_ls = []
        for xs, ys in train_dl:
            xs, ys = xs.to(device), ys.to(device)

            loss = trainer.step(xs, ys)
            train_ls.append(loss.detach().item() * xs.size(0))

        train_loss = sum(train_ls) / len(train_dl.dataset)
        train_losses.append(train_loss)

        with torch.no_grad():
            model.eval()
            test_ls, test_as = [], []
            for xs, ys in test_dl:
                xs, ys = xs.to(device), ys.to(device)
                logits = model(xs)
                loss = loss_fn(logits, ys)
                acc = (logits.argmax(dim=-1) == ys).float().mean()
                test_ls.append(loss.item() * xs.size(0))
                test_as.append(acc.item() * xs.size(0))
            test_loss = sum(test_ls) / len(test_dl.dataset)
            test_acc = sum(test_as) / len(test_dl.dataset)
            test_losses.append(test_loss)
            test_accs.append(test_acc)
        times.append(time.time() - start_time)
        print(
            f"Epoch {epoch+1:2d}: {train_loss=:.4f} | {test_loss=:.4f} | {test_acc=:.4f}"
        )
