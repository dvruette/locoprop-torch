import typing

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader

from locoprop import LocoLayer, LocopropCtx


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


class TrainedModel(pl.LightningModule):
    def __init__(self, module: nn.Module, loss: typing.Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                 optimizer_class: type = torch.optim.RMSprop,
                 optimizer_hparams: typing.Dict[str, typing.Any] = dict(lr=2e-5, eps=1e-6, momentum=0.999, alpha=0.9)):
        super().__init__()
        self.mod = module
        self.loss = loss
        self.optimizer_class = optimizer_class
        self.optimizer_hparams = optimizer_hparams

    def forward(self, *args, **kwargs):
        return self.mod(*args, **kwargs)

    def training_step(self, batch, batch_idx):
        return self.loss(self(batch[0]), batch[1])

    def configure_optimizers(self):
        return self.optimizer_class(self.parameters(), **self.optimizer_hparams)


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

    model = TrainedModel(LocoConvNet(), loss_fn).to(device)
    train_dl, test_dl = get_dataloaders(1000)
    trainer = pl.Trainer(max_epochs=50)
    trainer.fit(model, train_dl, test_dl)
