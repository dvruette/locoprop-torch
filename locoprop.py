import dataclasses
import typing
import warnings

import pytorch_lightning
import torch
from torch import nn
from torch.nn import functional as F
from typing import Optional


@dataclasses.dataclass
class LocopropCtx:
    implicit: bool = False
    learning_rate: float = 10
    iterations: int = 5
    correction: float = 0.1
    correction_eps: float = 1e-5
    optimizer: Optional[torch.optim.Optimizer] = None


class GetGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, y: torch.Tensor):
        return x

    @staticmethod
    def backward(ctx, dy: torch.Tensor):
        return dy, dy


class LocoLayer(nn.Module):
    def __init__(self, module, activation, lctx: LocopropCtx):
        super().__init__()
        self.module = module
        self.activation = activation
        self.storage_for_grad = None
        self.lctx = lctx

    def forward(self, x=None, hidden=None):
        if x is None and hidden is None:
            raise ValueError("No argument was given. Provide either input or hidden state.")

        if hidden is None:
            hidden = self.module(x)
        if self.storage_for_grad is None:
            self.storage_for_grad = torch.empty_like(hidden, requires_grad=True)
            hidden = GetGrad.apply(hidden, self.storage_for_grad)
        else:
            dy = self.storage_for_grad.grad
            self.module.requires_grad_(True)
            original_params = {n: p.clone() for n, p in self.module.named_parameters()}

            opt = self.lctx.optimizer
            base_lrs = [group["lr"] for group in opt.param_groups]
            opt.zero_grad()

            for i in range(self.lctx.iterations):
                for base_lr, group in zip(base_lrs, opt.param_groups):
                    group["lr"] = base_lr * max(1.0 - i / self.lctx.iterations, 0.25)
                with torch.enable_grad():
                    inp = x.detach().requires_grad_(True)
                    a = self.module.module(inp)  # pre-activation
                    y = self.module.activation(a)  # post-activation
                if i == 0:
                    old_out = a.detach()
                    with torch.no_grad():
                        post_target = (y - self.lctx.learning_rate * dy).detach()

                torch.autograd.backward([y], [(y - post_target) / a.size(0)], inputs=list(self.module.parameters()))
                opt.step()
                opt.zero_grad()

            for base_lr, group in zip(base_lrs, opt.param_groups):
                group["lr"] = base_lr

            for n, p in self.module.named_parameters():
                p.grad = original_params[n].data - p.data
                p.set_(original_params[n].data)
            self.storage_for_grad = None

            # correct input of next layer
            if self.lctx.correction > 0:
                with torch.no_grad():
                    delta = self.module(inp) - old_out
                    correction = self.lctx.correction / delta.std().clamp(min=self.lctx.correction_eps)
                    hidden = old_out + correction.clamp(max=1) * delta

        if self.implicit:
            return hidden

        return self.activation(hidden)


class LocopropTrainer(pytorch_lightning.LightningModule):
    def __init__(self, model: pytorch_lightning.LightningModule):
        self.model = model

    def training_step(self, *args, **kwargs):
        loss = self.model.training_step(*args, **kwargs)
        loss.backward()
        self.model.training_step(*args, **kwargs)
        return loss.detach().requires_grad_(True)

    def validation_step(self, *args, **kwargs):
        return self.model.validation_step(*args, **kwargs)

    def test_step(self, *args, **kwargs):
        return self.model.test_step(*args, **kwargs)

    def predict_step(self, *args, **kwargs):
        return self.model.predict_step(*args, **kwargs)

    def configure_optimizers(self, *args, **kwargs):
        return self.model.configure_optimizers(*args, **kwargs)
