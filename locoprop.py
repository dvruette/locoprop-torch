import copy
import dataclasses
import typing

import pytorch_lightning as pl
import torch
from torch import nn


@dataclasses.dataclass
class LocopropCtx:
    learning_rate: float = 10
    base_lr: float = 2e-5
    implicit: bool = True
    iterations: int = 5


class LocoLayer(nn.Module):
    def __init__(self, module: nn.Module, activation: nn.Module, lctx: LocopropCtx, **kwargs):
        super().__init__()
        self.module = module
        self.activation = activation
        self.lctx = copy.deepcopy(lctx)
        for k, v in kwargs:
            setattr(self.lctx, k, v)

    def forward(self, x=None, hidden=None):
        if x is None and hidden is None:
            raise ValueError("No argument was given. Provide either input or hidden state.")

        if hidden is None:
            hidden = self.module(x)

        if not self.lctx.implicit:
            hidden = self.activation(hidden)

        return locofn(self, hidden, self.lctx, (x, None), {})


class LocoFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx: typing.Any, module: LocoLayer, out: torch.Tensor, lctx: LocopropCtx, args: list, kwargs: dict):
        ctx.module = module
        ctx.args = args
        ctx.kwargs = kwargs
        ctx.lctx = lctx
        return out

    @staticmethod
    def backward(ctx, dy: torch.Tensor):
        ctx.module.requires_grad_(True)
        opt = torch.optim.SGD(ctx.module.parameters(), ctx.lctx.base_lr)
        original_params = {n: p.clone() for n, p in ctx.module.named_parameters()}
        for i in range(ctx.iterations):
            opt.param_groups[0]["lr"] = ctx.lctx.base_lr * max(1.0 - i / ctx.lctx.iterations, 0.25)
            with torch.enable_grad():
                out = ctx.module.module(*ctx.args, **ctx.kwargs)
            if i == 0:
                with torch.no_grad():
                    post_target = (out - ctx.lctx.learning_rate * dy).detach()
            torch.autograd.backward([out], [(ctx.module.activation(out) - post_target) / out.size(0)])
            opt.step()
            opt.zero_grad()
        for n, p in ctx.module.named_parameters():
            p.grad = original_params[n].data - p.data
            p.data = original_params[n].data
        ctx.module.requires_grad_(False)
        return None, dy, None, None, None, None, None


locofn = LocoFn.apply


