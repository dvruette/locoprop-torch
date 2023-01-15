import copy
import dataclasses
import typing

import torch
from torch import nn


@dataclasses.dataclass
class LocopropCtx:
    learning_rate: float = 10
    base_lr: float = 2e-5
    implicit: bool = True
    iterations: int = 5


class LocoLayer(nn.Module):
    def __init__(self, module: nn.Module, activation: nn.Module, lctx: typing.Optional[LocopropCtx] = None, **kwargs):
        super().__init__()
        self.module = module
        self.activation = activation
        if lctx is None:
            lctx = LocopropCtx()
        self.lctx = copy.deepcopy(lctx)
        for k, v in kwargs:
            setattr(self.lctx, k, v)

    def inner(self, x):
        hidden = self.module(x)
        if not self.lctx.implicit:
            hidden = self.activation(hidden)
        return hidden

    def forward(self, x):
        return locofn(self, self.lctx, x)


class LocoFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx: typing.Any, module: LocoLayer, lctx: LocopropCtx, inp: torch.Tensor):
        ctx.module = module
        ctx.inp = inp
        ctx.lctx = lctx
        return module.inner(inp)

    @staticmethod
    def backward(ctx, dy: torch.Tensor):
        ctx.module.requires_grad_(True)
        opt = torch.optim.SGD(ctx.module.parameters(), ctx.lctx.base_lr)
        original_params = {n: p.clone() for n, p in ctx.module.named_parameters()}
        inp = ctx.inp.detach().requires_grad_(True)
        opt.zero_grad()
        for i in range(ctx.lctx.iterations):
            opt.param_groups[0]["lr"] = ctx.lctx.base_lr * max(1.0 - i / ctx.lctx.iterations, 0.25)
            with torch.enable_grad():
                out = ctx.module.module(inp)
            if i == 0:
                with torch.no_grad():
                    post_target = (out - ctx.lctx.learning_rate * dy).detach()
            torch.autograd.backward([out], [(ctx.module.activation(out) - post_target) / out.size(0)])
            if i == 0:
                grad = inp.grad
            inp = inp.detach().requires_grad_(True)
            opt.step()
            opt.zero_grad()
        for n, p in ctx.module.named_parameters():
            p.grad = original_params[n].data - p.data
            p.data = original_params[n].data
        ctx.module.requires_grad_(False)
        return None, None, grad


locofn = LocoFn.apply
