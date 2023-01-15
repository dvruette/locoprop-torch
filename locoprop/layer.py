import dataclasses
from typing import Any, Optional

import torch
from torch import nn

@dataclasses.dataclass
class LocopropCtx:
    implicit: bool = False
    learning_rate: float = 10
    iterations: int = 5
    optimizer: Optional[torch.optim.Optimizer] = None


class LocoLayer(nn.Module):
    def __init__(self, module: nn.Module, activation: nn.Module, implicit: bool = False):
        super().__init__()
        self.module = module
        self.activation = activation
        self._ctx = LocopropCtx(implicit=implicit)

    def update_context(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self._ctx, k, v)

    def inner(self, x):
        hidden = self.module(x)
        if not self._ctx.implicit:
            hidden = self.activation(hidden)
        return hidden

    def forward(self, x):
        return LocoFn.apply(self, self._ctx, x.requires_grad_(True))


class LocoFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, module: LocoLayer, lctx: LocopropCtx, inp: torch.Tensor):
        ctx.module = module
        ctx.inp = inp
        ctx.lctx = lctx
        return module.inner(inp)

    @staticmethod
    def backward(ctx, dy: torch.Tensor):
        ctx.module.requires_grad_(True)
        original_params = {n: p.clone() for n, p in ctx.module.named_parameters()}

        opt = ctx.lctx.optimizer
        base_lrs = [group["lr"] for group in opt.param_groups]
        opt.zero_grad()

        for i in range(ctx.lctx.iterations):
            for base_lr, group in zip(base_lrs, opt.param_groups):
                group["lr"] = base_lr * max(1.0 - i / ctx.lctx.iterations, 0.25)
            with torch.enable_grad():
                inp = ctx.inp.detach().requires_grad_(True)
                a = ctx.module.module(inp)  # pre-activation
                y = ctx.module.activation(a)  # post-activation
            if i == 0:
                if ctx.lctx.implicit:
                    next_grad, = torch.autograd.grad([a], [inp], [dy], allow_unused=True, retain_graph=True)
                else:
                    a = a.requires_grad_(True)
                    dy, next_grad = torch.autograd.grad([y], [a, inp], [dy], allow_unused=True, retain_graph=True)
                with torch.no_grad():
                    post_target = (y - ctx.lctx.learning_rate * dy).detach()

            torch.autograd.backward([y], [(y - post_target) / a.size(0)], inputs=list(ctx.module.parameters()))
            opt.step()
            opt.zero_grad()

        for base_lr, group in zip(base_lrs, opt.param_groups):
            group["lr"] = base_lr

        for n, p in ctx.module.named_parameters():
            p.grad = original_params[n].data - p.data
            p.set_(original_params[n].data)

        return None, None, next_grad
