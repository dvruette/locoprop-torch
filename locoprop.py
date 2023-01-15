import copy
import dataclasses
import typing

import torch
from torch import nn


@dataclasses.dataclass
class LocopropCtx:
    learning_rate: float = 10
    implicit: bool = False
    iterations: int = 5
    base_lr: float = 2e-5
    eps: float = 1e-6
    momentum: float = 0.999
    alpha: float = 0.9


class LocoLayer(nn.Module):
    def __init__(self, module: nn.Module, activation: nn.Module, lctx: typing.Optional[LocopropCtx] = None, **kwargs):
        super().__init__()
        self.module = module
        self.activation = activation
        if lctx is None:
            lctx = LocopropCtx()
        self.lctx = copy.deepcopy(lctx)
        for k, v in kwargs.items():
            setattr(self.lctx, k, v)
        self.opt = torch.optim.RMSprop(module.parameters(), lr=lctx.base_lr, eps=lctx.eps, momentum=lctx.momentum,
                                       alpha=lctx.alpha)

    def inner(self, x):
        hidden = self.module(x)
        if not self.lctx.implicit:
            hidden = self.activation(hidden)
        return hidden

    def forward(self, x):
        return locofn(self, self.lctx, self.opt, x.requires_grad_(True))


class LocoFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx: typing.Any, module: LocoLayer, lctx: LocopropCtx, opt: torch.optim.Optimizer, inp: torch.Tensor):
        ctx.module = module
        ctx.inp = inp
        ctx.lctx = lctx
        ctx.opt = opt
        return module.inner(inp)

    @staticmethod
    def backward(ctx, dy: torch.Tensor):
        ctx.module.requires_grad_(True)
        original_params = {n: p.clone() for n, p in ctx.module.named_parameters()}
        ctx.opt.zero_grad()
        for i in range(ctx.lctx.iterations):
            ctx.opt.param_groups[0]["lr"] = ctx.lctx.base_lr * max(1.0 - i / ctx.lctx.iterations, 0.25)
            with torch.enable_grad():
                inp = ctx.inp.detach().requires_grad_(True)
                out = ctx.module.module(inp)
                act = ctx.module.activation(out)
            if i == 0:
                next_grad, = torch.autograd.grad([act], [inp], [dy], allow_unused=True, retain_graph=True)
                with torch.no_grad():
                    post_target = (act - ctx.lctx.learning_rate * dy).detach()
            torch.autograd.backward([act], [(act - post_target) / out.size(0)], inputs=list(ctx.module.parameters()))
            ctx.opt.step()
            ctx.opt.zero_grad()
        # for n, p in ctx.module.named_parameters():
        #    p.grad = original_params[n].data - p.data
        #    p.data.set_(original_params[n].data)
        return None, None, None, next_grad


locofn = LocoFn.apply
