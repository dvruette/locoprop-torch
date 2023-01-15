import dataclasses
from contextlib import contextmanager
from typing import Optional, Any

import torch
from torch import nn

TRAINING = False


@contextmanager
def is_training():
    global TRAINING
    try:
        TRAINING = True
        yield
        TRAINING = False
    finally:
        TRAINING = False


@dataclasses.dataclass
class LocopropCtx:
    implicit: bool = False
    learning_rate: float = 10
    iterations: int = 5
    correction: float = 0.
    correction_eps: float = 1e-5
    optimizer: Optional[torch.optim.Optimizer] = None


class GetGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, y: torch.Tensor):
        return x

    @staticmethod
    def backward(ctx, dy: torch.Tensor):
        return dy, dy


class BackwardHook(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, aux):
        ctx.aux = aux
        return x

    @staticmethod
    def backward(ctx, dy: torch.Tensor):
        with is_training():
            args, kwargs, model = ctx.aux
            model.forward(*args, **kwargs)
        return None, None


def hook_fn(fn: torch.autograd.Function, args, kwargs, aux):
    for i, a in enumerate(args[:]):
        if isinstance(a, torch.Tensor):
            args[i] = fn.apply(a.requires_grad_(True), (args, kwargs, aux))
            break
    else:
        for k, v in list(kwargs.items()):
            if isinstance(v, torch.Tensor):
                kwargs[k] = fn.apply(v.requires_grad_(True), (args, kwargs, aux))
                break
        else:
            raise ValueError("No tensor provided")


def maybe_detach(x: Any):
    if isinstance(x, torch.Tensor):
        return x.detach().requires_grad_(True)
    return x


class LocoLayer(nn.Module):
    def __init__(self, module, activation, implicit=False):
        super().__init__()
        self.module = module
        self.activation = activation
        self.storage_for_grad = None
        self.lctx = LocopropCtx(implicit=implicit)

    def _training_first_pass(self, *args, **kwargs):
        hidden = self.module(*args, **kwargs)
        self.storage_for_grad = torch.empty_like(hidden, requires_grad=True)
        return GetGrad.apply(hidden, self.storage_for_grad)

    def _training_second_pass(self, *args, **kwargs):
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
                # pre-activation:
                args = [maybe_detach(a) for a in args]
                kwargs = {k: maybe_detach(v) for k, v in kwargs.items()}
                a = self.module(*args, **kwargs)
                y = self.activation(a)  # <- post-activation
            if i == 0:
                with torch.no_grad():
                    hidden = a.detach()
                    post_target = (y - self.lctx.learning_rate * dy).detach()

            torch.autograd.backward([y], [(y - post_target) / a.size(0)], inputs=list(self.module.parameters()))
            opt.step()
            opt.zero_grad()

        for base_lr, group in zip(base_lrs, opt.param_groups):
            group["lr"] = base_lr

        with torch.no_grad():
            for n, p in self.module.named_parameters():
                p.grad = original_params[n].data - p.data
                p.set_(original_params[n].data)

        self.storage_for_grad = None
        if self.lctx.correction <= 0:
            return hidden

        # correct input of next layer
        with torch.no_grad():
            delta = self.module(*args, **kwargs) - hidden
            correction = self.lctx.correction / delta.std().clamp(min=self.lctx.correction_eps)
            return hidden + correction.clamp(max=1) * delta

    def forward(self, *args, **kwargs):
        if TRAINING and self.training:
            if self.storage_for_grad is None:
                hidden = self._training_first_pass(*args, **kwargs)
            else:
                hidden = self._training_second_pass(*args, **kwargs)
        else:
            hidden = self.module(*args, **kwargs)

        if self.lctx.implicit:
            return hidden

        return self.activation(hidden)


class LocopropTrainer(nn.Module):
    def __init__(self,
                 model: nn.Module,
                 learning_rate: float = 10,
                 iterations: int = 5,
                 correction: float = 0.1,
                 correction_eps: float = 1e-5,
                 inner_opt_class: type = torch.optim.RMSprop,
                 inner_opt_hparams: dict = dict(lr=2e-5, eps=1e-6, momentum=0.999, alpha=0.9)):
        super().__init__()

        for m in model.modules():
            if isinstance(m, LocoLayer):
                opt = inner_opt_class(m.parameters(), **inner_opt_hparams)
                m.lctx.learning_rate = learning_rate
                m.lctx.iterations = iterations
                m.lctx.optimizer = opt
                m.lctx.correction = correction
                m.lctx.correction_eps = correction_eps
        self.model = model

    def forward(self, *args, **kwargs):
        with is_training():
            args = list(args)
            hook_fn(BackwardHook, args, kwargs, self.model)
            return self.model(*args, **kwargs)
