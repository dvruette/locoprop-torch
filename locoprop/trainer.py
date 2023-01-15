from contextlib import contextmanager

import torch
from torch import nn

from locoprop.layer import LocoLayer, _IS_TRAINING


@contextmanager
def is_training():
    try:
        _IS_TRAINING[0] = True
        yield
    finally:
        _IS_TRAINING[0] = False


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
