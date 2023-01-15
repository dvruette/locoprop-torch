import dataclasses
from contextlib import contextmanager
from typing import Optional

import pytorch_lightning
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


class LocoLayer(nn.Module):
    def __init__(self, module, activation, implicit=False):
        super().__init__()
        self.module = module
        self.activation = activation
        self.storage_for_grad = None
        self.lctx = LocopropCtx(implicit=implicit)

    def forward(self, x=None, hidden=None):
        if x is None and hidden is None:
            raise ValueError("No argument was given. Provide either input or hidden state.")

        if TRAINING and self.training and self.storage_for_grad is None:
            hidden = self.module(x)
            self.storage_for_grad = torch.empty_like(hidden, requires_grad=True)
            hidden = GetGrad.apply(hidden, self.storage_for_grad)
        elif TRAINING and self.training:
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
                    a = self.module(inp)  # pre-activation
                    y = self.activation(a)  # post-activation
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

            # correct input of next layer
            if self.lctx.correction > 0:
                with torch.no_grad():
                    delta = self.module(inp) - hidden
                    correction = self.lctx.correction / delta.std().clamp(min=self.lctx.correction_eps)
                    hidden = hidden + correction.clamp(max=1) * delta
        elif hidden is None:
            hidden = self.module(x)

        if self.lctx.implicit:
            return hidden

        return self.activation(hidden)


class BackwardHook(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, args, model):
        ctx.args = args
        ctx.model = model
        return x

    @staticmethod
    def backward(ctx, dy: torch.Tensor):
        with is_training():
            ctx.model.training_step(*ctx.args)
        return None, None, None


class LocopropTrainer(pytorch_lightning.LightningModule):
    def __init__(self, model: pytorch_lightning.LightningModule,
                 learning_rate: float = 10,
                 iterations: int = 5,
                 correction: float = 0.1,
                 correction_eps: float = 1e-5,
                 inner_opt_class: type = torch.optim.RMSprop,
                 inner_opt_hparams: dict = dict(lr=2e-5, eps=1e-6, momentum=0.999, alpha=0.9), ):
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
        return self.model(*args, **kwargs)

    def training_step(self, inputs, idx):
        with is_training():
            inputs = list(inputs)
            for i, a in enumerate(inputs[:]):
                if isinstance(a, torch.Tensor):
                    inputs[i] = BackwardHook.apply(a.requires_grad_(True), (inputs, idx), self.model)
                    break
            loss = self.model.training_step(inputs, idx)
        return loss

    def validation_step(self, *args, **kwargs):
        return self.model.validation_step(*args, **kwargs)

    def test_step(self, *args, **kwargs):
        return self.model.test_step(*args, **kwargs)

    def predict_step(self, *args, **kwargs):
        return self.model.predict_step(*args, **kwargs)

    def configure_optimizers(self, *args, **kwargs):
        return self.model.configure_optimizers(*args, **kwargs)
