import dataclasses
from typing import Optional, Any

import torch
from torch import nn

# Using a list to enable read/write operations by different modules.
# Not strictly multiprocessing-compatible, expect unexpected behavior.
_IS_TRAINING = [False]


@dataclasses.dataclass
class LocopropCtx:
    implicit: bool = False
    learning_rate: float = 10
    iterations: int = 5
    correction: float = 0.
    correction_eps: float = 1e-5
    optimizer: Optional[torch.optim.Optimizer] = None


class CloneGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, y: torch.Tensor):
        return x

    @staticmethod
    def backward(ctx, dy: torch.Tensor):
        return dy, dy


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
        return CloneGrad.apply(hidden, self.storage_for_grad)

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
            norm = delta.flatten(1).norm(dim=1).clamp(min=self.lctx.correction_eps)
            mag = (self.lctx.correction / norm * delta.flatten(1).size(1) ** 0.5).clamp(max=1)
            delta = (mag.unsqueeze(-1) * delta.flatten(1)).view(*delta.shape)
            # => magnitude of delta is `correction * sqrt(n)` (n is dimension of hidden state)
            return hidden + delta

    def forward(self, *args, **kwargs):
        if _IS_TRAINING[0] and self.training:
            if self.storage_for_grad is None:
                hidden = self._training_first_pass(*args, **kwargs)
            else:
                hidden = self._training_second_pass(*args, **kwargs)
        else:
            hidden = self.module(*args, **kwargs)

        if self.lctx.implicit:
            return hidden

        return self.activation(hidden)
