import typing
import warnings

import torch
from torch import nn

from locoprop.layer import LocoLayer


class LocopropTrainer:
    def __init__(
            self,
            model: nn.Sequential,
            loss_fn: typing.Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
            optimizer_class=torch.optim.RMSprop,
            optimizer_hparams: typing.Dict = dict(lr=2e-5, eps=1e-6, momentum=0.999, alpha=0.9),
            learning_rate: float = 10,
            local_iterations: int = 5,
            momentum: float = 0.0,
            correction: float = 0.1,
    ):
        self.model = model
        self.loss_fn = loss_fn
        self.learning_rate = learning_rate
        self.local_iterations = local_iterations
        self.momentum = momentum
        self.correction = correction

        self.opts = []
        for layer in model:
            trainable_params = sum(
                p.numel() for p in layer.parameters() if p.requires_grad
            )
            if trainable_params > 0 and not isinstance(layer, LocoLayer):
                warnings.warn(
                    f"Layer {layer} is trainable but not a LocoLayer. Its parameters will not be updated."
                )

            if isinstance(layer, LocoLayer) and trainable_params > 0:
                self.opts.append(
                    optimizer_class(layer.parameters(), **optimizer_hparams)
                )
            else:
                self.opts.append(None)
        self.grads = []

    def step(self, xs, ys):
        inps = []
        hiddens = []
        curr = xs

        for layer, opt in zip(self.model, self.opts):
            inps.append(curr.detach())
            if opt is not None and isinstance(layer, LocoLayer):
                hidden = layer.module(curr)
                hidden.requires_grad_(True)
                hidden.retain_grad()
                hiddens.append(hidden)
                curr = layer(hidden=hidden)
            else:
                hiddens.append(None)
                curr = layer(curr)

        hidden_loss = self.loss_fn(curr, ys)
        hidden_loss.backward()

        grads = [None if h is None or not hasattr(h, "grad") or h.grad is None else h.grad for h in hiddens]
        if self.momentum > 0:
            if len(self.grads) == 0:
                self.grads = grads
            else:
                self.grads = [None if m is None else (1 - self.momentum) * g + self.momentum * m for g, m in zip(grads, self.grads)]
            grads = self.grads

        for p in self.model.parameters():
            p.requires_grad = True

        for i, (opt, layer, grad) in enumerate(zip(self.opts, self.model, grads)):
            if opt is None:
                continue
            if not isinstance(layer, LocoLayer):
                raise ValueError(
                    f"Expected trainable layers to be instance of `LocoLayer` but found `{layer.__class__}`")

            inp = inps[i]
            with torch.no_grad():
                a = layer.module(inp)
                y = layer.activation(a)
                post_target = (y - self.learning_rate * grad).detach()

            base_lr = opt.param_groups[0]["lr"]
            for j in range(self.local_iterations):
                opt.param_groups[0]["lr"] = base_lr * max(1.0 - j / self.local_iterations, 0.25)
                opt.zero_grad()

                """
                Let `f := "activation function"` and `y = post_target`.

                original:
                ```
                a = pseudo_inverse(y)
                return grad((F(pre_act) - F(a) - f(a)^T * (pre_act - a)).sum())
                ```

                Since `a` is constant w.r.t. `pre_act`, it does not require gradients, so we can simplify:
                `return grad((F(pre_act) - y^T * pre_act).sum())`
                with `f(a) = f(pseudo_inverse(y)) = y` by definition of the pseudo-inverse.

                The einsum can further be integrated using:
                `backward(outputs=[F(pre_act), pre_act], grad_outputs=[ones_like(pre_act), -y])`

                since `grad(F)(x) = f` by definition of `F`, this can be computed as:
                `backward(outputs=[pre_act], grad_outputs=[f(pre_act) - y])`
                """
                pre_act = layer.module(inp)
                torch.autograd.backward([pre_act], [(layer.activation(pre_act) - post_target) / inp.size(0)])
                opt.step()
            opt.param_groups[0]["lr"] = base_lr

            # correct input of next layer
            if self.correction > 0 and i + 1 < len(inps):
                with torch.no_grad():
                    delta = layer(inp) - inps[i + 1]
                    norm = delta.norm() + 1e-5
                    inps[i + 1] += min(norm, self.correction * delta.size(1) ** 0.5) * delta / norm

        return hidden_loss