import functools
import math
import typing
import warnings

import torch
import torch.nn as nn


def bregman_div(x: torch.Tensor, y: torch.Tensor, f, F):
    assert len(x.shape) == 2  # (batch_size, d_layer)
    assert x.shape == y.shape
    return F(x) - F(y) - (f(y) * (x - y)).sum(dim=-1)


class LocoLayer(nn.Module):
    def __init__(self, module, activation, implicit=False, eps=1e-5):
        super().__init__()
        self.module = module
        self.activation = activation
        self.implicit = implicit
        self.eps = eps

    def forward(self, x=None, hidden=None):
        if x is None and hidden is None:
            raise ValueError(
                "No argument was given. Provide either input or hidden state."
            )

        if hidden is None:
            hidden = self.module(x)

        if self.implicit:
            return hidden
        return self.activation(hidden)

    def hidden(self, x):
        return self.module(x)

    def bregman_loss(self, x, y):
        pre_act = self.module(x)
        with torch.no_grad():
            # compute pseudo-inverse of y
            if isinstance(self.activation, nn.Sigmoid):
                a = y.clip(self.eps, 1 - self.eps)
                a = torch.log(a / (1 - a))
            elif isinstance(self.activation, nn.Tanh):
                a = (y + 1) / 2
                a = a.clip(self.eps, 1 - self.eps)
                a = 0.5 * torch.log(a / (1 - a))
            elif isinstance(self.activation, nn.ReLU):
                y = y.clip(0, None)
                a = y  # y if y > 0 else 0 => already ReLU
            elif isinstance(self.activation, nn.Softmax):
                a = y.log()
            else:
                raise ValueError(f"Unsupported activation function: {self.activation}")
        pre_act = pre_act.view(pre_act.size(0), -1)
        a = a.view(a.size(0), -1)
        return self.bregman_div(pre_act, a)

    def bregman_div(self, x, y):
        if isinstance(self.activation, nn.Sigmoid):
            f = torch.sigmoid

            def F(x):
                return (x + (1 + (-x).exp()).log()).sum(dim=-1)

        elif isinstance(self.activation, nn.Tanh):
            f = torch.tanh

            def F(x):
                return x.cosh().log().sum(dim=-1)

        elif isinstance(self.activation, nn.ReLU):
            f = torch.nn.functional.relu

            def F(x):
                return 0.5 * (x * F.relu(x)).sum(dim=-1)

        elif isinstance(self.activation, nn.Softmax):
            f = functools.partial(torch.softmax, dim=-1)
            F = functools.partial(torch.logsumexp, dim=-1)
        else:
            raise ValueError(f"Unsupported activation function: {self.activation}")
        return bregman_div(x, y, f, F)


class LocopropTrainer:
    def __init__(
        self,
        model: nn.Sequential,
        loss_fn: typing.Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        optimizer_class=torch.optim.RMSprop,
        optimizer_hparams: typing.Dict = dict(
            lr=2e-5, eps=1e-6, momentum=0.999, alpha=0.9
        ),
        learning_rate: float = 10,
        local_iterations: int = 5,
        variant: typing.Literal["LocoPropS", "LocoPropM"] = "LocoPropM",
        momentum: float = 0.0,
        correction: float = 0.1,
    ):
        self.model = model
        self.loss_fn = loss_fn
        self.learning_rate = learning_rate
        self.local_iterations = local_iterations
        self.variant = variant
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
                hidden = layer.hidden(curr)
                hidden.requires_grad_(True)
                hidden.retain_grad()
                hiddens.append(hidden)
                curr = layer(hidden=hidden)
            else:
                hiddens.append(None)
                curr = layer(curr)

        hidden_loss = self.loss_fn(curr, ys)
        hidden_loss.backward()

        if len(self.grads) == 0:
            self.grads = [
                None if h is None else (1 - self.momentum) * h.grad for h in hiddens
            ]
        else:
            # naive momentum
            self.grads = [
                None if h is None else (1 - self.momentum) * h.grad + self.momentum * g
                for h, g in zip(hiddens, self.grads)
            ]

        for p in self.model.parameters():
            p.requires_grad = True

        for i, (opt, layer, grad) in enumerate(zip(self.opts, self.model, self.grads)):
            if opt is None:
                continue
            if not isinstance(layer, LocoLayer):
                raise ValueError(
                    f"Expected trainable layers to be instance of `LocoLayer` but found `{layer.__class__}`"
                )

            inp = inps[i]
            with torch.no_grad():
                a = layer.hidden(inp)
                y = layer.activation(a)
                pre_target = (a - self.learning_rate * grad).detach()
                post_target = (y - self.learning_rate * grad).detach()

            base_lr = opt.param_groups[0]["lr"]
            for j in range(self.local_iterations):
                opt.param_groups[0]["lr"] = base_lr * max(
                    1.0 - j / self.local_iterations, 0.25
                )
                opt.zero_grad()

                if self.variant == "LocoPropS":
                    pre_act = layer.hidden(inp)
                    loss = 0.5 * ((pre_act - pre_target) ** 2).mean(0).sum()
                else:
                    loss = layer.bregman_loss(inp, post_target).mean(0).sum()
                loss.backward()
                opt.step()
            opt.param_groups[0]["lr"] = base_lr

            # correct input of next layer
            if self.correction > 0 and i + 1 < len(inps):
                with torch.no_grad():
                    delta = layer(inp) - inps[i + 1]
                    norm = delta.norm() + 1e-5
                    inps[i + 1] += (
                        min(norm, self.correction * math.sqrt(delta.size(1)))
                        * delta
                        / norm
                    )
        return hidden_loss
