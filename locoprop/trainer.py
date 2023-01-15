from typing import Dict

import torch
from torch import nn

from locoprop.layer import LocoLayer


class LocoTrainer:
    def __init__(
        self,
        module: nn.Module,
        learning_rate: float = 10,
        iterations: int = 5,
        inner_opt_class: type = torch.optim.RMSprop,
        inner_opt_hparams: Dict = dict(lr=2e-5, eps=1e-6, momentum=0.999, alpha=0.9),
        outer_opt_class: type = torch.optim.SGD,
        outer_opt_hparams: Dict = dict(lr=1.0),
    ):
        self.module = module
        self.learning_rate = learning_rate
        self.iterations = iterations

        self.outer_optim = outer_opt_class(module.parameters(), **outer_opt_hparams)

        for m in self.module.modules():
            if isinstance(m, LocoLayer):
              opt = inner_opt_class(m.parameters(), **inner_opt_hparams)
              m.update_context(
                  learning_rate=learning_rate,
                  iterations=iterations,
                  optimizer=opt,
              )

    def zero_grad(self):
        self.outer_optim.zero_grad()

    def step(self):
        self.outer_optim.step()
