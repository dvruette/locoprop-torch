from torch import nn

class LocoLayer(nn.Module):
    def __init__(self, module, activation, implicit=False):
        super().__init__()
        self.module = module
        self.activation = activation
        self.implicit = implicit

    def forward(self, x=None, hidden=None):
        if x is None and hidden is None:
            raise ValueError("No argument was given. Provide either input or hidden state.")

        if hidden is None:
            hidden = self.module(x)

        if self.implicit:
            return hidden
        return self.activation(hidden)
