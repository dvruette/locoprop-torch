# LocoProp Torch

Implementation of the paper "LocoProp: Enhancing BackProp via Local Loss Optimization" in PyTorch.

Paper: https://proceedings.mlr.press/v151/amid22a/amid22a.pdf

Official code: https://github.com/google-research/google-research/blob/master/locoprop/locoprop_training.ipynb

## Installation

```
pip install locoprop
```

## Usage

```python
from locoprop import LocoLayer LocopropTrainer

# model needs to be instance of nn.Sequential
# each trainable layer needs to be instance of LocoLayer
# Example: deep auto-encoder
model = nn.Sequential(
    LocoLayer(nn.Linear(28*28, 1000), nn.Tanh()),
    LocoLayer(nn.Linear(1000, 500), nn.Tanh()),
    LocoLayer(nn.Linear(500, 250), nn.Tanh()),
    LocoLayer(nn.Linear(250, 30), nn.Tanh()),
    LocoLayer(nn.Linear(30, 250), nn.Tanh()),
    LocoLayer(nn.Linear(250, 500), nn.Tanh()),
    LocoLayer(nn.Linear(500, 1000), nn.Tanh()),
    LocoLayer(nn.Linear(1000, 28*28), nn.Sigmoid(), implicit=True),  # implicit means the activation only is applied during local optimization
)

def loss_fn(logits, labels):
    ...

trainer = LocopropTrainer(model, loss_fn)

dl = get_dataloader()

for x, y in dl:
    trainer.step(x, y)
```