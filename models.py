from torch import nn
from torch.nn import Sequential, ReLU, Tanh, Linear


class FashionMLP(nn.Module):
    def __init__(self, in_size, out_size):
        super(FashionMLP, self).__init__()

        self.mlp = Sequential(
            Linear(in_size, 10), ReLU(),
            Linear(10, out_size)
        )

    def forward(self, x):
        return self.mlp(x)