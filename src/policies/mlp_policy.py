from typing import List

from torch import nn

from src.policies.utils import get_nonlinearity_factory


class MlpPolicy(nn.Module):
    def __init__(self, input_size: int, net_arch: List[int], output_size: int, nonlinearity: str):
        super().__init__()
        nonlinearity = get_nonlinearity_factory(nonlinearity)
        inp_sizes = [input_size] + net_arch
        oup_sizes = net_arch + [output_size]
        num_layers = len(inp_sizes)

        # Construct model
        layers = []
        for i, (inp, oup) in enumerate(zip(inp_sizes, oup_sizes)):
            layers.append(nn.Linear(inp, oup))
            if i == num_layers - 1:
                layers.append(nonlinearity())
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
