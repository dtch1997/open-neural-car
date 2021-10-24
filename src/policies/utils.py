import torch.nn as nn

nonlinearity_registry = {"relu": nn.ReLU, "tanh": nn.Tanh}


def get_nonlinearity_factory(name: str):
    if name not in nonlinearity_registry:
        raise ValueError(f"Invalid function name {name}")
    return nonlinearity_registry[name]
