"""Lag-Llama Compatibility.

This file is used to create a dummy module hierarchy for the `gluonts.torch.modules.loss` module.
It is used in official Lag-llama notebook code because an error in `torch.load`.
https://colab.research.google.com/drive/1DRAzLUPxsd-0r8b-o4nlyFXrjw_ZajJJ?usp=sharing#scrollTo=BSNBysopSbXE
"""

import sys
from types import ModuleType


# Create dummy module hierarchy
def create_dummy_module(module_path):
    """
    Create a dummy module hierarchy for the given path.

    Returns the leaf module.
    """
    parts = module_path.split(".")
    current = ""
    parent = None

    for part in parts:
        current = current + "." + part if current else part
        if current not in sys.modules:
            module = ModuleType(current)
            sys.modules[current] = module
            if parent:
                setattr(sys.modules[parent], part, module)
        parent = current

    return sys.modules[module_path]


# Create dummy classes for the specific loss functions
class DistributionLoss:
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        return 0.0

    def __getattr__(self, name):
        return lambda *args, **kwargs: None


class NegativeLogLikelihood:
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        return 0.0

    def __getattr__(self, name):
        return lambda *args, **kwargs: None


def create_dummy_gluonts_torch_module():
    # Create the dummy gluonts module hierarchy
    gluonts_module = create_dummy_module("gluonts.torch.modules.loss")
    # Add the specific classes to the module
    gluonts_module.DistributionLoss = DistributionLoss
    gluonts_module.NegativeLogLikelihood = NegativeLogLikelihood
