import random
from argparse import ArgumentParser, Namespace
from collections import OrderedDict
from typing import OrderedDict, Union

import numpy as np
import torch
# from path import Path
import os
import sys

# PROJECT_DIR = Path(__file__).parent.parent.parent.abspath()
# OUTPUT_DIR = config["exp_output_base"]
# LOG_DIR = os.path.join(config["exp_output_base"],"logs")
# TEMP_DIR = os.path.join(config["exp_output_base"], "temp")
# DATA_DIR = "~/projects/dataset/fl_dataset/cifar10"


def fix_random_seed(seed: int) -> None:
    torch.cuda.empty_cache()
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def clone_parameters(
    src: Union[OrderedDict[str, torch.Tensor], torch.nn.Module]
) -> OrderedDict[str, torch.Tensor]:
    if isinstance(src, OrderedDict):
        return OrderedDict(
            {
                name: param.clone().detach().requires_grad_(param.requires_grad)
                for name, param in src.items()
            }
        )
    if isinstance(src, torch.nn.Module):
        return OrderedDict(
            {
                name: param.clone().detach().requires_grad_(param.requires_grad)
                for name, param in src.state_dict(keep_vars=True).items()
            }
        )
