import random
from argparse import ArgumentParser, Namespace
from collections import OrderedDict
from typing import OrderedDict, Union

import numpy as np
import torch
# from path import Path
import os
import sys

sys.path.append("./")
from exp_configs import config_test as config

# PROJECT_DIR = Path(__file__).parent.parent.parent.abspath()
OUTPUT_DIR = config["exp_output_base"]
LOG_DIR = os.path.join(config["exp_output_base"],"logs")
TEMP_DIR = os.path.join(config["exp_output_base"], "temp")
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

global_epochs = 100
def get_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--global_epochs", type=int, default=global_epochs)
    parser.add_argument("--local_epochs", type=int, default=100, help="this is actually the number of localsteps, not epochs")
    parser.add_argument("--local_lr", type=float, default=1e-2)
    #parser.add_argument("--verbose_gap", type=int, default=2)
    parser.add_argument("--verbose_gap", type=int, default=100000)
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["mnist", "cifar10", "cifar100", "emnist", "fmnist"],
        default="cifar10",
    )
    parser.add_argument("--global_test_data_dir", type=str, default="/home/stijani/projects/dataset/cifar10")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--valset_ratio", type=float, default=0.1)
    parser.add_argument("--testset_ratio", type=float, default=0.1)
    parser.add_argument("--gpu", type=int, default=2)
    parser.add_argument("--log", type=int, default=0)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--client_num_per_round", type=int, default=10)
    parser.add_argument("--global_test_period", type=int, default=1, help="print the global test results")
    parser.add_argument("--save_period", type=int, default=global_epochs, help="save the aggregated weights after a certain number of comms round")
    return parser.parse_args()
