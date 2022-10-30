import time
# from path import Path
from pathlib import Path

# _CURRENT_DIR = Path(__file__).parent.abspath()
import sys
sys.path.append("./data/utils")

# sys.path.append(_CURRENT_DIR)
# sys.path.append(_CURRENT_DIR.parent)
import json
import os
import pickle
import random
from argparse import ArgumentParser

import numpy as np
import torch
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100, EMNIST, MNIST, FashionMNIST

from constants import MEAN, STD
# from partition import dirichlet_distribution, randomly_assign_classes
from partition.assign_classes import randomly_assign_classes
from partition.dirichlet import dirichlet_distribution 
from dataset import CIFARDataset, MNISTDataset

DATASET = {
    "mnist": (MNIST, MNISTDataset),
    "emnist": (EMNIST, MNISTDataset),
    "fmnist": (FashionMNIST, MNISTDataset),
    "cifar10": (CIFAR10, CIFARDataset),
    "cifar100": (CIFAR100, CIFARDataset),
}


def main(args):
    # _DATASET_ROOT = (
    #     Path(args.root).abspath() / args.dataset
    #     if args.root is not None
    #     else _CURRENT_DIR.parent / args.dataset
    # )
    _DATASET_ROOT = os.path.join(args.root, args.dataset)

    # _PICKLES_DIR = _CURRENT_DIR.parent / args.dataset / "pickles"
    _PICKLES_DIR = os.path.join(_DATASET_ROOT, "pickles")

    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    classes_map = None
    transform = transforms.Compose(
        [transforms.Normalize(MEAN[args.dataset], STD[args.dataset]),]
    )
    target_transform = None

    if not os.path.isdir(_DATASET_ROOT):
        os.mkdir(_DATASET_ROOT)
    if os.path.isdir(_PICKLES_DIR):
        os.system(f"rm -rf {_PICKLES_DIR}")
    os.system(f"mkdir -p {_PICKLES_DIR}")

    client_num_in_total = args.client_num_in_total
    client_num_in_total = args.client_num_in_total
    ori_dataset, target_dataset = DATASET[args.dataset]
    if args.dataset == "emnist":
        trainset = ori_dataset(
            _DATASET_ROOT,
            train=True,
            download=True,
            split=args.emnist_split,
            transform=transforms.ToTensor(),
        )
        testset = ori_dataset(
            _DATASET_ROOT,
            train=False,
            split=args.emnist_split,
            transform=transforms.ToTensor(),
        )
    else:
        trainset = ori_dataset(_DATASET_ROOT, train=True, download=True,)
        testset = ori_dataset(_DATASET_ROOT, train=False,)
    concat_datasets = [trainset, testset]
    if args.alpha > 0:  # NOTE: Dirichlet(alpha)
        all_datasets, stats = dirichlet_distribution(
            ori_dataset=concat_datasets,
            target_dataset=target_dataset,
            num_clients=client_num_in_total,
            alpha=args.alpha,
            transform=transform,
            target_transform=target_transform,
        )
    else:  # NOTE: sort and partition
        # classes = len(ori_dataset.classes) if args.classes <= 0 else args.classes
        classes = args.num_classes_dataset if args.classes <= 0 else args.classes
        all_datasets, stats = randomly_assign_classes(
            ori_datasets=concat_datasets,
            target_dataset=target_dataset,
            num_clients=client_num_in_total,
            num_classes=classes,
            transform=transform,
            target_transform=target_transform,
        )

    for subset_id, client_id in enumerate(
        range(0, len(all_datasets), args.client_num_in_each_pickles)
    ):
        subset = all_datasets[client_id : client_id + args.client_num_in_each_pickles]
        # with open(_PICKLES_DIR / str(subset_id) + ".pkl", "wb") as f:
        with open(os.path.join(_PICKLES_DIR, f"{str(subset_id)}.pkl"), "wb") as f:
            pickle.dump(subset, f)

    # save stats
    if args.type == "user":
        train_clients_num = int(client_num_in_total * args.fraction)
        clients_4_train = [i for i in range(train_clients_num)]
        clients_4_test = [i for i in range(train_clients_num, client_num_in_total)]

        with open(os.path.join(_PICKLES_DIR, "seperation.pkl"), "wb") as f:
            pickle.dump(
                {
                    "train": clients_4_train,
                    "test": clients_4_test,
                    "total": client_num_in_total,
                },
                f,
            )

        train_clients_stats = dict(
            zip(clients_4_train, list(stats.values())[:train_clients_num])
        )
        test_clients_stats = dict(
            zip(clients_4_test, list(stats.values())[train_clients_num:],)
        )

        # with open(_CURRENT_DIR.parent / args.dataset / "all_stats.json", "w") as f:
        with open(os.path.join(_DATASET_ROOT, "all_stats.json"), "w") as f:
            json.dump({"train": train_clients_stats, "test": test_clients_stats}, f)

    else:  # NOTE: "sample"  save stats
        client_id_indices = [i for i in range(client_num_in_total)]
        # with open(_PICKLES_DIR / "seperation.pkl", "wb") as f:
        with open(os.path.join(_PICKLES_DIR, "seperation.pkl"), "wb") as f:
            pickle.dump(
                {"id": client_id_indices, "total": client_num_in_total,}, f,
            )
        # with open(_CURRENT_DIR.parent / args.dataset / "all_stats.json", "w") as f:
        with open(os.path.join(_DATASET_ROOT, "all_stats.json"), "w") as f:
            json.dump(stats, f)

    # args.root = (
    #     Path(args.root).abspath()
    #     if str(_DATASET_ROOT) != str(_CURRENT_DIR.parent / args.dataset)
    #     else None
    # )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["mnist", "cifar10", "cifar100", "emnist", "fmnist",],
        default="mnist",
    )
    ################# Dirichlet distribution only #################
    parser.add_argument(
        "--alpha",
        type=float,
        default=0,
        help="Only for controling data hetero degree while performing Dirichlet partition.",
    )
    ###############################################################
    parser.add_argument("--client_num_in_total", type=int, default=10)
    parser.add_argument(
        "--fraction", type=float, default=0.9, help="Propotion of train clients"
    )
    parser.add_argument(
        "--classes",
        type=int,
        default=-1,
        help="Num of classes that one client's data belong to.",
    )
    parser.add_argument(
        "--num_classes_dataset",
        type=int,
        default=10,
        help="number of categories in the entire data set.",
    )
    parser.add_argument("--seed", type=int, default=int(time.time()))

    ################# For EMNIST only #####################
    parser.add_argument(
        "--emnist_split",
        type=str,
        choices=["byclass", "bymerge", "letters", "balanced", "digits", "mnist"],
        default="byclass",
    )
    #######################################################
    parser.add_argument(
        "--type", type=str, choices=["sample", "user"], default="sample"
    )
    parser.add_argument("--client_num_in_each_pickles", type=int, default=10)
    parser.add_argument("--root", type=str, default=None)
    args = parser.parse_args()
    main(args)
    args_dict = dict(args._get_kwargs())
    # with open(_CURRENT_DIR.parent / "args.json", "w") as f:
    with open(os.path.join(os.path.join(args.root, args.dataset), "args.json"), "w") as f:
        json.dump(args_dict, f)
