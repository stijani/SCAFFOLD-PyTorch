import json
import math
import os
import pickle
from typing import Dict, List, Tuple, Union

from path import Path
from torch.utils.data import Subset, random_split

_CURRENT_DIR = Path(__file__).parent.abspath()
_ARGS_DICT = json.load(open(_CURRENT_DIR.parent / "args.json", "r"))


def get_dataset(
    dataset: str, client_id: int, batch_size=32, valset_ratio=0.1, testset_ratio=0.1,
) -> Dict[str, Subset]:
    client_num_in_each_pickles = _ARGS_DICT["client_num_in_each_pickles"]
    pickles_dir = _CURRENT_DIR.parent / dataset / "pickles"
    if os.path.isdir(pickles_dir) is False:
        raise RuntimeError("Please preprocess and create pickles first.")

    pickle_path = (
        pickles_dir / f"{math.floor(client_id / client_num_in_each_pickles)}.pkl"
    )
    with open(pickle_path, "rb") as f:
        subset = pickle.load(f)
    client_dataset = subset[client_id % client_num_in_each_pickles]
    val_samples_num = int(len(client_dataset) * valset_ratio)
    test_samples_num = int(len(client_dataset) * testset_ratio)
    train_samples_num = len(client_dataset) - val_samples_num - test_samples_num
    trainset, valset, testset = random_split(
        client_dataset, [train_samples_num, val_samples_num, test_samples_num]
    )
    return {"train": trainset, "val": valset, "test": testset}


def get_client_id_indices(
    dataset,
) -> Union[Tuple[List[int], List[int], int], Tuple[List[int], int]]:
    pickles_dir = _CURRENT_DIR.parent / dataset / "pickles"
    with open(pickles_dir / "seperation.pkl", "rb") as f:
        seperation = pickle.load(f)
    if _ARGS_DICT["type"] == "user":
        return seperation["train"], seperation["test"], seperation["total"]
    else:  # NOTE: "sample"
        return seperation["id"], seperation["total"]
