from collections import OrderedDict
from copy import deepcopy
from typing import Dict, List, OrderedDict, Tuple

import torch
import numpy as np
from rich.console import Console
from torch.utils.data import Subset, DataLoader
import sys
sys.path.append("./")
from data.utils.util import get_dataset


class ClientBase:
    def __init__(
        self,
        backbone: torch.nn.Module,
        dataset: str,
        processed_data_dir,
        batch_size: int,
        valset_ratio: float,
        testset_ratio: float,
        local_epochs: int,
        local_lr: float,
        momentum: float,
        logger: Console,
        gpu: int,
    ):
        self.device = torch.device(
            "cuda" if gpu and torch.cuda.is_available() else "cpu"
        )
        self.client_id: int = None
        self.valset: Subset = None
        self.trainset: Subset = None
        self.testset: Subset = None
        self.model: torch.nn.Module = deepcopy(backbone).to(self.device)
        self.dataset = dataset
        self.processed_data_dir = processed_data_dir
        self.batch_size = batch_size
        self.valset_ratio = valset_ratio
        self.testset_ratio = testset_ratio
        self.local_epochs = local_epochs
        self.local_lr = local_lr
        self.momentum = momentum
        self.criterion = torch.nn.CrossEntropyLoss()
        self.logger = logger
        self.untrainable_params: Dict[str, Dict[str, torch.Tensor]] = {}
        self.optimizer: torch.optim.Optimizer = self.get_optimizer()


    def get_optimizer(self):
        if self.momentum:
            return torch.optim.SGD(self.model.parameters(), lr=self.local_lr, momentum=self.momentum)
        else:
            return torch.optim.SGD(self.model.parameters(), lr=local_lr)

    @torch.no_grad()
    def evaluate(self, data, bs=32): # TODO: add type anno for data
        size_ = len(data)
        self.model.eval()
        # size = 0
        loss = 0
        correct = 0
        dataloader = DataLoader(data, bs) # TODO: change eval bs argument
        for x, y in dataloader:
            ##################################
            # I couldn't the transormation to work 
            # so I couldn't apply mean/std normalization
            # more info in src/data/utils/dataset.py
            # Just dividing each pixel by 255 suffices
            x /= 255.0
            ##################################
            x, y = x.to(self.device), y.to(self.device)
            logits = self.model(x)
            loss += self.criterion(logits, y)
            pred = torch.softmax(logits, -1).argmax(-1)
            correct += (pred == y).int().sum()
        acc = (correct.float() / size_) * 100.0
        loss = loss / size_
        return loss.item(), acc.item()

    def train(
        self,
        client_id: int,
        model_params: OrderedDict[str, torch.Tensor],
        verbose=True,
    ) -> Tuple[List[torch.Tensor], int]:
        self.client_id = client_id
        self.set_parameters(model_params)
        self.get_client_local_dataset()
        res, stats = self._log_while_training(evaluate=False, verbose=verbose)()
        return res, stats

    def _train(self):
        self.model.train()
        for _ in range(self.local_epochs):
            x, y = self.get_data_batch()
            ##################################
            # I couldn't the transormation to work 
            # so I couldn't apply mean/std normalization
            # more info in src/data/utils/dataset.py
            # Just dividing each pixel by 255 suffices
            x /= 255.0
            ##################################
            logits = self.model(x)
            loss = self.criterion(logits, y)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        return (
            #list(self.model.state_dict(keep_vars=True).values()),
            list(deepcopy(self.model.state_dict(keep_vars=True)).values()),
            len(self.trainset.dataset),
        )
   
    def test(
        self, data, model_params: OrderedDict[str, torch.Tensor], bs=32): 
        # TODO: add type anno for data
        self.set_parameters(model_params)
        loss, acc = self.evaluate(data, bs)
        stats = {"loss": loss, "acc": acc}
        return stats


    def get_client_local_dataset(self):
        datasets = get_dataset(
            self.dataset,
            self.processed_data_dir,
            self.client_id,
            self.batch_size,
            self.valset_ratio,
            self.testset_ratio,
        )
        self.trainset = datasets["train"]
        self.valset = datasets["val"]
        self.testset = datasets["test"]

    def _log_while_training(self, evaluate=False, verbose=False):
        def _log_and_train(*args, **kwargs):
            loss_before = 0
            loss_after = 0
            acc_before = 0
            acc_after = 0
            if evaluate:
                # first evaluate with the recieved global weights prior to 
                # training using the client's valset
                loss_before, acc_before = self.evaluate(self.valset)

            # carryout local training
            res = self._train(*args, **kwargs)

            if evaluate:
                # carryout evaluation after local training - using the local weights
                loss_after, acc_after = self.evaluate(self.valset)

            if verbose:
                self.logger.log(
                    "client [{}]   [bold red]loss: {:.4f} -> {:.4f}    [bold blue]accuracy: {:.2f}% -> {:.2f}%".format(
                        self.client_id, loss_before, loss_after, acc_before, acc_after
                    )
                )

            stats = {
                "loss_before": loss_before,
                "loss_after": loss_after,
                "acc_before": acc_before,
                "acc_after": acc_after,
            }
            return res, stats

        return _log_and_train

    def set_parameters(self, model_params: OrderedDict):
        self.model.load_state_dict(model_params, strict=False)
        if self.client_id in self.untrainable_params.keys():
            self.model.load_state_dict(
                self.untrainable_params[self.client_id], strict=False
            )

    def get_data_batch(self):
        batch_size = (
            self.batch_size
            if self.batch_size > 0
            else int(len(self.trainset) / self.local_epochs)
        )
        indices = torch.from_numpy(
            np.random.choice(self.trainset.indices, batch_size)
        ).long()
        data, targets = self.trainset.dataset[indices]
        return data.to(self.device), targets.to(self.device)

