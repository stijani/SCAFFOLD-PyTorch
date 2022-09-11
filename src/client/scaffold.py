from collections import OrderedDict
from copy import deepcopy
from typing import Dict, List, OrderedDict

import torch
from rich.console import Console

from .base import ClientBase


class SCAFFOLDClient(ClientBase):
    def __init__(
        self,
        backbone: torch.nn.Module,
        dataset: str,
        batch_size: int,
        valset_ratio: float,
        testset_ratio: float,
        local_epochs: int,
        local_lr: float,
        logger: Console,
        gpu: int,
    ):
        super(SCAFFOLDClient, self).__init__(
            backbone,
            dataset,
            batch_size,
            valset_ratio,
            testset_ratio,
            local_epochs,
            local_lr,
            logger,
            gpu,
        )
        self.c_local: Dict[List[torch.Tensor]] = {}

    def train(
        self,
        client_id: int,
        model_params: OrderedDict[str, torch.Tensor],
        c_global,
        verbose=True,
    ):
        self.client_id = client_id
        self.set_parameters(model_params)
        self.get_client_local_dataset()
        _, stats = self._log_while_training(evaluate=True, verbose=verbose)()

        # update local control variate
        with torch.no_grad():
            trainable_parameters = list(
                filter(lambda p: p.requires_grad, model_params.values())
            )
            if self.client_id not in self.c_local:
                self.c_local[self.client_id] = [
                    torch.zeros_like(param, device=self.device)
                    for param in self.model.parameters()
                    if param.requires_grad
                ]

            y_delta = [torch.zeros_like(param) for param in self.model.parameters()]
            # c_+ and c_delta both have the same shape as y_delta
            c_plus = deepcopy(y_delta)
            c_delta = deepcopy(y_delta)

            # compute y_delta (difference of model before and after training)
            for y_del, param_l, param_g in zip(
                y_delta, self.model.parameters(), trainable_parameters
            ):
                y_del.data = param_l - param_g

            # compute c_plus
            coef = 1 / (self.local_epochs * self.local_lr)
            for c_n, c_l, c_g, diff in zip(
                c_plus, self.c_local[self.client_id], c_global, y_delta
            ):
                c_n.data = c_l - c_g - diff / coef

            # compute c_delta
            for c_d, c_n, c_l in zip(c_delta, c_plus, self.c_local[self.client_id]):
                c_d.data = c_n - c_l

            self.c_local[self.client_id] = c_plus

        if self.client_id not in self.untrainable_params.keys():
            self.untrainable_params[self.client_id] = {}
        for name, param in self.model.state_dict(keep_vars=True).items():
            if not param.requires_grad:
                self.untrainable_params[self.client_id][name] = param.clone()

        return (y_delta, c_delta), stats

    def _train(self):
        self.model.train()
        for _ in range(self.local_epochs):
            x, y = self.get_data_batch()
            x, y = x.to(self.device), y.to(self.device)
            logits = self.model(x)
            loss = self.criterion(logits, y)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def test(
        self, client_id: int, model_params: OrderedDict[str, torch.Tensor],
    ):
        self.client_id = client_id
        self.set_parameters(model_params)
        self.get_client_local_dataset()
        self.model.to(self.device)
        loss, acc = self.evaluate()
        self.model.cpu()
        stats = {"loss": loss, "acc": acc}
        return stats
