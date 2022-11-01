from collections import OrderedDict
from typing import OrderedDict, List

import torch
from rich.console import Console

from .base import ClientBase


class Paper2Client(ClientBase):
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
        beta: float,
    ):
        super(FedProxClient, self).__init__(
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
        self.trainable_global_params: List[torch.Tensor] = None
        self.beta = beta
        #self.mu = 1.0

    def unbiased_step(self, model):
        model.train()
        x, y = self.get_data_batch()
        logits = model(x)
        loss = self.criterion(logits, y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return (
            deepcopy(model.parameters())
        )

    def _train(self):
        unbiased_weights = self.unbiased_step(deepcopy(self.model))
        self.model.train()
        for _ in range(self.local_epochs): # this is a local step, not epoch
            x, y = self.get_data_batch()
            logits = self.model(x)
            loss = self.criterion(logits, y)
            # proxy = 0
            # for p_g, p_l in zip(self.trainable_global_params, self.model.parameters()):
            #     proxy = proxy + torch.sum((p_g - p_l) * (p_g - p_l))
            # proxy = (self.mu / 2) * proxy
            # loss = loss + proxy
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        final_weights_update = []
        for unbiased_weight, model_weight in zip(unbiased_weights, self.model.parameters()):
            final_weights_update.append((1 - self.beta) * model_weight + self.beta * unbiased_weight)
        return (
            final_weights_update,
            len(self.trainset.dataset),
        )

    def set_parameters(
        self, model_params: OrderedDict[str, torch.Tensor],
    ):
        super().set_parameters(model_params)
        self.trainable_global_params = list(
            filter(lambda p: p.requires_grad, model_params.values())
        )

