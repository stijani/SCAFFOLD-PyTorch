from collections import OrderedDict
from typing import OrderedDict, List
from copy import deepcopy

import torch
import numpy as np
from rich.console import Console

from .base import ClientBase


class FedProxClient(ClientBase):
    def __init__(
        self,
        backbone: torch.nn.Module,
        dataset: str,
        processed_data_dir: str,
        batch_size: int,
        valset_ratio: float,
        testset_ratio: float,
        local_epochs: int,
        local_lr: float,
        lr_schedule_step: int,
        lr_schedule_rate: float,
        momentum: float,
        mu: float,
        logger: Console,
        gpu: int,
    ):
        super(FedProxClient, self).__init__(
            backbone,
            dataset,
            processed_data_dir,
            batch_size,
            valset_ratio,
            testset_ratio,
            local_epochs,
            local_lr,
            lr_schedule_step,
            lr_schedule_rate,
            momentum,
            logger,
            gpu,
        )
        self.trainable_global_params: List[torch.Tensor] = None
        self.mu = mu
        self.gpu = gpu

    def _train(self):
        self.model.train()
        for _ in range(self.local_epochs):
            x, y = self.get_data_batch()
            x /= 255.0
            logits = self.model(x)
            loss = self.criterion(logits, y)
            proxy = 0
            for p_g, p_l in zip(self.trainable_global_params, self.model.parameters()):
                proxy = proxy + torch.sum((p_g - p_l) * (p_g - p_l))
            #proxy = (self.mu / 2) * proxy
            proxy = (self.mu / 2) * proxy
            loss = loss + proxy
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        #print(proxy)
        #print(torch.sum(self.trainable_global_params[2]).item(), torch.sum(list(self.model.parameters())[2]).item())
        return (
            list(self.model.state_dict(keep_vars=True).values()),
            len(self.trainset.dataset),
        )

    def set_parameters(
        self, model_params: OrderedDict[str, torch.Tensor],
    ):
        super().set_parameters(model_params) # set the local model to the global params prior to a local training
        # self.trainable_global_params = list(
        #     filter(lambda p: p.requires_grad, model_params.values())
        # )
        ##################################################
        # We filter out all the non-trainable parameters from 
        # the global weights, those those won't contribute to the proxy loss
        # Since all the layers in the global params have already 
        # been marked non-trainable as a result of using the 
        #@torch.no_grad decorator in the aggregation function. The 
        # way we can identify the training layers is upon loading 
        # the params to a local model prior to local training
        ##################################################
        self.trainable_global_params = list(
            filter(lambda p: p.requires_grad, deepcopy(self.model).parameters()) 
        )
        # print(self.trainable_global_params)


