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
        logger: Console,
        args
    ):
        super(FedProxClient, self).__init__(
            backbone,
            logger,
            args
        )
        self.trainable_global_params: List[torch.Tensor] = None
        self.args = self.arg_dict

    def _train(self):
        self.model.train()
        for _ in range(self.args["local_epochs"]):
            x, y = self.get_data_batch()
            x /= 255.0
            logits = self.model(x)
            loss = self.criterion(logits, y)
            proxy = 0
            for p_g, p_l in zip(self.trainable_global_params, self.model.parameters()):
                proxy = proxy + torch.sum((p_g - p_l) * (p_g - p_l))
            #proxy = (self.args["mu"] / 2) * proxy
            proxy = (self.args["mu"] / 2) * proxy
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
        self.trainable_global_params = list(
            filter(lambda p: p.requires_grad, deepcopy(self.model).parameters()) 
        )