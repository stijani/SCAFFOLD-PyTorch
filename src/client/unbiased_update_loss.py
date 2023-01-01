from collections import OrderedDict
from typing import OrderedDict, List
from copy import deepcopy
import numpy as np
import torch
from rich.console import Console

from .base import ClientBase


class UnbiasedUpdateLossClient(ClientBase):
    def __init__(
        self,
        backbone: torch.nn.Module,
        logger: Console,
        args
    ):
        super(UnbiasedUpdateLossClient, self).__init__(
            backbone,
            logger,
            args
        )
        self.trainable_global_params: List[torch.Tensor] = None
        self.args = self.arg_dict

    def unbiased_step(self, model):
        optimizer = torch.optim.SGD(model.parameters(), lr=self.args["unbiased_update_lr"])
        model.train()
        x, y = self.get_data_batch_unbiased()
        x /= 255.0
        logits = model(x)
        loss = self.criterion(logits, y)
        #model.zero_grad()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return model

    def _train(self):
        unbiased_model =  self.unbiased_step(deepcopy(self.model))
        unbiased_weights =  unbiased_model.parameters()
        layer_names = self.model.state_dict().keys() 
        self.model.train()
        for _ in range(self.args["local_epochs"]):
            x, y = self.get_data_batch()
            x /= 255.0
            logits = self.model(x)
            loss = self.criterion(logits, y)
            proxy = 0.
            for p_g, p_l in zip(unbiased_weights, self.model.parameters()):
                proxy += torch.sum((p_g - p_l) * (p_g - p_l))
                loss = loss + proxy * self.args["mu"]
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        return (
            list(deepcopy(self.model.state_dict(keep_vars=True)).values()),
            len(self.trainset.dataset),
        )


    @torch.no_grad()
    def combine_weights(self, unbiased_weights, current_model):
        model_weights = current_model.parameters()
        final_weights_update = []
        for unbiased_weight, model_weight in zip(unbiased_weights, model_weights):
            final_weights_update.append((1 - self.args["beta"]) * model_weight + self.args["beta"] * unbiased_weight)
        return final_weights_update


    def get_data_batch_unbiased(self):
        batch_size = self.args["batch_size_unbiased_step"]
        indices = torch.from_numpy(
            np.random.choice(self.trainset.indices, batch_size)
        ).long()
        data, targets = self.trainset.dataset[indices]
        return data.to(self.device), targets.to(self.device)

    def get_parameter_grads(self, model_):
        """
        Extracts the gradients of the parameters in each of the layers of a model
        and returns them as a dictionary sharing keys with the model's state dictionary
        :param trained_model: a pytorch model
        :return: a python dictionary
        """
        grad_dict = {}
        layer_names = model_.state_dict().keys()
        with torch.no_grad():
            for name, layer_param in zip(layer_names, model_.parameters()):
                grad_dict[name] = deepcopy(layer_param.grad)
        return deepcopy(grad_dict)

