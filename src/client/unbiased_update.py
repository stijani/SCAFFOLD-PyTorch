from collections import OrderedDict
from typing import OrderedDict, List
from copy import deepcopy
import numpy as np
import torch
from rich.console import Console

from .base import ClientBase


class UnbiasedUpdateClient(ClientBase):
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
        momentum: float,
        logger: Console,
        gpu: int,
        beta: float,
        batch_size_unbiased_step: int
    ):
        super(UnbiasedUpdateClient, self).__init__(
            backbone,
            dataset,
            processed_data_dir,
            batch_size,
            valset_ratio,
            testset_ratio,
            local_epochs,
            local_lr,
            momentum,
            logger,
            gpu,
        )
        self.trainable_global_params: List[torch.Tensor] = None
        self.beta = beta
        self.batch_size_unbiased_step = batch_size_unbiased_step
        #self.mu = 1.0

    def unbiased_step(self, model):
        model.train()
        x, y = self.get_data_batch_unbiased()
        x /= 255.0
        logits = model(x)
        loss = self.criterion(logits, y)
        model.zero_grad()
        loss.backward()
        return model

    def _train(self):
        unbiased_model =  self.unbiased_step(deepcopy(self.model))
        unbiased_grads =  self.get_parameter_grads(unbiased_model)
        layer_names = self.model.state_dict().keys() 
        self.model.train()
        for _ in range(self.local_epochs): # this is a local step, not epoch
            x, y = self.get_data_batch()
            x /= 255.0
            logits = self.model(x)
            loss = self.criterion(logits, y)
            #self.optimizer.zero_grad()
            self.model.zero_grad()
            loss.backward()
            #self.optimizer.step()
            with torch.no_grad():
                new_params_dict = {}
                new_params_grads = {} # TODO: not used, remove
                for layer_name, layer_param in zip(layer_names, self.model.parameters()):
                    layer_grad = self.beta * unbiased_grads[layer_name] + (1 - self.beta) * layer_param.grad
                    new_layer_params = layer_param - layer_grad * self.local_lr
                    new_params_dict[layer_name] = new_layer_params
                    new_params_grads[layer_name] = layer_grad ######
            self.model.load_state_dict(new_params_dict)
        return (
            list(deepcopy(self.model.state_dict(keep_vars=True)).values()),
            len(self.trainset.dataset),
        )


    @torch.no_grad()
    def combine_weights(self, unbiased_weights, current_model):
        model_weights = current_model.parameters()
        final_weights_update = []
        for unbiased_weight, model_weight in zip(unbiased_weights, model_weights):
            final_weights_update.append((1 - self.beta) * model_weight + self.beta * unbiased_weight)
        return final_weights_update



    def get_data_batch_unbiased(self):
        batch_size = self.batch_size_unbiased_step
        # batch_size = len(self.trainset)
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

