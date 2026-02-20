import torch
from torch import nn

class Module(nn.Module):
    
    def __init__(self):
        super().__init__()

    def loss(self, y_hat, y):
        raise NotImplementedError

    def forward(self, X):
        raise NotImplementedError

    def training_step(self, batch):
        l = self.loss(self(*batch[:-1]), batch[-1])
        return l

    def validation_step(self, batch):
        l = self.loss(self(*batch[:-1]), batch[-1])
        return l

    def configure_optimizer(self):
        raise NotImplementedError