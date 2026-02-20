import torch
from torch import nn
from torch.nn import functional as F
from module.classifier import Classifier

class SoftmaxRegression(Classifier):
    
    def __init__(self, num_outputs, lr):
        super().__init__()
        self.num_outputs = num_outputs
        self.lr = lr
        self.net = nn.Sequential(nn.Flatten(), nn.LazyLinear(num_outputs))

    def forward(self, X):
        return self.net(X)

    def loss(self, y_hat, y, averaged = True):
        y_hat = y_hat.reshape((-1, y_hat.shape[-1]))
        y = y.reshape((-1,))
        return F.cross_entropy(y_hat, y, reduction = 'mean' if averaged else 'none')￼Enter
