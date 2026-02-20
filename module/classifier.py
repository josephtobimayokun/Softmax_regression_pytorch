import torch
from module.module import Module

class Classifier(Module):
    
    def __init__(self):
        super().__init__()

    def accuracy(self, y_hat, y, averaged = True):
        y_hat = y_hat.reshape((-1, y_hat.shape[-1]))
        preds = y_hat.argmax(axis = 1).type(y.dtype)
        compare = (preds == y.reshape((-1))).type(torch.float32)
        return compare.mean() if averaged else compare

    def validation_step(self, batch):
        y_hat = self(*batch[:-1])
        loss = self.loss(y_hat, batch[-1])
        acc = self.accuracy(y_hat, batch[-1])
        return loss, acc

    def configure_optimizer(self):
        return torch.optim.SGD(self.parameters(), lr = self.lr)
