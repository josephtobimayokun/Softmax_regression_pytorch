from data.fashionMNIST import FashionMNIST
from src.softmax_regression import SoftmaxRegression
from module.trainer import Trainer

data = FashionMNIST(batch_size = 256)
model = SoftmaxRegression(num_outputs = 10, lr = 0.1)
trainer = Trainer(max_epochs = 10)
trainer.fit(data, model)
trainer.plot()