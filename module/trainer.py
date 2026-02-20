import torch
import matplotlib.pyplot as plt

class Trainer:
    
    def __init__(self, max_epochs, num_gpus = 0, gradient_clip_val = 0):
        self.max_epochs = max_epochs
        self.num_gpus = num_gpus
        self.gradient_clip_val = gradient_clip_val
        self.train_loss = []
        self.val_loss = []
        self.acc = []
        assert num_gpus == 0, 'No GPU support yet'

    def prepare_data(self, data):
        self.train_dataloader = data.train_dataloader()
        self.val_dataloader = data.val_dataloader()
        self.num_train_batch = len(self.train_dataloader)
        self.num_val_batch = len(self.val_dataloader) if self.val_dataloader is not None else 0

    def prepare_model(self, model):
        model.trainer = self
        self.model = model

    def fit(self, data, model):
        self.prepare_data(data)
        self.prepare_model(model)
        self.optim = self.model.configure_optimizer()
        for self.epochs in range(self.max_epochs):
            self.fit_epochs()

    def fit_epochs(self):
        self.model.train()
        train_batch_loss = 0
        num_train_epochs = 0
        for batch in self.train_dataloader:
            loss = self.model.training_step(self.prepare_batch(batch))
            train_batch_loss += loss.detach().numpy()
            num_train_epochs += 1
            self.optim.zero_grad()
            with torch.no_grad():
                loss.backward()
                if self.gradient_clip_val > 0:
                    self.clip_gradient(self.gradient_clip_val, self.model)
                self.optim.step()
        avg_train_loss = train_batch_loss / num_train_epochs
        self.train_loss.append(avg_train_loss)
        if self.val_dataloader is None:
            return
        self.model.eval()
        val_batch_loss = 0
        num_val_epochs = 0
        accuracy = 0
        for batch in self.val_dataloader:
            with torch.no_grad():
                loss, acc = self.model.validation_step(self.prepare_batch(batch))
                val_batch_loss += loss.detach().numpy()
                num_val_epochs += 1
                accuracy += acc
        avg_val_loss = val_batch_loss / num_val_epochs
        avg_accuracy = accuracy / num_val_epochs
        self.val_loss.append(avg_val_loss)
        self.acc.append(avg_accuracy)

    def prepare_batch(self, batch):
        return batch

    def plot(self):
        plt.plot(self.train_loss, label = 'train_loss')
        plt.plot(self.val_loss, label = 'val_loss')
        plt.plot(self.acc, label = 'accuracy')
        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.title('training progress')
        plt.legend()
        plt.show()