import torch

class DataModule:
    
    def __init__(self, root = 'C:/Users/USER/Desktop/data', num_workers = 4):
        self.root = root
        self.num_workers = num_workers

    def get_dataloader(self, train):
        raise NotImplementedError

    def train_dataloader(self):
        return self.get_dataloader(train = True)

    def val_dataloader(self):
        return self.get_dataloader(train = False)

    def tensor_dataloader(self, tensor, indices, train):
        tensor = tuple(a[indices] for a in tensor)
        datasets = torch.utils.data.TensorDataset(*tensor)
        return torch.utils.data.DataLoader(datasets, self.batch_size, shuffle = train)