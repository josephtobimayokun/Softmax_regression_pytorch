import torch
import torchvision
from torchvision import datasets, transforms
from module.data_module import DataModule

class FashionMNIST(DataModule):
    
    def __init__(self, batch_size = 64, resize = (28, 28)):
        super().__init__()
        self.batch_size = batch_size
        self.resize = resize
        trans = transforms.Compose([transforms.Resize(resize), transforms.ToTensor()])
        self.train = datasets.FashionMNIST(root = self.root, train = True, transform = trans, download = True) 
        self.val = datasets.FashionMNIST(root = self.root, train = False, transform = trans, download = True)

    def get_dataloader(self, train):
        data = self.train if train else self.val
        return torch.utils.data.DataLoader(data, self.batch_size, shuffle = train, num_workers = 4)

    def text_labels(self, indices):
        labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat','sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
        return [labels[int(i)] for i in indices]￼Enter
