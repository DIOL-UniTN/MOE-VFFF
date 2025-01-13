import torch
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from torch.utils.data import Subset, DataLoader

class MNISTLoader:
    def __init__(self, batch_size: int, normalize, debug: bool):
        transform = transforms.Compose([transforms.ToTensor(), normalize])

        whole_tset = MNISTSubset(train=True, transform=transform, debug=debug)
        testset = MNISTSubset(train=False, transform=transform, debug=debug)

        whole_range = torch.randperm(len(whole_tset))

        val_len = int(len(whole_range)*0.1)
        train_len = len(whole_range)-val_len

        trainset = Subset(whole_tset, whole_range[:train_len])
        validset = Subset(whole_tset, whole_range[val_len:])

        num_workers = 0 if debug else 8
        self.train = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        self.valid = DataLoader(validset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        self.test = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        self.batch_size = batch_size

        self.in_chan = 1
        self.in_size = (28, 28)
        self.out_dim = 10

class MNISTSubset(MNIST):
    def __init__(self, train: bool, transform, debug: bool):
        super().__init__(root='./data', train=train, download=True, transform=transform)
        self.debug = debug

    def __len__(self):
        if self.debug:
            return 2
        return len(self.data)
