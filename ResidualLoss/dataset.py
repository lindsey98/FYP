import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import MNIST, CIFAR10, FashionMNIST, ImageNet
from torchvision import transforms

minst_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])


def minst_data_loader_test(batch_size):
    test_dataset = MNIST('../data', transform=minst_transform, train=False)
    return DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


def minst_data_loader_train(batch_size):
    train_dataset = MNIST('../data', transform=minst_transform)
    return DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


def fasion_minst_data_loader_test(batch_size):
    test_dataset = FashionMNIST('../data', transform=minst_transform, train=False, download=True)
    return DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


def fasion_minst_data_loader_train(batch_size):
    train_dataset = FashionMNIST('../data', transform=minst_transform)
    return DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


cifar10_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])


def cifar10_dataset_test(loc='../data'):
    return CIFAR10(loc, transform=cifar10_transform, train=False)


def cifar10_dataset_train(loc='../data'):
    return CIFAR10(loc, transform=cifar10_transform, train=True)


def cifar10_data_loader_test(batch_size, shuffle=True, loc='../data'):
    test_dataset = CIFAR10(loc, transform=cifar10_transform, train=False)
    return DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle)


def cifar10_data_loader_train(batch_size, shuffle=True, loc='../data'):
    train_dataset = CIFAR10(loc, transform=cifar10_transform)
    return DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)


class L2Dataset(Dataset):
    def __init__(self, dataset, feature_shape=(8,)):
        self.dataset = dataset
        self.length = len(dataset)
        self.feature_shape = feature_shape

        self.l2_loss = torch.tensor([float('inf')] * self.length)
        self.l2_ref = torch.zeros((self.length, ) + feature_shape)

    def __getitem__(self, index):
        return self.dataset.__getitem__(index) + (self.l2_ref[index], index)

    def __len__(self):
        return self.length

    def reset(self):
        self.l2_loss = torch.tensor([float('inf')] * self.length)
        self.l2_ref = torch.zeros((self.length,) + self.feature_shape)
