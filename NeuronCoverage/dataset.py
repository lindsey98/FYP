from torch.utils.data import DataLoader
from torchvision.datasets import MNIST, CIFAR10, FashionMNIST
from torchvision import transforms

img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])


def minst_data_loader_test(batch_size):
    test_dataset = MNIST('./data', transform=img_transform, train=False)
    return DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


def minst_data_loader_train(batch_size):
    train_dataset = MNIST('./data', transform=img_transform)
    return DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


def fasion_minst_data_loader_test(batch_size):
    test_dataset = FashionMNIST('./data', transform=img_transform, train=False, download=True)
    return DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


def fasion_minst_data_loader_train(batch_size):
    train_dataset = FashionMNIST('./data', transform=img_transform)
    return DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


def cifar10_data_loader_test(batch_size):
    test_dataset = CIFAR10('./data', transform=img_transform, train=False, download=True)
    return DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


def cifar10_data_loader_train(batch_size):
    train_dataset = CIFAR10('./data', transform=img_transform, download=True)
    return DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
