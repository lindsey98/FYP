from torch.utils.data import DataLoader
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


def cifar10_data_loader_test(batch_size):
    test_dataset = CIFAR10('../data', transform=cifar10_transform, train=False)
    return DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


def cifar10_data_loader_train(batch_size):
    train_dataset = CIFAR10('../data', transform=cifar10_transform)
    return DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


image_net_normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


def image_net_data_loader_test(batch_size):
    pre_processing = transforms.Compose([
        transforms.Scale(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        image_net_normalize,
    ])
    test_dataset = ImageNet('../data', transform=pre_processing, train=False, download=True)
    return DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


def image_net_data_loader_train(batch_size):
    pre_processing = transforms.Compose([
        transforms.RandomSizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        image_net_normalize,
    ])
    train_dataset = ImageNet('../data', transform=pre_processing, download=True)
    return DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
