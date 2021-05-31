import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import MNIST, CIFAR10, FashionMNIST, ImageNet
from torchvision import transforms
from torch.utils.data.sampler import Sampler, SubsetRandomSampler

class SubSampler(Sampler):
    '''
    Customized sampler to subsample data 
    '''
    def __init__(self, idlist):
        self.idlist = idlist

    def __iter__(self):
        return (self.indices[i] for i in self.idlist)

    def __len__(self):
        return len(self.mask)
    
minst_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])


def minst_data_loader_test(batch_size):
    test_dataset = MNIST('./data', transform=minst_transform, train=False, download=True)
    return DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


def minst_data_loader_train(batch_size):
    train_dataset = MNIST('./data', transform=minst_transform, download=True)
    return DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


def fasion_minst_data_loader_test(batch_size):
    test_dataset = FashionMNIST('./data', transform=minst_transform, train=False, download=True)
    return DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


def fasion_minst_data_loader_train(batch_size):
    train_dataset = FashionMNIST('./data', transform=minst_transform, download=True)
    return DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


cifar10_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])


def cifar10_dataset_test(loc='./data'):
    return CIFAR10(loc, transform=cifar10_transform, train=False, download=True)


def cifar10_dataset_train(loc='./data'):
    return CIFAR10(loc, transform=cifar10_transform, train=True, download=True)


def cifar10_data_loader_test(batch_size, shuffle=True, loc='./data'):
    '''
    Create dataloader
    '''
    test_dataset = CIFAR10(loc, transform=cifar10_transform, train=False, download=True)
    return DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle)


def cifar10_data_loader_train(batch_size, shuffle=True, loc='./data'):
    '''
    Create dataloader
    '''
    train_dataset = CIFAR10(loc, transform=cifar10_transform, download=True)
    return DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)


# One small remark: apparently sampler is not compatible with shuffle, so in order to achieve the same result one can do: torch.utils.data.DataLoader(trainset, batch_size=4, sampler=SubsetRandomSampler(np.where(mask)[0]),shuffle=False, num_workers=2)
def cifar10_data_loader_test_subsample(batch_size, subsample_id, shuffle=False, loc='./data'):
    '''
    Create dataloader with only certain indices
    '''
    test_dataset = CIFAR10(loc, transform=cifar10_transform, train=False, download=True)
    sampler = SubSampler(subsample_id)
    if shuffle == False:
        return DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle, sampler=sampler)
    else:
        return DataLoader(test_dataset, batch_size=batch_size, sampler=SubsetRandomSampler(subsample_id),shuffle=False)


def cifar10_data_loader_train_subsample(batch_size, subsample_id, shuffle=False, loc='./data'):
    '''
    Create dataloader with only certain indices
    '''
    train_dataset = CIFAR10(loc, transform=cifar10_transform, download=True)
    sampler = SubSampler(subsample_id)
    if shuffle == False:
        return DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, sampler=sampler)
    else:
        return DataLoader(train_dataset, batch_size=batch_size, sampler=SubsetRandomSampler(subsample_id),shuffle=False)
        
