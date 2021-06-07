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
        return iter(self.idlist)

    def __len__(self):
        return len(self.idlist)
    
    
def data_loader(dataset_name, batch_size, train=True, subsample_id=None, shuffle=False):
    '''
    Dataloader 
    '''
    # define dataset & transformation
    if dataset_name == 'MNIST':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        dataset = MNIST('./data', transform=transform, train=train, download=True)
        
    elif dataset_name == 'FashionMNIST':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        dataset = FashionMNIST('./data', transform=transform, train=train, download=True)
        
    elif dataset_name == 'CIFAR10':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        
        dataset = CIFAR10('./data', transform=transform, train=train, download=True)
    else: 
        raise NotImplementError
    
    # Load subset of data or full data
    if subsample_id is None:
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    else:
        sampler = SubSampler(subsample_id)
        if shuffle == False:
            return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, sampler=sampler, drop_last=False)
        else:
            # SubSampler does not support shuffle by default
            # So we need to use SubsetRandomSampler in order to get re-arranged batch
            return DataLoader(dataset, batch_size=batch_size, sampler=SubsetRandomSampler(subsample_id), 
                              shuffle=shuffle, drop_last=False)

