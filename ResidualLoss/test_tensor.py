import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST, FashionMNIST
import torch.nn.functional as F

import numpy as np
from NeuronCoverage.model import MINST_3, MINST_8
from NeuronCoverage.neuron_coverage_model import NeuronCoverageReLUModel
from torch.utils.data import DataLoader, Dataset
import os
from torchvision import transforms

img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

dataset = FashionMNIST('../data', transform=img_transform, train=False, download=True)


class TestData(Dataset):  # 继承Dataset
    def __init__(self):
        self.hidden_layer_dict = dict()

    def __len__(self):  # 返回整个数据集的大小
        return dataset.__len__()

    def __getitem__(self, index):  # 根据索引index返回dataset[index]
        if index % 2 == 0:
            result = {"a": torch.Tensor([[1, 2], [3, 4]]), "b": torch.Tensor([5, 6])}
        else:
            result = {"a": torch.Tensor([[1, 2], [3, 4]])}
        return dataset.data[index], dataset.targets[index], result


dataloader = DataLoader(TestData(), batch_size=1000, shuffle=False)

for idx, (data, target, test) in enumerate(dataloader):
    print(idx)
    print(test)
