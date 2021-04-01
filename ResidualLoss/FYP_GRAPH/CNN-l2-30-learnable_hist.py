import torch
import matplotlib.pyplot as plt
import PIL
import os
import numpy as np
from matplotlib import ticker
from torchvision.datasets import CIFAR10

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

result = torch.load("../analysis-and-draw/data/CNN-30-CIFAR_17-result.pt").cpu()
correct_num = result.sum(dim=1)
occur = result.int().sum(dim=0)

lst = list()
for i in range(31):
    lst.append((occur == i).int().sum().item())


plt.bar(np.arange(31), lst, width=0.8)
plt.xlabel('Learnable Rate')
plt.ylabel('Percentage')

ax = plt.gca()


def to_percent(temp, position):
    return '%d' % (temp / 500) + '%'


ax.yaxis.set_major_formatter(ticker.FuncFormatter(to_percent))
plt.show()
