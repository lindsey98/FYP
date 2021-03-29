import torch
import matplotlib.pyplot as plt
import os
import numpy as np
from matplotlib import ticker

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

result = torch.load("../analysis-and-draw/data/CNN-30-CIFAR_17-result.pt").cpu()
correct_num = result.sum(dim=1)
occur = result.int().sum(dim=0)

lst = list()
for i in range(31):
    lst.append((occur == i).int().sum().item())

lst1 = [0] * 31
lst2 = [0] * 31

false_list = torch.load("../analysis-and-draw/data/CNN_CIFAR_17-false.pt")

for i in range(50000):
    if i in false_list:
        lst2[occur[i]] += 1
    else:
        lst1[occur[i]] += 1

for i in range(31):
    assert lst1[i] + lst2[i] == lst[i]

plt.rcParams['figure.figsize'] = (10.0, 6.0)

plt.title("Learnable Rate Histogram Against Model To Fix")
plt.bar(np.arange(31), lst, width=0.25, label="All Samples")
plt.bar(np.arange(31) + 0.25, lst1, width=0.25, label="Correct Samples in Model to Fix")
plt.bar(np.arange(31) + 0.5, lst2, width=0.25, label="Wrong Samples in Model to Fix")
plt.xticks(np.arange(31) + 0.25, np.arange(31))
plt.xlabel('Learnable Rate')
plt.ylabel('Percentage')
plt.legend(loc='upper left')
plt.xlim(-0.5, 31)
ax = plt.gca()


def to_percent(temp, position):
    return '%d' % (temp / 500) + '%'


ax.yaxis.set_major_formatter(ticker.FuncFormatter(to_percent))
plt.show()