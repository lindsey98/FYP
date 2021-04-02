import torch
import matplotlib.pyplot as plt
import os
import numpy as np
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

from ResidualLoss.dataset import cifar10_data_loader_train
from ResidualLoss.model import CIFAR_17

result = torch.load("../analysis-and-draw/data/CNN-30-CIFAR_17-back-result-whole.pt").cpu()

print(result.shape)

correct_for_train = torch.load("../analysis-and-draw/data/CNN-30-CIFAR_17-lower_10.pt")
train_result_list = list()
not_train_result_list = list()
for i in range(50000):
    if i in correct_for_train:
        train_result_list.append(result[:, i])
    else:
        not_train_result_list.append(result[:, i])
train_result = torch.vstack(train_result_list)
not_train_result = torch.vstack(not_train_result_list)

# plt.rcParams['figure.figsize'] = (12.0, 6.0)

plt.plot(range(90), result.sum(dim=1).detach().cpu()[:90])
plt.plot(range(90), train_result.sum(dim=0).detach().cpu()[:90])
plt.plot(range(90), not_train_result.sum(dim=0).detach().cpu()[:90])
plt.plot(range(90), [len(correct_for_train)] * 90)
plt.legend(labels=['Total Accuracy',
                   'Accuracy for Low Learning Rate Samples',
                   'Accuracy for Other Samples',
                   'Total Num for Low Learning Rate Samples'],
           loc="upper left")

lst = []
for i in range(50, 2001, 50):
    if i % 500 == 0:
        lst.append(str(i))
    else:
        lst.append(None)
for i in range(1, 51):
    if i % 10 == 0:
        lst.append(str(2000 + i))
    else:
        lst.append(None)


plt.xticks(range(90), lst)

plt.ylim(0, 35000)
plt.show()
plt.xlabel("epoch")
plt.ylabel("accuracy")

