import torch
import matplotlib.pyplot as plt
import os
import numpy as np
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

from ResidualLoss.dataset import cifar10_data_loader_train
from ResidualLoss.model import CIFAR_20

model = CIFAR_20().cuda()
model.eval()

evaluation_batch_size = 10000
evaluation_data_loader = cifar10_data_loader_train(batch_size=evaluation_batch_size, shuffle=False, loc="../../data")

result_list = list()

false_list = torch.load("./data/CNN_CIFAR_20-false.pt")

result = torch.load("./data/CNN_CIFAR_20-A-B.pt")

false_num = len(false_list)

train_result_list = list()
for idx in false_list:
    train_result_list.append(result[:, idx])
train_result = torch.vstack(train_result_list)
print(train_result.shape)

plt.rcParams['figure.figsize'] = (12.0, 6.0)
# plt.rcParams['figure.dpi'] = 100

plt.title("Training Accuracy")
plt.plot([str(i) for i in range(51)], result.sum(dim=1).detach().cpu())
plt.plot([str(i) for i in range(51)], train_result.sum(dim=0).detach().cpu())
plt.plot([str(i) for i in range(51)], [false_num] * 51)
plt.legend(labels=['all_acc', 'train_acc', 'train_num'])
# plt.ylim(00, 7000)
plt.show()
plt.xlabel("epoch")
plt.ylabel("accuracy")

