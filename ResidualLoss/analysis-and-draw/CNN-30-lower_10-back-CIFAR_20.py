import torch
import matplotlib.pyplot as plt
import os
import numpy as np
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

from ResidualLoss.dataset import cifar10_data_loader_train
from ResidualLoss.model import CIFAR_20

# model = CIFAR_20().cuda()
# model.eval()
#
# evaluation_batch_size = 5000
# evaluation_data_loader = cifar10_data_loader_train(batch_size=evaluation_batch_size, shuffle=False, loc="../../data")
#
# result_list = list()
# correct_list = list()
# with torch.no_grad():
#     state_dict = torch.load('../CNN-Train/CIFAR_20/false_epoch-2000.pt')
#     model.load_state_dict(state_dict)
#
#     start_index = 0
#     for data, target in evaluation_data_loader:
#         data, target = data.cuda(), target.cuda()
#         output = model(data)
#
#         pred = output.argmax(dim=1)
#         correct = pred.eq(target)
#         correct_list.append(correct)
# result_list.append(torch.hstack(correct_list).detach())
#
# for i in range(1, 51):
#     correct_list = list()
#     with torch.no_grad():
#         state_dict = torch.load('../CNN-Train/CIFAR_20/back_epoch-%s.pt' % i)
#         model.load_state_dict(state_dict)
#
#         start_index = 0
#         for data, target in evaluation_data_loader:
#             data, target = data.cuda(), target.cuda()
#             output = model(data)
#
#             pred = output.argmax(dim=1)
#             correct = pred.eq(target)
#             correct_list.append(correct)
#     result_list.append(torch.hstack(correct_list).detach())
#     print(i)
#
# torch.save(torch.vstack(result_list), "./data/CNN-30-CIFAR_20-back-result.pt")

result = torch.load("./data/CNN-30-CIFAR_20-back-result.pt")

correct_for_train = torch.load("./data/CNN-30-CIFAR_20-lower_10.pt")
train_result_list = list()
not_train_result_list = list()
for i in range(50000):
    if i in correct_for_train:
        train_result_list.append(result[:, i])
    else:
        not_train_result_list.append(result[:, i])
train_result = torch.vstack(train_result_list)
not_train_result = torch.vstack(not_train_result_list)

plt.rcParams['figure.figsize'] = (12.0, 6.0)

plt.title("Accuracy Trend For CIFAR_20")
plt.plot([str(i) for i in range(51)], result.sum(dim=1).detach().cpu())
plt.plot([str(i) for i in range(51)], train_result.sum(dim=0).detach().cpu())
plt.plot([str(i) for i in range(51)], not_train_result.sum(dim=0).detach().cpu())
plt.plot([str(i) for i in range(51)], [len(correct_for_train)] * 51)
plt.legend(labels=['all_acc', 'bad_acc', 'good_acc', 'train_num'])
plt.ylim(0, 30000)
plt.show()
plt.xlabel("epoch")
plt.ylabel("accuracy")

