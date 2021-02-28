import torch
import matplotlib.pyplot as plt
import os
import numpy as np
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

from ResidualLoss.dataset import cifar10_data_loader_train
from ResidualLoss.model import CIFAR_17

# model = CIFAR_17().cuda()
# model.eval()
#
# evaluation_batch_size = 5000
# evaluation_data_loader = cifar10_data_loader_train(batch_size=evaluation_batch_size, shuffle=False, loc="../../data")
#
# result_list = list()
# correct_list = list()
# with torch.no_grad():
#     state_dict = torch.load('../CNN-Train/CIFAR_17/false_epoch-2000.pt')
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
#         state_dict = torch.load('../CNN-Train/CIFAR_17/back_epoch-%s.pt' % i)
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
# torch.save(torch.vstack(result_list), "./data/CNN-30-CIFAR_17-back-result.pt")

result = torch.load("./data/CNN-30-CIFAR_17-back-result.pt").cpu()

correct_for_train = torch.load("./data/CNN-30-CIFAR_17-lower_10.pt")
train_result_list = list()
not_train_result_list = list()
for i in range(50000):
    if i in correct_for_train:
        train_result_list.append(result[:, i])
    else:
        not_train_result_list.append(result[:, i])
train_result = torch.vstack(train_result_list)
not_train_result = torch.vstack(not_train_result_list)

# result = result.sum(dim=1).detach().cpu()
train_result_data = train_result.sum(dim=0).detach().cpu()
# not_train_result = not_train_result.sum(dim=0).detach().cpu()

s = 5
t = 50

for i in range(46):
    if torch.abs(train_result_data[i + s] - train_result_data[i]) < t:
        init_train_list = list()
        init_not_train_list = list()

        for j in range(50000):
            if result[0, j].item():
                if j in correct_for_train:
                    init_train_list.append(j)
                else:
                    init_not_train_list.append(j)

        print(len(init_train_list))
        print(len(init_not_train_list))

        after_train_list = list()
        after_not_train_list = list()

        for j in range(50000):
            if result[i, j].item():
                if j in correct_for_train:
                    after_train_list.append(j)
                else:
                    after_not_train_list.append(j)

        print(len(after_train_list))
        print(len(after_not_train_list))

        torch.save((correct_for_train, init_train_list, init_not_train_list, after_train_list, after_not_train_list),
                   "./data/CNN-30-CIFAR_17-A_B.pt")
        break

