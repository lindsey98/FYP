import torch
import matplotlib.pyplot as plt
import PIL
import os
import numpy as np
from torchvision.datasets import CIFAR10

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

from ResidualLoss.dataset import cifar10_data_loader_train
from ResidualLoss.model import CIFAR_17

model = CIFAR_17().cuda()
model.eval()

evaluation_batch_size = 10000
evaluation_data_loader = cifar10_data_loader_train(batch_size=evaluation_batch_size, shuffle=False, loc="../../data")

# result_list = list()
# for i in range(1, 31):
#     correct_list = list()
#     with torch.no_grad():
#         state_dict = torch.load('../CNN-30/iter-%s.pt' % i)
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
#     result_list.append(torch.hstack(correct_list))
#     print(i)
#
# torch.save(torch.vstack(result_list), "./data/CNN-30-result.pt")

result = torch.load("./data/CNN-30-result.pt")
correct_num = result.sum(dim=1)
occur = result.int().sum(dim=0)

# raw_data = CIFAR10("../../data")
# worst_sample = list()
# label = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
# for i in range(50000):
#     if occur[i] == 0:
#         raw_data[i][0].save("../worst-image/%s-%s.jpg" % (i, label[raw_data[i][1]]))


# raw_data = CIFAR10("../../data")
# good_sample = list()
# label = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
# count = 0
# for i in range(50000):
#     if occur[i] == 30:
#         if count % 10 == 0:
#             raw_data[i][0].save("../best-image/%s-%s.jpg" % (i, label[raw_data[i][1]]))
#         count += 1

# lst = list()
# for i in range(31):
#     lst.append((occur == i).int().sum().item())
#
# plt.title("All Accuracy for 30 epochs")
# plt.ylim(36000, 37300)
# plt.bar(range(30), correct_num.cpu().numpy().tolist())
# plt.show()
#
# plt.title("Learnable Rate Hist")
# plt.bar(range(31), lst)
# plt.xlabel('occur')
# plt.ylabel('num')
# plt.xlim(0, 30)
# plt.show()
#
# state_dict = torch.load('../CIFAR-17-1.pt')
# model.load_state_dict(state_dict)
#
# lst_1 = [0] * 31
# lst_2 = list()
# start_index = 0
# for data, target in evaluation_data_loader:
#     data, target = data.cuda(), target.cuda()
#     output = model(data)
#
#     pred = output.argmax(dim=1)
#     correct = pred.eq(target)
#     for i in range(evaluation_batch_size):
#         if not correct[i].item():
#             lst_1[occur[start_index + i]] += 1
#             if occur[start_index + i] <= 10:
#                 lst_2.append(start_index + i)
#     start_index += evaluation_batch_size
#
# plt.rcParams['figure.figsize'] = (12.0, 6.0)
# plt.rcParams['figure.dpi'] = 100
#
# plt.title("Learnable Rate Hist")
# plt.bar(np.arange(31), lst, width=0.4)
# plt.bar(np.arange(31) + 0.4, lst_1, width=0.4)
# plt.xticks(np.arange(31) + 0.4/2, np.arange(31))
# plt.xlabel('occur')
# plt.ylabel('num')
# plt.show()
#
# torch.save(lst_2, "./data/CNN-30-potential.pt")
# print(lst_1)
# print(len(lst_2))

lst_3 = list()
for i in range(50000):
    if occur[i] == 0:
        lst_3.append(i)

print(len(lst_3))
torch.save(lst_3, "./data/CNN-l2-30-worst-1.pt")