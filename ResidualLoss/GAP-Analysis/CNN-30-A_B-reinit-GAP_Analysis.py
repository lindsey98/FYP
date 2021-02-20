import torch
import matplotlib.pyplot as plt
import os
import numpy as np
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

from ResidualLoss.dataset import cifar10_data_loader_train
from ResidualLoss.model import CIFAR_17

k = 10

model = CIFAR_17().cuda()
model.eval()

evaluation_batch_size = 10000
evaluation_data_loader = cifar10_data_loader_train(batch_size=evaluation_batch_size, shuffle=False, loc="../../data")

# feature_list = list()
# with torch.no_grad():
#     state_dict = torch.load('../CNN-30/A_and_B.pt')
#     model.load_state_dict(state_dict)
#
#     start_index = 0
#     for data, target in evaluation_data_loader:
#         data, target = data.cuda(), target.cuda()
#         output, features = model.features(data, retain_grad=False)
#
#         resize_feature = features[2].view(features[2].size(0), features[2].size(1), -1).mean(axis=2)
#
#         feature_list.append(resize_feature)
#
#
# torch.save(torch.vstack(feature_list), "./data/CNN-30-A_B-reinit-feature.pt")
correct_list = torch.load("./data/CNN-30-A_B-reinit-result.pt")
feature_list = torch.load("./data/CNN-30-A_B-reinit-feature.pt")
A, B = torch.load("./data/CNN-30-A-B.pt")
#
# correct_A_now = list()
# correct_B_now = list()

record = dict()
record_list = list()
for i in range(k + 1):
    record[i] = 0

for i in range(50000):
    if i in B:
        dist = feature_list.add(-feature_list[i].expand(feature_list.shape)).pow(2).sum(dim=1).pow(.5)
        knn_indices = dist.topk(k + 1, largest=False, sorted=False)[1]
        total = 0
        for j in knn_indices[1:]:
            if j.item() in A:
                total += 1
        record[total] += 1

for i in range(k + 1):
    record_list.append(record[i])

plt.title("Reinit KNN for B's, k = %s" % k)
plt.xlabel("Num")
plt.ylabel("A's occur_time")
plt.bar([str(i) for i in range(k + 1)], record_list)

plt.show()

