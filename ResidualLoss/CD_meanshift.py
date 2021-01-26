import json
import time

import numpy as np
import torch
from sklearn.cluster import k_means

# grads = np.load('CD/8-dim.npy')
# print(grads.shape)
#
correct_list = np.load('CD/correct.npy')
# print(correct_list.sum())
#
# before = time.time()
# center, idx, _ = k_means(grads, n_clusters=100)
# print(center)
# print(idx)
# np.save('CD/k_means-8-dim-100-n-center', center)
# np.save('CD/k_means-8-dim-100-n-idx', idx)
# after = time.time()
#
# print(after - before)

# with open('data.json', 'w') as f:
#     json.dump(data, f)

# with open('data.json', 'r') as f:
#     data = json.load(f)

center = np.load('CD/k_means-8-dim-100-n-center.npy')
idx = np.load('CD/k_means-8-dim-100-n-idx.npy')

cluster = dict()
for i in range(100):
    cluster[i] = list()

for i in range(len(idx)):
    cluster[idx[i]].append(i)

high_value = 0
high_labels = list()
low_value = 0
low_labels = list()

percentage_list = list()
normal_labels = list()

for label, lst in cluster.items():
    correct_sum = 0
    for i in lst:
        if correct_list[i]:
            correct_sum += 1

    percentage = correct_sum / len(lst)
    percentage_list.append(percentage)

    # print(len(lst), percentage)
    if percentage > 0.90:
        high_value += len(lst)
        high_labels.append(label)
    elif percentage < 0.50:
        low_value += len(lst)
        low_labels.append(label)
    else:
        normal_labels.append(label)

print(high_value)
print(low_value)

weighted_center = 0
total_weight = 0
for label in low_labels:
    weight = len(cluster[label])
    weighted_center += center[label] * weight
    total_weight += weight

weighted_center /= total_weight

normal_labels_dist = list()
# high_labels_dist = list()

ignore_idx_lst = list()


for label in normal_labels:
    dist = np.linalg.norm(weighted_center - center[label], ord=2)
    normal_labels_dist.append((dist, label, len(cluster[label])))
    if dist > 5e-06:
        ignore_idx_lst.extend(cluster[label])

# for label in high_labels:
#     high_labels_dist.append((np.linalg.norm(weighted_center - center[label], ord=2), label, len(cluster[label])))

# print(list(sorted(normal_labels_dist, key=lambda x: x[0], reverse=True)))
# print(list(sorted(high_labels_dist, key=lambda x: x[0], reverse=True)))

# print(ignore_idx_lst)
torch.save(ignore_idx_lst, 'CD/ignore_idx_lst.pt')
