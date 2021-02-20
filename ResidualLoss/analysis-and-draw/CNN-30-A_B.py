import torch
import matplotlib.pyplot as plt
import os
import numpy as np
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

lst_1 = list()  # A
lst_2 = list()  # B


result_1 = torch.load("./data/CNN-30-lower_10-result.pt")
result_2 = torch.load("./data/CNN-30-lower_10-back-result.pt")

for i in range(50000):
    if not result_1[-1][i] and result_2[0][i]:
        lst_1.append(i)

print(len(lst_1))

correct_for_train = torch.load("./data/CNN-30-potential.pt")
train_result_list = list()
for idx in correct_for_train:
    if result_1[-1][idx] and not result_2[0][idx]:
        lst_2.append(idx)


print(len(lst_2))

torch.save((lst_1, lst_2), "./data/CNN-l2-30-A-B.pt")

