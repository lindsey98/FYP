import torch
import matplotlib.pyplot as plt
import os
import numpy as np
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

from ResidualLoss.dataset import cifar10_data_loader_train
from ResidualLoss.model import CIFAR_17

model = CIFAR_17().cuda()
model.eval()

evaluation_batch_size = 10000
evaluation_data_loader = cifar10_data_loader_train(batch_size=evaluation_batch_size, shuffle=False, loc="../../data")

correct_list = list()
with torch.no_grad():
    state_dict = torch.load('../CNN-30/A_and_B-reinit.pt')
    model.load_state_dict(state_dict)

    start_index = 0
    for data, target in evaluation_data_loader:
        data, target = data.cuda(), target.cuda()
        output = model(data)

        pred = output.argmax(dim=1)
        correct = pred.eq(target)
        correct_list.append(correct)


torch.save(torch.hstack(correct_list), "./data/CNN-30-A_B-reinit-result.pt")

result = torch.load("./data/CNN-30-A_B-reinit-result.pt")
print(result.shape)

A, B = torch.load("./data/CNN-30-A-B.pt")

correct_A_now = list()
correct_B_now = list()

for i in range(50000):
    if result[i].item():
        if i in A:
            correct_A_now.append(i)
        if i in B:
            correct_B_now.append(i)


label_list = ["total_accuracy", "A+B to train", "A+B correct after train", "A to train", "A correct after train", "B to train", "B correct after train"]
result_list = [(result == 1).int().sum().item(), len(A) + len(B), len(correct_A_now) + len(correct_B_now), len(A), len(correct_A_now), len(B), len(correct_B_now)]
plt.rcParams['figure.figsize'] = (12.0, 6.0)
# plt.rcParams['figure.dpi'] = 100

plt.title("Training Accuracy")
plt.bar(label_list, result_list)

plt.show()

