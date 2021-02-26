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
correct_list = list()
with torch.no_grad():
    state_dict = torch.load('../CNN-Train/CIFAR_20/false_epoch-2000.pt')
    model.load_state_dict(state_dict)

    start_index = 0
    for data, target in evaluation_data_loader:
        data, target = data.cuda(), target.cuda()
        output = model(data)

        pred = output.argmax(dim=1)
        correct = pred.eq(target)
        correct_list.append(correct)
result_list.append(torch.hstack(correct_list).detach())

for i in range(1, 51):
    correct_list = list()
    with torch.no_grad():
        state_dict = torch.load('../CNN-Train/CIFAR_20/back_epoch-%s.pt' % i)
        model.load_state_dict(state_dict)

        start_index = 0
        for data, target in evaluation_data_loader:
            data, target = data.cuda(), target.cuda()
            output = model(data)

            pred = output.argmax(dim=1)
            correct = pred.eq(target)
            correct_list.append(correct)
    result_list.append(torch.hstack(correct_list).detach())
    print(i)

torch.save(torch.vstack(result_list), "./data/CNN-30-CIFAR_20-back-result.pt")

result = torch.load("./data/CNN-30-CIFAR_20-back-result.pt")

false_list = torch.load("./data/CNN-30-CIFAR_20-lower_10.pt")
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

