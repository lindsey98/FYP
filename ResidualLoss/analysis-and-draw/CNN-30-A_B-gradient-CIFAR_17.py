import torch
import matplotlib.pyplot as plt
import os
from torch import optim
import torch.nn.functional as F

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

from ResidualLoss.dataset import cifar10_dataset_train
from ResidualLoss.model import CIFAR_17

model = CIFAR_17().cuda()
model.train()
learning_rate = 0.0001
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
state_dict = torch.load('../CNN-Train/CIFAR_17/false_epoch-2000.pt')
model.load_state_dict(state_dict)
train_dataset = cifar10_dataset_train(loc="../../data")

_, init_train_list, init_not_train_list, after_train_list, after_not_train_list = torch.load("./data/CNN-30-CIFAR_17-A_B.pt")

B_17 = set(init_train_list) - set(after_train_list)
A_17 = set(after_not_train_list) - set(init_not_train_list)

batch_A_data_list = list()
batch_A_label_list = list()

for i in A_17:
    data, label = train_dataset.__getitem__(i)
    batch_A_data_list.append(data)
    batch_A_label_list.append(torch.tensor(label))

batch_A_data = torch.stack(batch_A_data_list).cuda()
batch_A_label = torch.stack(batch_A_label_list).cuda()

batch_B_data_list = list()
batch_B_label_list = list()

for i in B_17:
    data, label = train_dataset.__getitem__(i)
    batch_B_data_list.append(data)
    batch_B_label_list.append(torch.tensor(label))

batch_B_data = torch.stack(batch_B_data_list).cuda()
batch_B_label = torch.stack(batch_B_label_list).cuda()

optimizer.zero_grad()

output = model(batch_A_data)
pred = output.argmax(dim=1)
# print(pred.eq(batch_A_label).int().sum().item())
loss = F.nll_loss(output, batch_A_label)
loss.backward()

grad_A_dict = dict()
for name, param in model.named_parameters():
    if "conv" in name and "weight" in name:
        grad_A_dict[name] = param.grad.sign().detach().clone().cpu()

optimizer.zero_grad()

output = model(batch_B_data)
pred = output.argmax(dim=1)
# print(pred.eq(batch_B_label).int().sum().item())
loss = F.nll_loss(output, batch_B_label)
loss.backward()

grad_B_dict = dict()
for name, param in model.named_parameters():
    if "conv" in name and "weight" in name:
        grad_B_dict[name] = param.grad.sign().detach().clone().cpu()


color_dict = [str(i / 10) for i in reversed(range(0, 10))]

plt.rcParams['figure.figsize'] = (12.0, 12.0)
# plt.rcParams['figure.dpi'] = 100

start_x = 0
for name, grad_A in grad_A_dict.items():
    grad_B = grad_B_dict[name]

    conflict_grad = (grad_A != grad_B).int().sum(dim=(2, 3))
    print(conflict_grad)
    in_channel = conflict_grad.size(1)
    output_channel = conflict_grad.size(0)

    for i in range(output_channel):
        for j in range(in_channel):
            color = color_dict[conflict_grad[i][j]]
            plt.plot([start_x, start_x + 1], [j, i], color=color)
    start_x += 1

plt.show()
