import torch
from ResidualLoss.dataset import cifar10_data_loader_train
from ResidualLoss.model import CIFAR_17

model = CIFAR_17().cuda()
model.eval()

evaluation_batch_size = 25000
evaluation_data_loader = cifar10_data_loader_train(batch_size=evaluation_batch_size, shuffle=False, loc="../../data")

result_list = list()
for i in range(50, 2001, 50):
    correct_list = list()
    with torch.no_grad():
        state_dict = torch.load('../CNN-Train/CIFAR_17/false_epoch-%s.pt' % i)
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

for i in range(1, 201):
    correct_list = list()
    with torch.no_grad():
        state_dict = torch.load('../CNN-Train/CIFAR_17/back_epoch-%s.pt' % i)
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

torch.save(torch.vstack(result_list), "./data/CNN-30-CIFAR_17-back-result-whole.pt")