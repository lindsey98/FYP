import torch

from ResidualLoss.dataset import cifar10_data_loader_train
from ResidualLoss.model import CIFAR_17

# model = CIFAR_17().cuda()
# model.eval()
#
# evaluation_batch_size = 10000
# evaluation_data_loader = cifar10_data_loader_train(batch_size=evaluation_batch_size, shuffle=False, loc="../../data")
#
# result_list = list()
# for i in range(6, 31, 6):
#     correct_list = list()
#     with torch.no_grad():
#         state_dict = torch.load('../CNN-l2-30/iter-%s.pt' % i)
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
# torch.save(torch.vstack(result_list), "./data/CNN-l2-30-result.pt")

result = torch.load("./data/CNN-l2-30-result.pt")
print(result.shape)
result = result.int().sum(dim=0)

print("all_correct: ", (result == 5).int().sum())
print("all_wrong: ", (result == 0).int().sum())
