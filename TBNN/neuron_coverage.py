import torch
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
import torch.nn.functional as F

from TBNN.model import MINST_3, MINST_8
from TBNN.neuron_coverage_model import NeuronCoverageReLUModel
from TBNN.train_test import img_transform

batch_size = 128
model = MINST_3()
model = NeuronCoverageReLUModel(model)

model.load_state_dict(torch.load('./MINST-3.pth'))
model.eval()

test_dataset = MNIST('./data', transform=img_transform, train=False)
test_data_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
test_loss = 0
correct = 0

neuron_activation_map_correct = dict()
neuron_activation_map_wrong = dict()
with torch.no_grad():
    for data, target in test_data_loader:
        data = data.view(data.size(0), -1).cuda()
        output, map = model.forward(data)
        test_loss += F.nll_loss(output, target.cuda(), reduction='sum').item()  # sum up batch loss
        pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        target = target.cuda().view_as(pred)
        correct_pred = pred.eq(target).int()

        for name, neuron_activation in map.items():
            pred_neuron_activation_correct = (neuron_activation * correct_pred).sum(axis=0)
            pred_neuron_activation_wrong = (neuron_activation * (1 - correct_pred)).sum(axis=0)

            if neuron_activation_map_correct.get(name) is None:
                neuron_activation_map_correct[name] = pred_neuron_activation_correct
            else:
                neuron_activation_map_correct[name] += pred_neuron_activation_correct

            if neuron_activation_map_wrong.get(name) is None:
                neuron_activation_map_wrong[name] = pred_neuron_activation_wrong
            else:
                neuron_activation_map_wrong[name] += pred_neuron_activation_wrong

        correct += correct_pred.sum().item()

length = len(test_data_loader.dataset)
test_loss /= length

# f = open("./result.csv", "w")
# for name, item in neuron_activation_map_correct.items():
#     f.write(name + ",\n")
#     for i in item:
#         f.write(str(i.item()) + ", ")
#     f.write("\n")
# for name, item in neuron_activation_map_wrong.items():
#     f.write(name + ",\n")
#     for i in item:
#         f.write(str(i.item()) + ", ")
#     f.write("\n")

print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss, correct, length, 100. * correct / length))
