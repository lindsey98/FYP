import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
import torch.nn.functional as F

from TBNN.model import MINST_3, MINST_8
from TBNN.neuron_coverage_model import NeuronCoverageReLUModel
from TBNN.train_test import img_transform, num_epochs, train_data_loader, learning_rate

batch_size = 128
model = MINST_3()
model = NeuronCoverageReLUModel(model)

model.load_state_dict(torch.load('./MINST-3.pth'))
model.coverage()

test_dataset = MNIST('./data', transform=img_transform, train=False)
test_data_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


def coverage():
    test_loss = 0
    correct = 0

    neuron_activation_map_correct = dict()
    neuron_activation_map_wrong = dict()
    with torch.no_grad():
        for data, target in train_data_loader:
            data = data.view(data.size(0), -1).cuda()
            output = model.forward(data)
            test_loss += F.nll_loss(output, target.cuda(), reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            target = target.cuda().view_as(pred)
            correct_pred = pred.eq(target).int()

            for name, neuron_activation in model.neuron_activation_map.items():
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

    length = len(train_data_loader.dataset)
    test_loss /= length
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss, correct, length, 100. * correct / length))

    for name, activation_map_correct in neuron_activation_map_correct.items():
        activation_map_wrong = neuron_activation_map_wrong[name]

        contribution = (activation_map_correct.double() / correct) / ((activation_map_wrong + 1).double() / (length - correct))
        contribution_ratio = 1 - 1 / (contribution + 1)
        model.non_frozen_neuron_map[name] = (contribution_ratio < 0.5).int()


def freeze_train():
    model.freeze_train()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=1e-5)
    for epoch in range(10):
        for data in train_data_loader:
            img, label = data
            img = img.view(img.size(0), -1)
            img = Variable(img).cuda()

            # ===================forward=====================
            optimizer.zero_grad()
            output = model.forward(img)

            loss = F.nll_loss(output, label.cuda())
            # ===================backward====================
            loss.backward()
            optimizer.step()

        print('epoch [{}/{}], loss:{:.4f}'
              .format(epoch + 1, num_epochs, loss.data))
    torch.save(model.state_dict(), './MINST-3-A.pth')


def test():
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_data_loader:
            data = data.view(data.size(0), -1).cuda()
            output = model(data)

            test_loss += F.nll_loss(output, target.cuda(), reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.cuda().view_as(pred)).sum().item()

    length = len(test_data_loader.dataset)
    test_loss /= length
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss, correct, length, 100. * correct / length))


if __name__ == '__main__':
    coverage()
    freeze_train()
    test()
