from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from torch import optim, save, load

import torch.nn.functional as F
import torch

from TBNN.model import PaperModel, MINST_3
from TBNN.neuron_coverage_model import NeuronCoverageReLUModel

num_epochs = 100
batch_size = 100
learning_rate = 0.0001

img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

train_dataset = MNIST('./data', transform=img_transform)
train_dataset, validation_dataset = torch.utils.data.random_split(train_dataset, [50000, 10000])
train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = MNIST('./data', transform=img_transform, train=False)
test_data_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

model = MINST_3().cuda()

model = NeuronCoverageReLUModel(model)

optimizer = optim.Adam(
    model.parameters(), lr=learning_rate, weight_decay=1e-5)

def train():
    model.train()
    for epoch in range(num_epochs):
        for data in train_data_loader:
            img, label = data
            img = img.view(img.size(0), -1)
            img = Variable(img).cuda()

            # ===================forward=====================
            optimizer.zero_grad()
            output, _ = model.forward(img)

            loss = F.nll_loss(output, label.cuda())
            # ===================backward====================
            loss.backward()
            optimizer.step()

        print('epoch [{}/{}], loss:{:.4f}'
              .format(epoch + 1, num_epochs, loss.data))
    save(model.state_dict(), './MINST-3-1.pth')


def test():
    model.load_state_dict(load('./MINST-3-1.pth'))
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

    test_loss /= len(test_data_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_data_loader.dataset),
        100. * correct / len(test_data_loader.dataset)))


if __name__ == '__main__':
    train()
    test()
