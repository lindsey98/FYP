from torch import nn, optim, save, load
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
import torch.nn.functional as F
import torch


class TestModel(nn.Module):
    def __init__(self):
        super(TestModel, self).__init__()
        self.hidden1 = nn.Linear(784, 784)
        self.hidden2 = nn.Linear(784, 784)
        self.output = nn.Linear(784, 10)

    def forward(self, x):
        x = self.hidden1(x)
        x = F.relu(x)
        x = self.hidden2(x)
        x = F.relu(x)
        x = self.output(x)
        x = F.log_softmax(x, dim=1)
        return x


num_epochs = 50
batch_size = 128
learning_rate = 1e-3

img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

dataset = MNIST('./data', transform=img_transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

model = TestModel().cuda()
optimizer = optim.Adam(
    model.parameters(), lr=learning_rate, weight_decay=1e-5)

def train():
    model.train()
    for epoch in range(num_epochs):
        for data in dataloader:
            img, label = data
            img = img.view(img.size(0), -1)
            img = Variable(img).cuda()

            # ===================forward=====================
            optimizer.zero_grad()
            output = model(img)

            loss = F.nll_loss(output, label.cuda())
            # ===================backward====================
            loss.backward()
            optimizer.step()

        print('epoch [{}/{}], loss:{:.4f}'
              .format(epoch + 1, num_epochs, loss.data))
    save(model.state_dict(), './my_test_model.pth')


def test():
    model.load_state_dict(load('./my_test_model.pth'))
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in dataloader:
            data = data.view(data.size(0), -1).cuda()
            output = model(data)

            test_loss += F.nll_loss(output, target.cuda(), reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.cuda().view_as(pred)).sum().item()

    test_loss /= len(dataloader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(dataloader.dataset),
        100. * correct / len(dataloader.dataset)))


if __name__ == '__main__':
    for parameter in model.parameters():
        print(1)
    print(model.parameters()[1])