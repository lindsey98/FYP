import torch
from torch import nn
from PIL import Image
from torchvision import transforms
from torchvision.datasets import MNIST


class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 128),
            nn.ReLU(True),
            nn.Linear(128, 64),
            nn.ReLU(True), nn.Linear(64, 12)
        )
        self.last_encoder_layer = nn.Sequential(
            nn.ReLU(True), nn.Linear(12, 3)
        )
        self.decoder = nn.Sequential(
            nn.Linear(3, 12),
            nn.ReLU(True),
            nn.Linear(12, 64),
            nn.ReLU(True),
            nn.Linear(64, 128),
            nn.ReLU(True), nn.Linear(128, 28 * 28), nn.Tanh()
        )

    def forward(self, x):
        y = self.encoder(x)

        z = self.last_encoder_layer(y)
        w = self.decoder(z)
        return y, z, w


img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

dataset = MNIST('./data', transform=img_transform)

model = AutoEncoder().cuda()

a = torch.cat([img_transform(Image.fromarray(i.numpy(), mode='L')) for i in dataset.data[:2000]])

b = dataset.targets[:2000]

x1, y1, z1 = model(a.view(2000, -1).cuda())


import matplotlib.pyplot as plt

fig = plt.figure()
ax1 = plt.axes(projection='3d')

y1 = y1.cpu().detach().numpy()

for i in range(y1.shape[0]):
    if b[i] == 0:
        color = 'red'
    if b[i] == 1:
        color = 'orange'
    if b[i] == 2:
        color = 'yellow'
    if b[i] == 3:
        color = 'green'
    if b[i] == 4:
        color = 'blue'
    if b[i] == 5:
        color = 'indigo'
    if b[i] == 6:
        color = 'pink'
    if b[i] == 7:
        color = 'violet'
    if b[i] == 8:
        color = 'gray'
    if b[i] == 9:
        color = 'black'
    ax1.scatter3D(y1[i, 0], y1[i, 1], y1[i, 2], color=color)

plt.show()