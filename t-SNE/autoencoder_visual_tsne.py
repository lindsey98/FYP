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

print(b.size())
print(b.dtype)

x, y, z = model(a.view(2000, -1).cuda())

print(x.size())
print(y.size())


from sklearn.manifold import TSNE
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
import numpy as np

model = TSNE(n_components=2, random_state=0, perplexity=50)
tsne_data = model.fit_transform(y.cpu().detach().numpy())

# creating a new data fram which help us in ploting the result data
tsne_data = np.vstack((tsne_data.T, b)).T
tsne_df = pd.DataFrame(data=tsne_data, columns=("Dim_1", "Dim_2", "label"))

# Ploting the result of tsne
sn.FacetGrid(tsne_df, hue="label", size=6).map(plt.scatter, 'Dim_1', 'Dim_2').add_legend()
plt.title('With perplexity = 50')
plt.show()
