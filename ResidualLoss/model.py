import torch
from torch import nn
import torch.nn.functional as F
from collections import OrderedDict

class CIFAR_11_6(nn.Module):
    def __init__(self, head_size=10):
        
        super(CIFAR_11_6, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2))
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 16, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(16, 16, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        self.dense = nn.Sequential(
            nn.Linear(16 * 4 * 4, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, head_size)
        )

    def forward(self, x):
        x = x.reshape(x.shape[0], 3, 32, 32)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.shape[0], -1)
        x = self.dense(x)
        return x
    
class CIFAR_17(nn.Module):
    def __init__(self, head_size=10):
        super(CIFAR_17, self).__init__()
        
        self.body = nn.Sequential(OrderedDict([
            ('cnn1', nn.Sequential(OrderedDict([
                            ('conv', nn.Conv2d(3, 8, 3, 1, 1)),
                            ('relu', nn.ReLU(inplace=True)),
                            ('pool', nn.MaxPool2d(2))
                        ]))),
            ('cnn2', nn.Sequential(OrderedDict([
                            ('conv', nn.Conv2d(8, 8, 3, 1, 1)),
                            ('relu', nn.ReLU(inplace=True)),
                            ('pool', nn.MaxPool2d(2))
                        ]))),             
            ('cnn3', nn.Sequential(OrderedDict([
                            ('conv', nn.Conv2d(8, 8, 3, 1, 1)),
                            ('relu', nn.ReLU(inplace=True)),
                            ('pool', nn.MaxPool2d(2)),
                        ])))
        ]))
        
        self.head = nn.Sequential(OrderedDict([
            ('dense', nn.Sequential(OrderedDict([
                            ('fc1', nn.Conv2d(8 * 4 * 4, 32, kernel_size=1, bias=True)),  # implement dense layer in CNN way
                            ('relu', nn.ReLU(inplace=True)),
                            ('fc2', nn.Conv2d(32, head_size, kernel_size=1, bias=True)),
                        ])))
            ]))
        

    def features(self, x):
        feat = self.body(x)
        feat = x.view(x.shape[0], -1)
        return feat
    
    def forward(self, x):
        x = self.body(x)
        x = x.view(x.shape[0], -1, 1, 1) # flatten
        x = self.head(x)
        x = x.view(x.shape[0], -1)
        return x
    
    
        
    
    
class CIFAR_20(nn.Module):
    def __init__(self, head_size=10):
        super(CIFAR_20, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2))
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 16, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(16, 8, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        self.dense1 = nn.Sequential(
            nn.Linear(128, 16),
            nn.ReLU(inplace=True)
        )
        self.dense2 = nn.Linear(16, head_size)

    def forward(self, x):
        x = x.reshape(x.shape[0], 3, 32, 32)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.shape[0], -1)
        x = self.dense1(x)
        x = self.dense2(x)
        return x
    
KNOWN_MODELS = OrderedDict([
    ('CIFAR17', lambda *a, **kw: CIFAR_17(10, *a, **kw)),
    ('CIFAR11', lambda *a, **kw: CIFAR_11_6(10, *a, **kw)),
    ('CIFAR20', lambda *a, **kw: CIFAR_20(10, *a, **kw)),
])
