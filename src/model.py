import torch
from torch import nn
import torch.nn.functional as F
from collections import OrderedDict

    
class CIFAR_17(nn.Module):
    '''
    BaseModel which has 3 CNN layers and 2 FC layers 
    '''
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
    
class CIFAR_17_Add(CIFAR_17):
    '''
    Add extra CNN filters on certain CNN layer(s)
    '''
    def __init__(self, extra_filter=[0, 0, 0], head_size=10):
        super(CIFAR_17_Add, self).__init__()
        self.extra_filter = extra_filter
        self.body = nn.Sequential(OrderedDict([
            ('cnn1', nn.Sequential(OrderedDict([
                            ('conv', nn.Conv2d(3, 8+extra_filter[0], 3, 1, 1)),
                            ('relu', nn.ReLU(inplace=True)),
                            ('pool', nn.MaxPool2d(2))
                        ]))),
            ('cnn2', nn.Sequential(OrderedDict([
                            ('conv', nn.Conv2d(8+extra_filter[0], 8+extra_filter[1], 3, 1, 1)),
                            ('relu', nn.ReLU(inplace=True)),
                            ('pool', nn.MaxPool2d(2))
                        ]))),             
            ('cnn3', nn.Sequential(OrderedDict([
                            ('conv', nn.Conv2d(8+extra_filter[1], 8+extra_filter[2], 3, 1, 1)),
                            ('relu', nn.ReLU(inplace=True)),
                            ('pool', nn.MaxPool2d(2)),
                        ])))
        ]))
        
        self.head = nn.Sequential(OrderedDict([
            ('dense', nn.Sequential(OrderedDict([
                            ('fc1', nn.Conv2d((8+extra_filter[2]) * 4 * 4, 32, kernel_size=1, bias=True)), # implement dense layer in CNN way
                            ('relu', nn.ReLU(inplace=True)),
                            ('fc2', nn.Conv2d(32, head_size, kernel_size=1, bias=True)),
                        ])))
            ]))
        
    def load_from(self, weights_path):
        pretrain_weights = torch.load(weights_path)
        with torch.no_grad():
            if extra_filter[1] == 0 and extra_filter[2] == 0:
                self.body.cnn3.conv.weight.copy_ (pretrain_weights['body.cnn3.conv.weight']) 
                self.head.dense.fc1.weight.copy_ (pretrain_weights['head.dense.fc1.weight'])
                self.head.dense.fc2.weight.copy_ (pretrain_weights['head.dense.fc2.weight'])
            elif extra_filter[0] == 0 and extra_filter[2] == 0:
                self.body.cnn1.conv.weight.copy_ (pretrain_weights['body.cnn1.conv.weight'])
                self.head.dense.fc1.weight.copy_ (pretrain_weights['head.dense.fc1.weight'])
                self.head.dense.fc2.weight.copy_ (pretrain_weights['head.dense.fc2.weight'])
            elif extra_filter[0] == 0 and extra_filter[1] == 0:
                self.body.cnn1.conv.weight.copy_ (pretrain_weights['body.cnn1.conv.weight'])
                self.body.cnn2.conv.weight.copy_ (pretrain_weights['body.cnn2.conv.weight'])
                self.head.dense.fc2.weight.copy_ (pretrain_weights['head.dense.fc2.weight'])

                

        
    
KNOWN_MODELS = OrderedDict([
    ('CIFAR17', lambda *a, **kw: CIFAR_17(10, *a, **kw)),
    ('CIFAR17_add1', lambda *a, **kw: CIFAR_17_Add([1, 0, 0], 10, *a, **kw)),
    ('CIFAR17_add2', lambda *a, **kw: CIFAR_17_Add([0, 1, 0],10, *a, **kw)),
    ('CIFAR17_add3', lambda *a, **kw: CIFAR_17_Add([0, 0, 1],10, *a, **kw)),
    ('CIFAR17_double1', lambda *a, **kw: CIFAR_17_Add([8, 0, 0], 10, *a, **kw)),
    ('CIFAR17_double2', lambda *a, **kw: CIFAR_17_Add([0, 8, 0],10, *a, **kw)),
    ('CIFAR17_double3', lambda *a, **kw: CIFAR_17_Add([0, 0, 8],10, *a, **kw)),
    
])
