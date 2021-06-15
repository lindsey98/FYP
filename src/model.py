import torch
from torch import nn
import torch.nn.functional as F
from collections import OrderedDict
import numpy as np
from torch.nn import init
import itertools

class BaseModel(nn.Module):
    '''
    BaseModel which has 3 CNN layers and 2 FC layers 
    '''
    def __init__(self, head_size=10,
                width=32, height=32, # CIFAR10 is of this shape
                ):
        super(BaseModel, self).__init__()
        
        self.body = nn.Sequential(OrderedDict([
            ('cnn1', self._add_block(in_channels=3, out_channels=8,
                                    kernel_size=3, padding=1)),
            
            ('cnn2', self._add_block(in_channels=8, out_channels=8,
                                    kernel_size=3, padding=1)),  
            
            ('cnn3', self._add_block(in_channels=8, out_channels=8,
                                    kernel_size=3, padding=1))
        ]))
        
        cw, ch = self._get_wh_change(width, height)
        
        self.head = nn.Sequential(OrderedDict([
            ('dense', nn.Sequential(OrderedDict([
                            ('fc1', nn.Conv2d(8 * cw * ch, 32, kernel_size=1, bias=True)),  # implement dense layer in CNN way
                            ('relu', nn.ReLU(inplace=True)),
                            ('fc2', nn.Conv2d(32, head_size, kernel_size=1, bias=True)),
                        ])))
            ]))
        
    def _get_wh_change(self, w, h):
        '''
           Get the change in width or height, didnt consider decovonlution 
        '''
        cw, ch = w, h
        for m in self.body.modules():
            if isinstance(m, nn.MaxPool2d) or isinstance(m, nn.AvgPool2d):
                cw, ch = cw // m.kernel_size, ch // m.kernel_size

            if isinstance(m, nn.Conv2d):
                cw = ((cw - m.kernel_size[1] + 2 * m.padding[1]) // m.stride[1]) + 1
                ch = ((ch - m.kernel_size[0] + 2 * m.padding[0]) // m.stride[0]) + 1

        return cw, ch

    
                
    def _add_block(self, in_channels, out_channels, kernel_size, padding, stride=1):
        '''
           Add CNN block, didnt include BatchNorm (Found it not that effective in underfitting model)
        '''
        return nn.Sequential(OrderedDict([
                            ('conv', nn.Conv2d(in_channels=in_channels, out_channels=out_channels, 
                                               kernel_size=kernel_size, stride=stride, 
                                               padding=padding)),
                            ('relu', nn.ReLU(inplace=True)),
                            ('pool', nn.MaxPool2d(2))
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
   
 
class ChildModel(BaseModel):
    '''
    Add extra CNN filters/ Increase CNN filter size on certain CNN layer(s)
    '''
    def __init__(self, extra_filter=[0, 0, 0], extra_size=[0, 0, 0], # extra filters to add, or larger kernel size
                 head_size=10, # number of classes
                 width=32, height=32, # CIFAR10 is of this shape
                 parent_dict_path=None): # inheritate weights from parents?
        
        super(ChildModel, self).__init__()
        self.extra_filter = extra_filter
        self.extra_size = extra_size
        assert np.sum([(x+3) % 2 == 1 for x in self.extra_size]) == 3 # kernel size should be odd
        
        self.body = nn.Sequential(OrderedDict([
            ('cnn1', self._add_block(in_channels=3, out_channels=8+extra_filter[0],
                                    kernel_size=3+extra_size[0], padding=(3+extra_size[0])//2)),
            
            ('cnn2', self._add_block(in_channels=8+extra_filter[0], out_channels=8+extra_filter[1],
                                    kernel_size=3+extra_size[1], padding=(3+extra_size[1])//2)),   
            
            ('cnn3', self._add_block(in_channels=8+extra_filter[1], out_channels=8+extra_filter[2],
                                    kernel_size=3+extra_size[2], padding=(3+extra_size[2])//2))
        ]))
        
        cw, ch = self._get_wh_change(width, height)
        
        self.head = nn.Sequential(OrderedDict([
            ('dense', nn.Sequential(OrderedDict([
                            ('fc1', nn.Conv2d(in_channels=(8+extra_filter[2]) * ch * cw, out_channels=32, 
                                              kernel_size=1, bias=True)), # implement dense layer in CNN way
                            ('relu', nn.ReLU(inplace=True)),
                            ('fc2', nn.Conv2d(in_channels=32, out_channels=head_size, 
                                              kernel_size=1, bias=True)),
                        ])))
            ]))
        
        if parent_dict_path is not None:
            # inheritate weights
            self._load_from_parent(parent_dict_path)

        
        
    def _load_from_parent(self, weights_path): # might not be useful
        '''
           Customize weight loading function to solve size mismatch problem
        '''
        pretrain_weights = torch.load(weights_path)
        assert set(pretrain_weights.keys()) == set([x[0] for x in self.named_parameters()]) # FIXME: assert that the keys are identical?
        
        for name in pretrain_weights.keys():
            trained_weights = pretrain_weights[name]
            try:
                self.state_dict()[name].copy_(trained_weights)
            except RuntimeError as e: # solve size mismatch exception
                print("Layer {} has been mutated, call dynamic inheritance".format(name))
                self._dynamic_inherit(name, trained_weights)

    
    def _dynamic_inherit(self, name, trained_weights):
        '''
           Dynamic weight inheritance when shape are mismatched
        '''
        if 'weight' in name:
#             print('After initialization', self.state_dict()[name][0])
            initialized_weights = self.state_dict()[name].clone()
            dummy_weights = torch.zeros_like(initialized_weights)
            
            if len(dummy_weights.shape) == 2: # for dense layers
                # only inheritate first few dimensions
                dummy_weights[:trained_weights.shape[0], :trained_weights.shape[1]] = trained_weights 
                mask = (dummy_weights == 0)

                self.state_dict()[name].copy_(initialized_weights*mask + dummy_weights)
#                 print('After copying', self.state_dict()[name][0])
                                            
            elif len(dummy_weights.shape) == 4: # for CNN layers
                dummy_weights[:trained_weights.shape[0], :trained_weights.shape[1],
                              :trained_weights.shape[2], :trained_weights.shape[3]] = trained_weights 
                mask = (dummy_weights == 0)
                self.state_dict()[name].copy_(initialized_weights*mask + dummy_weights)
#                 print('After copying', self.state_dict()[name][0])
            
            else:
                print('Not inheritating layer {} because it\'s number of dimensions is not 2 or 4'.format(name))
        
        elif 'bias' in name:
#             print('Before copying', self.state_dict()[name])
            initialized_weights = self.state_dict()[name].clone()
            dummy_weights = torch.zeros_like(initialized_weights)
            
            dummy_weights[:trained_weights.shape[0]] = trained_weights
            mask = (dummy_weights == 0)
            
            self.state_dict()[name].copy_(initialized_weights*mask + dummy_weights)
#             print('After copying', self.state_dict()[name])
            pass
        
        else:
            print('Not inheritating layer {} because it has neither weight or bias parameters'.format(name))        
    
