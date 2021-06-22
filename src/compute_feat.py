
import os

import torch
from torch import nn
import torch.nn.functional as F
from collections import OrderedDict
import numpy as np
from torch.nn import init
import itertools
from src.model import *
from src.dataset import *
from tqdm import tqdm
import pickle
import random
from itertools import combinations
from sklearn.preprocessing import StandardScaler, MaxAbsScaler
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics.pairwise import rbf_kernel, cosine_similarity
from sklearn.metrics import pairwise_distances


device = 'cuda'


## Register hook

# %%

##### HELPER FUNCTION FOR FEATURE EXTRACTION
def get_features(name):
    def hook(model, input, output):
        features[name] = output.detach()

    return hook


# %%
if __name__ == '__main__':

    # model = BaseModel()
    # model.load_state_dict(torch.load('./checkpoints/CIFAR17_add000000-CIFAR10-model1/199.pt'))

    model = ChildModel(extra_filter=[1, 0, 5])
    model.load_state_dict(torch.load('./checkpoints/CIFAR17_add010005-CIFAR10-model1/199.pt'))

    model.body.cnn1.register_forward_hook(get_features('cnn1'))
    model.body.cnn2.register_forward_hook(get_features('cnn2'))
    model.body.cnn3.register_forward_hook(get_features('cnn3'))

    model.head.dense.fc1.register_forward_hook(get_features('fc1'))
    model.head.dense.fc2.register_forward_hook(get_features('fc2'))

    model.to(device)


    # # test on random tensor
    # x = torch.randn((1, 3, 32, 32))

    # # placeholder for batch features
    # features = {}
    # output = model(x)

    # features['fc1']


    ## Get intermediate representation for all training data


    train_data_loader = data_loader(dataset_name='CIFAR10',
                                    batch_size=128,
                                    train=True)


    # Total training is 50K, too many, downsample 10K, run only ONCE
    random.seed(1234)
    subsample_id = random.sample(range(len(train_data_loader.dataset)), 10000)

    np.save('subsample_train', subsample_id)


    # Directly load the pre-computed subsampled data
    subsample_id = np.load('subsample_train.npy')

    subtrain_data_loader = data_loader(dataset_name='CIFAR10',
                                       batch_size=128,
                                       train=True,
                                       subsample_id=subsample_id)


    ##### FEATURE EXTRACTION LOOP

    # placeholders
    LABELS = []
    FEATS = {}

    # loop through batches
    for inputs, targets in tqdm(subtrain_data_loader):

        # move to device
        inputs = inputs.to(device)

        # placeholder for batch features
        features = {}

        # forward pass [with feature extraction]
        preds = model(inputs)

        # add labels to list
        LABELS.extend(targets.numpy())

        # add feats to lists
        for k in features.keys():
            if k not in FEATS.keys():
                FEATS[k] = features[k].cpu().numpy()
            else:
                FEATS[k] = np.concatenate((FEATS[k], features[k].cpu().numpy()), axis=0)

    LABELS = np.asarray(LABELS)

    ##### SAVE

    with open('checkpoints/CIFAR17_add010005-CIFAR10-model1/feat_representation.pkl', 'wb') as handle:
        pickle.dump(FEATS, handle)

    with open('checkpoints/CIFAR17_add010005-CIFAR10-model1/label.pkl', 'wb') as handle:
        pickle.dump(LABELS, handle)
