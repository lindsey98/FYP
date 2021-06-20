# %%

import os

# os.chdir('..')

# %%

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
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import networkx as nx

# %%

device = 'cuda' if torch.cuda.is_available() else 'cpu'


# %% md

## Register hook

##### HELPER FUNCTION FOR FEATURE EXTRACTION
def get_features(name):
    def hook(model, input, output):
        features[name] = output.detach()

    return hook


def Kernel_Mask(X, LABELS, method='linear'):
    '''
    Compute Distance with Labels
    :param X: shape N x p
    :param LABELS: gt labels shape N
    :param method: distance type
    '''

    if method == 'linear':
        # first standardize
        scaler = MinMaxScaler()  # 0-1之间
        X = scaler.fit_transform(X)
        L_X = cosine_similarity(X, X)  # 越大离得越近
    #         print(L_X.shape)

    elif method == 'rbf':
        # first standardize
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        L_X = rbf_kernel(X, X)  # 越大离得越近
    #         print(L_X.shape)

    elif method == 'euclidean':
        # first standardize
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        L_X = pairwise_distances(X, X, metric='euclidean')  # 越小离得越近
    #         print(L_X.shape)

    else:
        raise NotImplementError

    # create mask for observations with same labels except for itself
    mask = ((LABELS[:, None] == LABELS).astype(float)) * (1 - np.eye(L_X.shape[0]))
    # create mask for observations with diff labels except for itself
    reverse_mask = ((LABELS[:, None] != LABELS).astype(float)) * (1 - np.eye(L_X.shape[0]))

    same_sum = np.dot(mask, L_X) * np.eye(L_X.shape[0])  # avg distance with same label
    diff_sum = np.dot(reverse_mask, L_X) * np.eye(L_X.shape[0])  # avg distance with different label

    same_avg = np.diagonal(same_sum) / np.sum(mask, axis=0)
    diff_avg = np.diagonal(diff_sum) / np.sum(reverse_mask, axis=0)

    return L_X, same_avg, diff_avg


def PCA_clustering(X):
    '''
    First do PCA dimension reduction, then do clustering
    :param X: shape Nxp
    :return:
    '''
    # PCA requires standardized data
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # if X.shape[1] > 10:
    #     pca = PCA(n_components=10, svd_solver='full')
    #     X_proj = pca.fit_transform(X)
        # the information lost alot for the first few layers
        # plt.plot(np.cumsum(pca.explained_variance_ratio_))
        # plt.xlabel('number of components')
        # plt.ylabel('cumulative explained variance')
        # plt.show()
    # else:
    #     X_proj = X

    X_proj = X
    # perform clustering
    kmeans = KMeans(n_clusters=10).fit(X_proj)
    cluster_labels = kmeans.labels_

    index_dict = {}
    for i in range(max(cluster_labels)+1):
        index_dict['c'+str(i)] = np.where(cluster_labels == i)[0]

    return X_proj, index_dict

def bipatite_max_weight_matching(source, target):
    '''
    Maximum weight bipatite matching
    :param source: source nodes
    :param target: target nodes
    :return:
    '''
    G = nx.Graph()
    top_nodes = list(target.keys())
    bottom_nodes = list(source.keys())
    G.add_nodes_from(top_nodes, bipartite=0)
    G.add_nodes_from(bottom_nodes, bipartite=1)

    for t in target.keys():
        for s in source.keys():
            match_ratio = 1 - len(set(source[s]).intersection(set(target[t]))) / len(set(source[s]).union(set(target[t])))
            G.add_edge(t, s, weight=match_ratio)

    matching = nx.algorithms.bipartite.minimum_weight_full_matching(G, top_nodes, "weight")
    matched_dict = [(t, s) for (t, s) in matching.items()]

    objective = []
    for t, s in matching.items():
        objective.append(1 - G.get_edge_data(t, s)['weight'])

    return matched_dict, objective

if __name__ == '__main__':

    # model = BaseModel()
    # model.load_state_dict(torch.load('./checkpoints/CIFAR17_add000000-CIFAR10-model1/199.pt'))

    # model = ChildModel(extra_filter=[1, 0, 5])
    # model.load_state_dict(torch.load('./checkpoints/CIFAR17_add010005-CIFAR10-model1/199.pt'))
    #
    # model.body.cnn1.register_forward_hook(get_features('cnn1'))
    # model.body.cnn2.register_forward_hook(get_features('cnn2'))
    # model.body.cnn3.register_forward_hook(get_features('cnn3'))
    #
    # model.head.dense.fc1.register_forward_hook(get_features('fc1'))
    # model.head.dense.fc2.register_forward_hook(get_features('fc2'))
    #
    # model.to(device)

    # Directly load the pre-computed subsampled data
    subsample_id = np.load('subsample_train.npy')

    subtrain_data_loader = data_loader(dataset_name='CIFAR10',
                                       batch_size=128,
                                       train=True,
                                       subsample_id=subsample_id)

    ##### FEATURE EXTRACTION LOOP

    # placeholders
    # LABELS = []
    # FEATS = {}
    #
    # # loop through batches
    # for inputs, targets in tqdm(subtrain_data_loader):
    #
    #     # move to device
    #     inputs = inputs.to(device)
    #
    #     # placeholder for batch features
    #     features = {}
    #
    #     # forward pass [with feature extraction]
    #     preds = model(inputs)
    #
    #     # add labels to list
    #     LABELS.extend(targets.numpy())
    #
    #     # add feats to lists
    #     for k in features.keys():
    #         if k not in FEATS.keys():
    #             FEATS[k] = features[k].cpu().numpy()
    #         else:
    #             FEATS[k] = np.concatenate((FEATS[k], features[k].cpu().numpy()), axis=0)
    #
    # LABELS = np.asarray(LABELS)
    #
    #
    # with open('checkpoints/CIFAR17_add010005-CIFAR10-model1/feat_representation.pkl', 'wb') as handle:
    #     pickle.dump(FEATS, handle)
    #
    # with open('checkpoints/CIFAR17_add010005-CIFAR10-model1/label.pkl', 'wb') as handle:
    #     pickle.dump(LABELS, handle)


    ## Compute distance

    with open('./checkpoints/CIFAR17_add000000-CIFAR10-model1/feat_representation.pkl', 'rb') as handle:
        FEATS = pickle.load(handle)

    with open('./checkpoints/CIFAR17_add000000-CIFAR10-model1/label.pkl', 'rb') as handle:
        LABELS = pickle.load(handle)


    # flatten H, W dimensions
    for k in FEATS.keys():
        # use pooling to do dimension reduction
        if 'cnn' in k:
            FEATS[k] = F.adaptive_avg_pool2d(torch.from_numpy(FEATS[k]), (4,4))
            # print(FEATS[k].shape)
            FEATS[k] = FEATS[k].numpy()
        FEATS[k] = FEATS[k].reshape(FEATS[k].shape[0], FEATS[k].shape[1], -1)


    ## get index that belongs to different labels
    index_label_dict = {}
    for i in range(np.max(LABELS)+1):
        index_label_dict['l'+str(i)] = np.where(LABELS == i)[0]

    # get distingushing potential
    for k in FEATS.keys():
        if 'cnn' in k:
            for dim in range(FEATS[k].shape[1]):
                print('Layer {}, FM {}'.format(k, str(dim+1)))
                X_proj, index_cluster_dict = PCA_clustering(FEATS[k][:, dim, :])
                matched_dict, edge_matched_ratio = bipatite_max_weight_matching(source=index_cluster_dict,
                                                                                target=index_label_dict)
                print(np.mean(edge_matched_ratio))

        elif 'fc' in k:
            print('Layer {}'.format(k))
            X_proj, index_cluster_dict = PCA_clustering(FEATS[k][:, :, 0])
            matched_dict, edge_matched_ratio = bipatite_max_weight_matching(source=index_cluster_dict,
                                                                            target=index_label_dict)
            print(np.mean(edge_matched_ratio))






