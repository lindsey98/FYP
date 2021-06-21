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
from sklearn.cluster import KMeans, SpectralClustering
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.mixture import GaussianMixture

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def customize_clustering(X, method):
    '''
    First do PCA dimension reduction, then do clustering
    :param X: shape Nxp
    :return:
    '''
    # PCA requires standardized data
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_proj = X
    # perform clustering
    if method == 'kmeans':
        kmeans = KMeans(n_clusters=10).fit(X_proj)
        cluster_labels = kmeans.labels_
    elif method == 'gm':
        gm = GaussianMixture(n_components=10, random_state=0, n_init=5).fit(X_proj)
        cluster_labels = gm.predict(X_proj)
    elif method == 'spectral':
        clustering = SpectralClustering(n_clusters=10,
                                        assign_labels = 'discretize',
                                        random_state = 0).fit(X_proj)
        cluster_labels = clustering.labels_
    else:
        raise NotImplementedError

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

    # Directly load the pre-computed subsampled data
    subsample_id = np.load('subsample_train.npy')

    subtrain_data_loader = data_loader(dataset_name='CIFAR10',
                                       batch_size=128,
                                       train=True,
                                       subsample_id=subsample_id)

    ## Compute distance

    with open('./checkpoints/CIFAR17_add010004-CIFAR10-model1/feat_representation.pkl', 'rb') as handle:
        FEATS = pickle.load(handle)

    with open('./checkpoints/CIFAR17_add010004-CIFAR10-model1/label.pkl', 'rb') as handle:
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
                X_proj, index_cluster_dict = customize_clustering(FEATS[k][:, dim, :], method='spectral')
                matched_dict, edge_matched_ratio = bipatite_max_weight_matching(source=index_cluster_dict,
                                                                                target=index_label_dict)
                print(np.mean(edge_matched_ratio))

        elif 'fc' in k:
            print('Layer {}'.format(k))
            X_proj, index_cluster_dict = customize_clustering(FEATS[k][:, :, 0], method='spectral')
            matched_dict, edge_matched_ratio = bipatite_max_weight_matching(source=index_cluster_dict,
                                                                            target=index_label_dict)
            print(np.mean(edge_matched_ratio))






