
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
from sklearn.preprocessing import StandardScaler, MaxAbsScaler, MinMaxScaler
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics.pairwise import rbf_kernel, cosine_similarity
from sklearn.metrics import pairwise_distances
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, SpectralClustering
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.mixture import GaussianMixture

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def Kernel_Mask(X, LABELS, method='linear'):
    '''
    Compute Distance with Labels
    :param X: shape N x p
    :param LABELS: gt labels shape N
    :param method: distance type, supports linear, rbf, euclidean
    '''

    if method == 'linear':
        # first standardize
        scaler = MinMaxScaler()  # 0-1之间
        X = scaler.fit_transform(X)
        L_X = cosine_similarity(X, X)  # 越大离得越近

    elif method == 'rbf':
        # first standardize
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        L_X = rbf_kernel(X, X)  # 越大离得越近

    elif method == 'euclidean':
        # first standardize
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        L_X = pairwise_distances(X, X, metric='euclidean')  # 越小离得越近

    else:
        raise NotImplementedError

    # create mask for observations with same labels except for itself
    mask = ((LABELS[:, None] == LABELS).astype(float)) * (1 - np.eye(L_X.shape[0]))
    # create mask for observations with diff labels except for itself
    reverse_mask = ((LABELS[:, None] != LABELS).astype(float)) * (1 - np.eye(L_X.shape[0]))

    same_sum = np.dot(mask, L_X) * np.eye(L_X.shape[0])  # avg distance with same label
    diff_sum = np.dot(reverse_mask, L_X) * np.eye(L_X.shape[0])  # avg distance with different label

    same_avg = np.diagonal(same_sum) / np.sum(mask, axis=0)
    diff_avg = np.diagonal(diff_sum) / np.sum(reverse_mask, axis=0)

    return L_X, same_avg, diff_avg

def customize_clustering(X, method):
    '''
    Clustering algorithm
    :param X: shape Nxp
    :param method: clustering methods, support kmeans, gaussianmixture, spectral clustering
    :return:
    '''

    X_proj = X
    # perform clustering
    if method == 'kmeans':
        kmeans = KMeans(n_clusters=10).fit(X_proj)
        cluster_labels = kmeans.labels_
    elif method == 'gm': # soft version of kmeans
        gm = GaussianMixture(n_components=10, random_state=0, n_init=5).fit(X_proj)
        cluster_labels = gm.predict(X_proj)
    elif method == 'spectral': # measure nearest-neighbor distance not euclidean distanc
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


def compute_mu_sigma(layer, labels):
    '''
    Compute class-specific mean and sigma
    :param layer: intermediate feature representations of shape N x p
    :param labels: array of gt class labels of shape N,
    :return:
    '''

    mus = []
    sigmas = []

    for cls in range(np.max(labels)+1):
        clsindex = np.where(labels == cls)[0]
        cls_collection = layer[clsindex, :]
        covariance = np.cov(cls_collection, rowvar=False) # get covariance matrix
        mean = np.mean(cls_collection, axis=0) # get mean vector
        mus.append(mean)
        sigmas.append(covariance)

    mus = np.asarray(mus) # C x p
    sigmas = np.asarray(sigmas) # C x p x p
    return mus, sigmas


def mahalanobis_dist(X, mus, sigmas):
    '''
    Compute Mahalanobis distance
    :param X:
    :param mus: C x p
    :param sigmas: C x p x p
    :return:
    '''
    num_cls = len(mus)
    cls_distances = np.zeros((num_cls, X.shape[0])) # shape C x N
    for cls in range(num_cls): # iterate over classes
        for i in range(len(X)): # iterate over observations
            mean_shift = X[i, :]-mus[cls].T
            distance = np.dot(np.dot(mean_shift, np.linalg.pinv(sigmas[cls])), mean_shift.T) # use pseudo-inverse in case sigma is singular
            cls_distances[cls, i] = distance

    index = np.argsort(cls_distances, axis=0)
    sorted_cls_distance = np.take_along_axis(cls_distances, index, axis=0)
    top1_top2_distance = sorted_cls_distance[1, :] - sorted_cls_distance[0, :] # return top1 distance - top2 distance
    return cls_distances, top1_top2_distance

if __name__ == '__main__':


    # Directly load the pre-computed subsampled data
    subsample_id = np.load('subsample_train.npy')

    subtrain_data_loader = data_loader(dataset_name='CIFAR10',
                                       batch_size=128,
                                       train=True,
                                       subsample_id=subsample_id)

    ## Compute distance

    with open('./checkpoints/CIFAR17_add010005-CIFAR10-model1/feat_representation.pkl', 'rb') as handle:
        FEATS = pickle.load(handle)

    with open('./checkpoints/CIFAR17_add010005-CIFAR10-model1/label.pkl', 'rb') as handle:
        LABELS = pickle.load(handle)

    # flatten H, W dimensions
    Mus = {}
    Sigmas = {}
    for k in FEATS.keys():
        # use pooling to do dimension reduction
        if 'cnn' in k:
            # FEATS[k] = F.adaptive_avg_pool2d(torch.from_numpy(FEATS[k]), (4,4)) # reshape to 4 by 4
            FEATS[k] = F.adaptive_avg_pool2d(torch.from_numpy(FEATS[k]), (1,1)) # reshape to 1 by 1

            FEATS[k] = FEATS[k].numpy()
            FEATS[k] = FEATS[k].reshape(FEATS[k].shape[0], FEATS[k].shape[1], -1) # flatten HxW
            temporal_mus, temporal_sigmas = [], []
            # for dim in range(FEATS[k].shape[1]):
            mus, sigmas = compute_mu_sigma(FEATS[k][:, :, 0], LABELS)
            temporal_mus.append(mus)
            temporal_sigmas.append(sigmas)
            Mus[k] = np.asarray(temporal_mus) # shape D x C x p
            Sigmas[k] = np.asarray(temporal_sigmas) # shape D x C x p x p
        else:
            FEATS[k] = FEATS[k].reshape(FEATS[k].shape[0], FEATS[k].shape[1], -1)
            temporal_mus, temporal_sigmas = compute_mu_sigma(FEATS[k][:, :, 0], LABELS)
            Mus[k] = np.asarray(temporal_mus) # shape D x C x p
            Sigmas[k] = np.asarray(temporal_sigmas) # shape D x C x p x p

    ## get index that belongs to different labels
    # index_label_dict = {}
    # for i in range(np.max(LABELS)+1):
    #     index_label_dict['l'+str(i)] = np.where(LABELS == i)[0]

    # get distingushing potential
    # for k in FEATS.keys():
    #     if 'cnn' in k:
    #         for dim in range(FEATS[k].shape[1]):
    #             print('Layer {}, FM {}'.format(k, str(dim+1)))
    #             X_proj, index_cluster_dict = customize_clustering(FEATS[k][:, dim, :], method='spectral')
    #             matched_dict, edge_matched_ratio = bipatite_max_weight_matching(source=index_cluster_dict,
    #                                                                             target=index_label_dict)
    #             print(np.mean(edge_matched_ratio))
    #
    #     elif 'fc' in k:
    #         print('Layer {}'.format(k))
    #         X_proj, index_cluster_dict = customize_clustering(FEATS[k][:, :, 0], method='spectral')
    #         matched_dict, edge_matched_ratio = bipatite_max_weight_matching(source=index_cluster_dict,
    #                                                                         target=index_label_dict)
    #         print(np.mean(edge_matched_ratio))

    for k in FEATS.keys():
        if 'cnn' in k:
            # for dim in range(FEATS[k].shape[1]):
            #     print('Layer {}, FM {}'.format(k, str(dim + 1)))
                # thismu = Mus[k][dim] # shape C x p
                # thissigma = Sigmas[k][dim] # shape C x p x p
            print('Layer {}'.format(k))
            thismu = Mus[k][0]  # shape C x p
            thissigma = Sigmas[k][0] # shape C x p x p
            # cls_distances, top1_top2_distance = mahalanobis_dist(FEATS[k][:, dim, :], thismu, thissigma)
            cls_distances, top1_top2_distance = mahalanobis_dist(FEATS[k][:, :, 0], thismu, thissigma)
            print(np.mean(top1_top2_distance))
        else:
            print('Layer {}'.format(k))
            thismu = Mus[k]  # shape C x p
            thissigma = Sigmas[k]  # shape C x p x p
            cls_distances, top1_top2_distance = mahalanobis_dist(FEATS[k][:, :, 0], thismu, thissigma)
            print(np.mean(top1_top2_distance))


