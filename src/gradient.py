from tqdm import tqdm
import torch
from src.train_test import test_correct
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from torch import optim
from torch import nn
from src.dataset import data_loader
from scipy.spatial import distance
from sklearn.cluster import KMeans

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_graddict(model,
                 model_name, data_name,
                 train_data_loader, num_trail, 
                 pos_thre, neg_thre,
                 compute_index=True, 
                 vis=True, save=True):
    '''
    Get gradient contradiction dictionary for anchor model
    :param model: initialized pytorch model
    :param model_name: name for model
    :param data_name: name for dataset
    :param train_data_loader: dataloader
    :param num_trail: total number of trails trained
    :param pos_thre: threshold for positive samples
    :param neg_thre: threshold for negative samples
    :param compute_index: if True, the function will try to get those positive and negative samples indices, if False, the function will load the pre-computed indices
    :param vis: if True, a histogram of the number of trails that predict this data correctly will be displayed
    :param save: if True, the function will try to save those positive and negative samples indices
    '''
    # if we need to compute correct index
    if compute_index:
        print('Compute correctly predicted data indices...')
        # get correctly predicted index
        correct_index_list = dict()
        for trail in range(1, num_trail+1):
            correct_index_list[str(trail)] = [] # initialize
            print(model)
            print(model_name)
            checkpoint = 'checkpoints/{}-{}-model{}/199.pt'.format(model_name, data_name, trail)
            model.load_state_dict(torch.load(checkpoint))
            print('Trail {}'.format(str(trail)))
            correct_index = test_correct(model, train_data_loader)
            correct_index_list[str(trail)] = correct_index
        
        # aggregate over different trails
        df = pd.DataFrame.from_dict(correct_index_list)
        correct_times = list(df.agg("sum", axis="columns"))

        if vis:
            # visualize histogram of number of trails that predict this data correctly
            plt.figure(figsize=(10,10))
            plt.bar(range(1, num_trail+1), [np.sum(np.asarray(correct_times)==i) for i in range(1, num_trail+1)])
            plt.show()

        if save:
            # save positive index and negative index or not
            # need to define threshold for what is considered as positive and negative
            pos_index = np.where(np.asarray(correct_times) >= pos_thre)[0]
            neg_index = np.where(np.asarray(correct_times) <= neg_thre)[0]

            np.save('./datasets/{}_train_pos_index_{}'.format(data_name, model_name), pos_index)
            np.save('./datasets/{}_train_neg_index_{}'.format(data_name, model_name), neg_index)
        
    # create loader for positive samples and negative samples
    pos_index = np.load('./datasets/{}_train_pos_index_{}.npy'.format(data_name, model_name))
    neg_index = np.load('./datasets/{}_train_neg_index_{}.npy'.format(data_name, model_name))

    train_data_loader_pos = data_loader(batch_size=1,  # batch size must be 1
                                        dataset_name = data_name, 
                                        subsample_id=pos_index.tolist(), 
                                        train=True,
                                        shuffle=False) # shuffle should be disabled

    train_data_loader_neg = data_loader(batch_size=1, 
                                        dataset_name = data_name, 
                                        subsample_id=neg_index.tolist(), 
                                        train=True,
                                        shuffle=False) # shuffle should be disabled
    
    print('Number of postive samples: ', len(train_data_loader_pos))
    print('Number of negative samples: ', len(train_data_loader_neg))
    
    
    # get average gradient for pos samples and neg samples
    trail = 1 #FIXME: use trail 1 to compute gradients
    checkpoint = 'checkpoints/{}-{}-model{}/199.pt'.format(model_name, data_name, str(1))
    model.load_state_dict(torch.load(checkpoint))
    optimizer = optim.Adam(model.parameters(), lr=0.001) # this optimizer is dummy
    print('Use trail {} to compute conflicting gradients'.format(str(trail)))

    pos_grad_dict = record_grad(model, train_data_loader_pos, 
                                criterion=nn.CrossEntropyLoss(reduction='sum'), 
                                optimizer=optimizer) # should use reduction method to be sum


    neg_grad_dict = record_grad(model, train_data_loader_neg, 
                                criterion=nn.CrossEntropyLoss(reduction='sum'), 
                                optimizer=optimizer)
    
    return pos_grad_dict, neg_grad_dict


def get_neighbor_graddict(model_name,
                          neighbor_model,
                          neighbor_model_name,
                          data_name,
                          train_data_loader
                          ):
    '''
    Get gradient contradiction dictionary for neighbor model on those contradicted samples for anchor model
    :param model_name: anchor model name
    :param neighbor_model: initialized pytorch model
    :param neighbor_model_name: neighbor model name
    :param data_name: name for dataset
    :param train_data_loader: dataloader
    '''
    # create loader for positive samples and negative samples
    pos_index = np.load('./datasets/{}_train_pos_index_{}.npy'.format(data_name, model_name))
    neg_index = np.load('./datasets/{}_train_neg_index_{}.npy'.format(data_name, model_name))

    train_data_loader_pos = data_loader(batch_size=1,  # batch size must be 1
                                        dataset_name = data_name, 
                                        subsample_id=pos_index.tolist(), 
                                        train=True,
                                        shuffle=False) # shuffle should be disabled

    train_data_loader_neg = data_loader(batch_size=1, 
                                        dataset_name = data_name, 
                                        subsample_id=neg_index.tolist(), 
                                        train=True,
                                        shuffle=False) # shuffle should be disabled
    
    print('Number of postive samples: ', len(train_data_loader_pos))
    print('Number of negative samples: ', len(train_data_loader_neg))
    
    
    # get average gradient for pos samples and neg samples
    trail = 1 # FIXME: use trail 1 to compute gradients
    checkpoint = 'checkpoints/{}-{}-model{}/199.pt'.format(neighbor_model_name, data_name, trail)
    neighbor_model.load_state_dict(torch.load(checkpoint))
    optimizer = optim.Adam(neighbor_model.parameters(), lr=0.001) # this optimizer is dummy
    print('Use trail {} to compute conflicting gradients'.format(str(trail)))

    pos_grad_dict = record_grad(neighbor_model, 
                                train_data_loader_pos, 
                                criterion=nn.CrossEntropyLoss(reduction='sum'), 
                                optimizer=optimizer) # should use reduction method to be sum


    neg_grad_dict = record_grad(neighbor_model, 
                                train_data_loader_neg, 
                                criterion=nn.CrossEntropyLoss(reduction='sum'), 
                                optimizer=optimizer)

    return pos_grad_dict, neg_grad_dict




def record_grad(model, data_loader, criterion, optimizer):
    '''
    Record average gradient for all weights on a given dataloader
    :param model: initialized pytorch model
    :param data_loader: dataloader
    :param criterion: loss function, should set reduction='sum'
    :param optimizer: optimizer is used to zero-out gradients before backward propagation
    '''
    grad_dict = dict()
    model.train() # enable gradient flow
    data_ct = 0
    for data in tqdm(data_loader):
        img, target = data
        img = img.to(device)
        target = target.to(device)
        data_ct += img.shape[0]

        output = model.forward(img)
        
        # zero the parameter gradients
        optimizer.zero_grad()
        loss = criterion(output, target)
        loss.backward()
        
        for name, param in model.named_parameters():
            grad = param.grad.detach().clone().cpu()
            if 'weight' in name:
                if name not in grad_dict.keys():
                    grad_dict[name] = grad.mean(dim=tuple(range(1, len(grad.shape)))) # average over each cnn filter
                else:
                    grad_dict[name] = torch.add(grad_dict[name], grad.mean(dim=tuple(range(1, len(grad.shape))))) # sum the gradients over all samples
            
    model.eval() 
    # zero the parameter gradients
    optimizer.zero_grad()
    print('Length of data', data_ct)

    # get average grad_dict
    for k in grad_dict.keys():
        grad_dict[k] = torch.div(grad_dict[k], data_ct)
   
    return grad_dict



# def record_allgrad(model, data_loader, criterion, optimizer):
#     '''
#     Record all gradient for all weights on a given dataloader
#     :param model: initialized pytorch model
#     :param data_loader: dataloader
#     :param criterion: loss function, should set reduction='sum'
#     :param optimizer: optimizer is used to zero-out gradients before backward propagation
#     '''
#     grad_dict = dict()
#     model.train() # enable gradient flow
#     data_ct = 0
#     for data in tqdm(data_loader):
#         img, target = data
#         img = img.to(device)
#         target = target.to(device)
#         data_ct += img.shape[0]

#         output = model.forward(img)
        
#         # zero the parameter gradients
#         optimizer.zero_grad()
#         loss = criterion(output, target)
#         loss.backward()
        
#         for name, param in model.named_parameters():
#             grad = param.grad.detach().clone().cpu()
#             if 'weight' in name:
#                 if name not in grad_dict.keys():
#                     grad_dict[name] = [grad.mean(dim=tuple(range(1, len(grad.shape))))]
#                 else:
#                     grad_dict[name].append(grad.mean(dim=tuple(range(1, len(grad.shape)))))
            
#     model.eval() 
#     # zero the parameter gradients
#     optimizer.zero_grad()
#     print('Length of data', data_ct)
    
#     # get average grad_dict
#     for k in grad_dict.keys():
#         grad_dict[k] = np.asarray([x.numpy() for x in grad_dict[k]])
   
#     return grad_dict


# def record_clustergrad(model, data_loader, criterion, optimizer):
#     '''
#     Record gradient clusters given a data_loader
#     :param model: initialized pytorch model
#     :param data_loader: dataloader
#     :param criterion: loss function, should set reduction='sum'
#     :param optimizer: optimizer is used to zero-out gradients before backward propagation
#     '''
#     grad_dict = dict()
#     model.train() # enable gradient flow
#     data_ct = 0
#     for data in tqdm(data_loader):
#         img, target = data
#         img = img.to(device)
#         target = target.to(device)
#         data_ct += img.shape[0]

#         output = model.forward(img)
        
#         # zero the parameter gradients
#         optimizer.zero_grad()
#         loss = criterion(output, target)
#         loss.backward()
        
#         for name, param in model.named_parameters():
#             grad = param.grad.detach().clone().cpu()
#             if 'weight' in name:
#                 if name not in grad_dict.keys():
#                     grad_dict[name] = [grad.mean(dim=tuple(range(1, len(grad.shape))))]
#                 else:
#                     grad_dict[name].append(grad.mean(dim=tuple(range(1, len(grad.shape)))))
            
#     model.eval() 
#     # zero the parameter gradients
#     optimizer.zero_grad()
#     print('Length of data', data_ct)
    
#     print('Start clustering')
#     clustered_grad = dict()
#     for key in grad_dict.keys():
#         if 'weight' in key:
#             data = np.asarray([x.numpy() for x in grad_dict[key]])
#             kmeans = KMeans(init="random", n_clusters=20, random_state=42).fit(data)
#             represent_data = kmeans.cluster_centers_
# #             data_dist = distance.cdist(data, data, 'cosine')
# #             print(data_dist.shape)
# #             data_distavg = data_dist.mean(axis=1)
# #             represent_data = data[np.argsort(data_distavg)][:10]
#             clustered_grad[key] = represent_data
        
#     print('End clustering')
   
#     return clustered_grad