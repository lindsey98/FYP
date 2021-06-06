from tqdm import tqdm
import torch
from src.train_test import test_correct
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from torch import optim
from torch import nn
from src.dataset import data_loader

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_graddict(model,
                 model_name, data_name,
                 train_data_loader, num_trail, 
                 pos_thre, neg_thre,
                 compute_index=True, 
                 vis=True, save=True):
    
    # if we need to compute correct index
    if compute_index:
        print('Compute correctly predicted data indices...')
        # get correctly predicted index
        correct_index_list = dict()
        for trail in range(1, num_trail+1):
            correct_index_list[str(trail)] = [] # initialize

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
    trail = 1
    checkpoint = 'checkpoints/{}-{}-model{}/199.pt'.format(model_name, data_name, trail)
    model.load_state_dict(torch.load(checkpoint))
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5) # this optimizer is dummy
    print('Use trail {} to compute conflicting gradients'.format(str(trail)))

    pos_grad_dict = record_grad(model, train_data_loader_pos, 
                                criterion=nn.CrossEntropyLoss(reduction='sum'), 
                                optimizer=optimizer) # should use reduction method to be sum


    neg_grad_dict = record_grad(model, train_data_loader_neg, 
                                criterion=nn.CrossEntropyLoss(reduction='sum'), 
                                optimizer=optimizer)
    
    return pos_grad_dict, neg_grad_dict



def record_grad(model, data_loader, criterion, optimizer):
    '''
    Record average gradient for all weights on a given dataloader
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
            if name not in grad_dict.keys():
                grad_dict[name] = grad
            else:
                grad_dict[name] = torch.add(grad_dict[name], grad) # sum the gradients over all samples
            
    model.eval() 
    # zero the parameter gradients
    optimizer.zero_grad()
    print('Length of data', data_ct)

    # get average grad_dict
    for k in grad_dict.keys():
        grad_dict[k] = torch.div(grad_dict[k], data_ct)
   
    return grad_dict