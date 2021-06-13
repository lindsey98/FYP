from torch.autograd import Variable
from torch import optim, save, load

import torch.nn.functional as F
import torch

from src.dataset import *
from src.model import *
from src.train_test import *
from src.vis import *
import argparse
from tqdm import tqdm
import os
import numpy as np
import logging
import torch.multiprocessing as mp
import time
torch.multiprocessing.set_start_method('spawn', force=True)# good solution !!!!

device = 'cuda' if torch.cuda.is_available() else 'cpu'

    

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Train underfitting model')
    parser.add_argument('--model_name', type=str, required=True, 
                        help='base model (already been trained) to start with')
    parser.add_argument('--data_name', type=str, required=True,
                        help='dataset you want to train')
    
    parser.add_argument('--batch_size', type=int, default=256, help='batch size')
    parser.add_argument('--epochs', type=int, default=200, help='number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    
    parser.add_argument('--num_trail', type=int, default=5, help='number of trails for each model')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints', help='directory for pretrained weights')
    parser.add_argument('--budget', type=str, default=50, help='budget to increase kernel filters')

    args = parser.parse_args()
    
    num_epochs = args.epochs
    batch_size = args.batch_size
    learning_rate = args.lr
    model_name = args.model_name
    dataset = args.data_name
    trail = args.num_trail
    
    # create logger
#     if os.path.exists('log/train_trace.log'):
#         os.unlink('log/train_trace.log') # delete old log
    logging.basicConfig(filename='log/train_trace.log', level=logging.INFO)
    logger = logging.getLogger('trace')

    
    # create dataloader
    logger.info("Start loading data")
    train_data_loader = data_loader(dataset_name = dataset, 
                                    batch_size = batch_size, 
                                    train=True)
    test_data_loader = data_loader(dataset_name = dataset, 
                                    batch_size = batch_size, 
                                    train=False) 
    logger.info("Finish loading data")

    # train ! 
    round_ = 0
    while True:
        start_time = time.time()
        # current index of model
        last_model_i, last_model_j, last_model_k = int(model_name.split('add')[1][0:2]), \
                                                   int(model_name.split('add')[1][2:4]), \
                                                   int(model_name.split('add')[1][4:6]) 
        
        # train all its neighbors
        neighbour_dict = [[1,0,0], [0,1,0], [0,0,1]]
        neighbour_names = []
        neighbour_acc = []
        neighbour_loss = []
        
        for neighbor in neighbour_dict:
            cur_model_i = last_model_i + neighbor[0]
            cur_model_j = last_model_j + neighbor[1]
            cur_model_k = last_model_k + neighbor[2]
            
            # ensure naming consistency
            if len(str(cur_model_i)) == 1:
                cur_model_i = '0'+str(cur_model_i)
            else:
                cur_model_i = str(cur_model_i)
                
            if len(str(cur_model_j)) == 1:
                cur_model_j = '0'+str(cur_model_j)   
            else:
                cur_model_j = str(cur_model_j)
                
            if len(str(cur_model_k)) == 1:
                cur_model_k = '0'+str(cur_model_k) 
            else:
                cur_model_k = str(cur_model_k)
                
            cur_model_name = model_name.split('add')[0]+ 'add' + str(cur_model_i) + str(cur_model_j) + str(cur_model_k)
            neighbour_names.append(cur_model_name)
            logger.info('Start training {}'.format(cur_model_name))
            
            # load model
            model = KNOWN_MODELS[cur_model_name]
            model = model.to(device)

            optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
            criterion = nn.CrossEntropyLoss(reduction='sum')
                
            # multiprocessing
            num_processes = trail
            logger.info('Training for in total {} trails'.format(trail))
            
            processes = []
            for rank in range(1, num_processes+1):
                # training each neighbor for 5 times in parallel!
                p = mp.Process(target=train, args=(model, 
                                              cur_model_name,
                                              dataset, 
                                              rank,
                                              train_data_loader, test_data_loader, 
                                              criterion, optimizer,
                                              num_epochs,
                                              logger))
                p.start()
                processes.append(p)
            for p in processes:
                p.join()

#             for rank in range(1, trail+1):
#                 train(model, 
#                       model_name,
#                       dataset, 
#                       rank,
#                       train_data_loader, test_data_loader, 
#                       criterion, optimizer,
#                       num_epochs,
#                       logger)

            # get average training acc over trails
            train_acc = np.mean(plot_training_acc(model, 
                                                  train_data_loader, 
                                                  model_name=cur_model_name, 
                                                  data_name=dataset, 
                                                  total_trails=trail, 
                                                  logger=logger, 
                                                  vis=False))
            
            logger.info('Average training acc {:.4f}'.format(train_acc))
            neighbour_acc.append(train_acc)
            
            # get average training loss over trails
            train_loss = np.mean(plot_training_loss(model, 
                                          train_data_loader, 
                                          model_name=cur_model_name, 
                                          data_name=dataset, 
                                          total_trails=trail, 
                                          logger=logger, 
                                          vis=False))
            
            logger.info('Average training loss {:.4f}'.format(train_loss))
            neighbour_loss.append(train_loss)            
        
        # compute best neighbor, update model
#         best_neigh = np.asarray(neighbour_names)[np.asarray(neighbour_acc) == max(neighbour_acc)][0]
        best_neigh = np.asarray(neighbour_names)[np.asarray(neighbour_loss) == min(neighbour_loss)][0]
        model_name = best_neigh
        logger.info('Update model name as {}'.format(model_name))
        
        round_ += 1
        logger.info('Current round {}'.format(str(round_)))
        # stop when reaching budget
        if round_ >= args.budget:
            break
        
        end_time = time.time() - start_time
        logger.info('One round finished, taken {:.4f} seconds'.format(end_time))