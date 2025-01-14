from torch.autograd import Variable
from torch import optim, save, load

import torch.nn.functional as F
import torch

from src.dataset import *
from src.model import *
import argparse
from tqdm import tqdm
import os
import logging
import torch.multiprocessing as mp
import time

torch.multiprocessing.set_start_method('spawn', force=True)# good solution !!!!
os.environ["CUDA_VISIBLE_DEVICES"]="1"

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def train(model, 
          model_name,
          data_name,
          trail,
          train_data_loader, test_data_loader, 
          criterion, optimizer,
          num_epochs,
          logger):
    
    '''
    Main model training logic
    '''
    
    length = len(train_data_loader.dataset)
    logger.info('length of training data {}'.format(str(length)))
    best_test_acc = 0
    
    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        total_correct = 0
        
        for data in tqdm(train_data_loader):
            img, target = data
            img = img.to(device)
            target = target.to(device)

            optimizer.zero_grad()
            output = model.forward(img)

            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            total_correct += pred.eq(target.view_as(pred)).sum().item()

        total_train_loss /= length
        test_acc = test(model, test_data_loader, criterion, logger)
        
        # save model
        os.makedirs('./checkpoints/{}-{}-model{}/'.format(model_name, data_name, str(trail)), exist_ok=True)
        if epoch == num_epochs - 1 or epoch % 50 == 0:
            logger.info('epoch [{}/{}], loss:{:.4f} Accuracy: {}/{}'.format(epoch + 1, num_epochs, total_train_loss, total_correct, length))
            location = './checkpoints/{}-{}-model{}/'.format( model_name, data_name, str(trail)) + str(epoch) + '.pt'
            logger.info("Save model in {} at epoch {}".format(location, str(epoch)))
            save(model.state_dict(), location)
            
            
    return model


def test(model, test_data_loader, criterion, logger):
    '''
    Get testing accuracy
    '''
    model.eval()
    test_loss = 0
    correct = 0
    
    with torch.no_grad():
        for data, target in tqdm(test_data_loader):
            data = data.to(device)
            target = target.to(device)
            output = model(data)
            
            loss = criterion(output, target)
            test_loss += loss.item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.cuda().view_as(pred)).sum().item()

    test_loss /= len(test_data_loader.dataset)
    
    if logger is not None:
        print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_data_loader.dataset),
            100. * correct / len(test_data_loader.dataset)))
    
    return 100. * correct / len(test_data_loader.dataset)


def test_loss(model, test_data_loader, criterion, logger):
    '''
    Get testing accuracy
    '''
    model.eval()
    test_loss = 0
    correct = 0
    
    with torch.no_grad():
        for data, target in tqdm(test_data_loader):
            data = data.to(device)
            target = target.to(device)
            output = model(data)
            
            loss = criterion(output, target)
            test_loss += loss.item()  # sum up batch loss

    test_loss /= len(test_data_loader.dataset)
    
    if logger is not None:
        logger.info('Test set: Average loss: {:.4f}\n'.format(test_loss))
    
    return test_loss

def test_correct(model, test_data_loader):
    '''
    Get correct prediction index
    '''
    
    model.eval()
    correct = []
    
    with torch.no_grad():
        for data, target in tqdm(test_data_loader):
            data = data.to(device)
            target = target.to(device)
            output = model(data)
            
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct_thisbatch = torch.flatten(pred.eq(target.cuda().view_as(pred)).detach().cpu()).numpy()
            correct.extend(correct_thisbatch)
            
    return correct
    
    
    

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Train underfitting model')
    parser.add_argument('--model_name', type=str, required=True,
                        help='model you want to train')
    parser.add_argument('--data_name', type=str, required=True,
                        help='dataset you want to train')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size')
    parser.add_argument('--epochs', type=int, default=200, help='number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--trail', type=int, default=5, help='index of trained model')
    parser.add_argument('--retrain', type=bool, default=False, help='normal training or model retraining')
    parser.add_argument('--weights', type=str, default='.', help='pretrained weights')
    
    
    args = parser.parse_args()
    
    num_epochs = args.epochs
    batch_size = args.batch_size
    learning_rate = args.lr
    model_name = args.model_name
    dataset = args.data_name
    trail = args.trail
   
            
    logging.basicConfig(filename='log/train_initial_{}.log'.format(model_name), level=logging.INFO)
    logger = logging.getLogger('trace')
    
    # load model
    logger.info("Loading model")
    model_i, model_j, model_k = int(model_name.split('add')[1][0:2]), \
                                int(model_name.split('add')[1][2:4]), \
                                int(model_name.split('add')[1][4:6]) 

    model = ChildModel(extra_filter=[model_i, model_j, model_k], parent_dict_path=None)
    print(model)
    
    logger.info(model)
    model = model.to(device)
    logger.info("Finish loading model")
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss(reduction='sum')
    
    # create dataloader
    logger.info("Start loading data")
    train_data_loader = data_loader(dataset_name = dataset, 
                                    batch_size = batch_size, 
                                    train=True)
    test_data_loader = data_loader(dataset_name = dataset, 
                                    batch_size = batch_size, 
                                    train=False)  
    logger.info("Finish loading data")
    
    if args.retrain == True:
        # load pretrained model
        model.load_from(args.weights)

    # train ! 
    processes = []
    for rank in range(1, trail+1):
        # training each neighbor for 5 times in parallel!
        p = mp.Process(target=train, args=(model, 
                                      model_name,
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
    
 