from torch.autograd import Variable
from torch import optim, save, load

import torch.nn.functional as F
import torch

from ResidualLoss.dataset import *
from ResidualLoss.model import *
import argparse
from tqdm import tqdm
import os
# os.environ["CUDA_VISIBLE_DEVICES"]="0"

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def train(model, 
          data_name,
          model_name,
          trail,
          train_data_loader, test_data_loader, 
          criterion, optimizer):
    
    '''
    Main model training logic
    '''
    
    length = len(train_data_loader.dataset)
    print('length of training data', length)
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
        print('epoch [{}/{}], loss:{:.4f} Accuracy: {}/{}'.format(epoch + 1, num_epochs, total_train_loss, total_correct, length))
        test_acc = test(model, test_data_loader, criterion)
        
        # save model
        os.makedirs('./checkpoints/{}-{}-model{}/'.format(data_name, model_name, str(trail)), exist_ok=True)
        location = './checkpoints/{}-{}-model{}/'.format(data_name, model_name, str(trail)) + str(epoch) + '.pt'
        save(model.state_dict(), location)


def test(model, test_data_loader, criterion):
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

    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_data_loader.dataset),
        100. * correct / len(test_data_loader.dataset)))
    
    return 100. * correct / len(test_data_loader.dataset)



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
    parser.add_argument('--epochs', type=int, default=1000, help='number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--trail', type=int, default=1, help='index of trained model')
    parser.add_argument('--retrain', type=bool, default=False, help='normal training or model retraining')
    parser.add_argument('--weights', type=str, default='.', help='pretrained weights')
    
    
    args = parser.parse_args()
    
    num_epochs = args.epochs
    batch_size = args.batch_size
    learning_rate = args.lr
    model_name = args.model_name
    dataset = args.data_name
    trail = args.trail
   
    
    # load model
    model = KNOWN_MODELS[model_name]()
    model = model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss()
    
    if dataset == "CIFAR10":
        train_data_loader = cifar10_data_loader_train(batch_size)
        test_data_loader = cifar10_data_loader_test(batch_size)
    elif dataset == "MNIST":
        train_data_loader = minst_data_loader_train(batch_size)
        test_data_loader = minst_data_loader_test(batch_size)   
    else:
        raise NotImplementError
    
    if args.retrain == True:
        # load pretrained model
        model.load_from(args.weights)
        
    # train ! 
    train(model, 
          model_name,
          dataset, 
          trail,
          train_data_loader, test_data_loader, 
          criterion, optimizer)