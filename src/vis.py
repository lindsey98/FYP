
from src.train_test import test
import numpy as np
import matplotlib.pyplot as plt
import torch

def plot_training_acc( model, train_data_loader,
                      model_name, data_name, total_trails,):

    # training acc
    training_acc_list = []
    for trail in range(1, total_trails+1):
        checkpoint = 'checkpoints/{}-{}-model{}/999.pt'.format(model_name, data_name, trail)
        model.load_state_dict(torch.load(checkpoint))
        print('Trail {}'.format(str(trail)))
        train_acc = test(model, train_data_loader, criterion=torch.nn.CrossEntropyLoss(reduction='sum'))
        training_acc_list.append(train_acc)
    
    # plot training acc
    plt.bar(range(1, total_trails+1), height=training_acc_list)
    plt.ylim(bottom=70, top=77)
    plt.axhline(y=np.mean(training_acc_list), color='r', linestyle='-')
    plt.xlabel('Training model index')
    plt.ylabel('Training Acc')
    plt.show()
    
    print('Average training acc: {}'.format(np.mean(training_acc_list)))
    
    return training_acc_list


def weight_contradict(pos_grad_dict, neg_grad_dict, method='sign'):
    
    color_dict = [str(i / 10) for i in reversed(range(0, 10))]

    plt.rcParams['figure.figsize'] = (12.0, 12.0)
    ax = plt.gca()
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.spines['left'].set_color('none')
    ax.spines['bottom'].set_color('none')
    ax.set_xticks([])
    ax.set_yticks([])

    start_x = 0
    for name, grad_pos in pos_grad_dict.items():
        if 'weight' in name and 'conv' in name: # only visualize conv weights, no bias
            grad_neg = neg_grad_dict[name]
            
            if method == 'sign':
                conflict_grad = (torch.sign(grad_pos) != torch.sign(grad_neg)).int().sum(dim=(2, 3))
                in_channel = conflict_grad.size(1) # dimension 1 is input channel size
                output_channel = conflict_grad.size(0) # dimension 0 is output channel size

                for i in range(output_channel):
                    for j in range(in_channel):
                        color = color_dict[conflict_grad[i][j]]
                        plt.plot([start_x, start_x + 1], [j, i], color=color)
                start_x += 1
                
            elif method == 'level':
                conflict_level = (torch.sign(grad_pos) != torch.sign(grad_neg)) * (torch.abs(grad_pos - grad_neg))
                in_channel = conflict_level.size(1)
                output_channel = conflict_level.size(0)

                conflict_level = conflict_level.sum(dim=(2, 3)) # sum over kernel size
                print(conflict_level)
                conflict_level /= conflict_level.max() # rescale it 
                
                for i in range(output_channel):
                    for j in range(in_channel):
                        color = str(1 - conflict_level[i][j].item())
                        plt.plot([start_x, start_x + 1], [j, i], color=color)
                start_x += 1
                
            else:
                raise NotImplementError
                

    plt.title('Weight contradiction {} visualization'.format(method))
    plt.show()
