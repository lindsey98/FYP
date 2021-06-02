from tqdm import tqdm
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def record_grad(model, data_loader, criterion, optimizer):
    '''
    Record average gradient for all weights on a given dataloader
    '''
    print('Length of dataloader', len(data_loader))
    grad_dict = dict()
    model.train() # enable gradient flow
    for data in tqdm(data_loader):
        img, target = data
        img = img.to(device)
        target = target.to(device)

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
    
    # get average grad_dict
    for k in grad_dict.keys():
        grad_dict[k] = torch.div(grad_dict[k], len(data_loader))
   
    return grad_dict