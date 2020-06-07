import os
import numpy as np
from PIL import Image, ImageOps
from torch.autograd import Variable
from torchvision import transforms
import matplotlib.pyplot as plt
import torch

from siamese.Siamese import load_model, process_image


def pred(img_path, model):
    img = process_image(img_path)
    img = Variable(img)
    logo_feat = model(img)

    logo_feat = logo_feat.squeeze(0)
    return logo_feat

def visualize_gradient_against_cosine():
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    '''feed screenshot and most similar logo into network and get the embeddings'''
    img_o_feat = pred(img_o_path, model)
    img_s_feat = pred(img_s_path, model)

    model.train()
    print(img_o_feat.dot(img_s_feat))

    img_o = process_image(img_o_path)
    img_s = process_image(img_s_path)

    # From src to dest
    model.train()
    img = Variable(img_o, requires_grad=True)

    img_feat = model(img).squeeze(0)

    loss = img_feat.dot(img_s_feat)
    loss.backward()
    grad = img.grad.abs()

    max = grad.view(-1).max()

    grad /= max

    plt.imshow(img_o.squeeze(0).squeeze(0).cpu(), cmap='gray')
    plt.title('processed source')
    plt.show()

    plt.imshow(grad.squeeze(0).squeeze(0).cpu(), cmap='rainbow')
    plt.title('gradient for source')
    plt.show()

    plt.imshow(img_o.squeeze(0).squeeze(0).cpu(), cmap='gray')
    plt.imshow(grad.squeeze(0).squeeze(0).cpu(), alpha=0.6, cmap='rainbow')
    plt.title('mixed for source')
    plt.show()

    model.load_state_dict(state_dict)
    img = Variable(img_s, requires_grad=True)

    img_feat = model(img).squeeze(0)

    loss = img_feat.dot(img_o_feat)
    loss.backward()
    grad = img.grad.abs()

    max = grad.view(-1).max()

    grad /= max

    plt.imshow(img_s.squeeze(0).squeeze(0).cpu(), cmap='gray')
    plt.title('processed dest')
    plt.show()

    plt.imshow(grad.squeeze(0).squeeze(0).cpu(), cmap='rainbow')
    plt.title('gradient for dest')
    plt.show()

    plt.imshow(img_s.squeeze(0).squeeze(0).cpu(), cmap='gray')
    plt.imshow(grad.squeeze(0).squeeze(0).cpu(), alpha=0.6, cmap='rainbow')
    plt.title('mixed for dest')
    plt.show()


if __name__ == "__main__":
    '''Configuration'''
    os.sep = '/'
    model_name = './dws_checkpoint_gray_v6.pth.tar'
    pair_path = './data/D_S/0/'

    '''Initialize model and load state dictionary'''
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model()

    print(model._modules.items())

    if device == 'cpu':
        state_dict = torch.load(model_name, map_location='cpu')
    else:
        state_dict = torch.load(model_name)

    img_o_path = pair_path + 'logo.png'
    img_s_path = pair_path + 'yolo_box.png'

    visualize_gradient_against_cosine()
















