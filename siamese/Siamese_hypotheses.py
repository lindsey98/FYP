import os
import numpy as np
from PIL import Image, ImageOps
from torch.autograd import Variable
from torchvision import transforms
import matplotlib.pyplot as plt
import torchvision.models as models
from torch import nn, autograd
import torch.nn.functional as F
import torch

from siamese.Siamese import load_model, procees_image


def pred(img_path, model, imshow=False, title=None):
    img = procees_image(img_path, imshow=imshow, title=title)
    img = Variable(img)
    logo_feat = model(img)

    logo_feat = logo_feat.squeeze(0)
    return logo_feat


def calculate_representitive_feature(img_o_path, img_s_path, feature_dict):
    '''feed screenshot and most similar logo into network and get the embeddings'''
    img_o_feat = pred(img_o_path, model, imshow=False, title='Sampled Yolo box')
    img_s_feat = pred(img_s_path, model, imshow=False, title="Most similar logo")

    # similarity = img_o_feat.dot(img_s_feat)

    lst = list()
    for i in range(128):
        lst.append((i, img_o_feat[i].item() * img_s_feat[i].item()))
    lst.sort(key=lambda x:x[1], reverse=True)

    for i in range(16):
        if feature_dict.get(lst[i][0]) is None:
            feature_dict[lst[i][0]] = 1
        else:
            feature_dict[lst[i][0]] += 1


if __name__ == "__main__":
    '''Configuration'''
    batch_k = 5
    emb_dim = 128
    os.sep = '/'
    model_name = './dws_checkpoint_gray_v6.pth.tar'

    '''Initialize model and load state dictionary'''
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model()

    if device == 'cpu':
        state_dict = torch.load(model_name, map_location='cpu')
    else:
        state_dict = torch.load(model_name)

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    feature_dict = dict()

    for i in range(32):
        pair_path = './data/D_S/' + str(i) + '/'
        img_o_path = pair_path + 'logo.png'
        img_s_path = pair_path + 'yolo_box.png'
        calculate_representitive_feature(img_o_path, img_s_path, feature_dict)

    feature_list = list(feature_dict.items())
    feature_list.sort(key=lambda x: x[1], reverse=True)
    print(feature_list)















