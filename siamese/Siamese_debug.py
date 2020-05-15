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


def l2_norm(x):
    if len(x.shape):
        x = x.reshape((x.shape[0],-1))
    return F.normalize(x, p=2, dim=1)


class MarginNet(nn.Module):
    def __init__(self, base_net, emb_dim, batch_k, normalize=False):
        super().__init__()
        self.conv = nn.Conv2d(1, 3, kernel_size=(1, 1))
        self.base_net = base_net
        in_dim = base_net.fc.out_features
        self.emb_dim = emb_dim
        self.dense1 = nn.Linear(in_dim, emb_dim)
        self.normalize = l2_norm

    def forward(self, x):
        x = self.conv(x)
        x = self.base_net(x)  # renet18 embedding
        x = self.dense1(x)

        x = self.normalize(x)
        return x


def procees_image(img_path, imshow=False, title=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        img = Image.open(os.path.join(img_path)).convert("L")
        img = ImageOps.expand(img, (
            (max(img.size) - img.size[0]) // 2, (max(img.size) - img.size[1]) // 2, (max(img.size) - img.size[0]) // 2,
            (max(img.size) - img.size[1]) // 2), fill=255)
        img = img.resize((100, 100))
        img = np.asarray(img)
        if imshow:
            plt.imshow(img, cmap='gray')
            plt.title(title)
            plt.show()
        to_tensor = transforms.ToTensor()
        img = to_tensor(img)
        img = img.unsqueeze(0)
        return img.to(device)


def pred(img_path, model, imshow=False, title=None):
    img = procees_image(img_path, imshow=False, title=None)
    img = Variable(img)
    logo_feat = model(img)

    logo_feat = logo_feat.squeeze(0)
    return logo_feat


def load_model():
    basenet = models.__dict__['resnet50'](pretrained=True)
    basenet.fc = nn.Linear(2048, 256)
    ct = 0
    for child in basenet.children():
        ct += 1
        if ct < 7:
            for param in child.parameters():
                param.requires_grad = False
    model = MarginNet(base_net=basenet, emb_dim=emb_dim, batch_k=batch_k)
    return model


if __name__ == "__main__":
    '''Configuration'''
    batch_k = 5
    emb_dim = 128
    os.sep = '/'
    model_name = './dws_checkpoint_gray_v6.pth.tar'
    pair_path = './similar_pair/0/' # store the protected logos for 182 brands

    '''Initialize model and load state dictionary'''
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model()
    if device == 'cpu':
        model.load_state_dict(torch.load(model_name, map_location='cpu'))
    else:
        model.load_state_dict(torch.load(model_name))
    model.to(device)
    # model.eval()

    '''feed screenshot and most similar logo into network and get the embeddings'''
    img_feat = pred(pair_path + 'cropped.png', model, imshow=False, title='Sampled Yolo box')
    most_sim_logo_feat = pred(pair_path + 'pred_target.png', model, imshow=False, title="Most similar logo")
    sim = most_sim_logo_feat.dot(img_feat)

    difference = img_feat - most_sim_logo_feat
    lst = list()
    for i in range(128):
        lst.append((i, abs(difference[i])))


    img_o = procees_image(pair_path + 'cropped.png', imshow=False, title=None)
    img_s = procees_image(pair_path + 'pred_target.png', imshow=False, title=None)

    model.train()
    img = Variable(img_s, requires_grad=True)
    logo_feat = model(img).squeeze(0)

    t = logo_feat.dot(img_feat)

    t.backward()

    grad = img.grad

    max = grad.view(-1).max()
    min = grad.view(-1).min()

    grad -= min
    grad /= (max - min)



    plt.matshow(grad.squeeze(0).squeeze(0).cpu())
    plt.title('similarity')
    plt.show()

# import cv2
# cv2.imwrite('./test1.png', grad)
#
# print(img_o.squeeze(0))

    # max = total.view(-1).max()
    # min = total.view(-1).min()
    #
    # total -= min
    # total /= (max - min)
    #
    # plt.matshow(total.squeeze(0).squeeze(0).cpu())
    # plt.show()
















