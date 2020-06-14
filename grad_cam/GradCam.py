import argparse
import cv2
import numpy as np
import torch
from torch.autograd import Function
from torchvision import models

from grad_cam.GradCamModel import GradCam
from siamese.Siamese import load_model, process_image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':
    model_name = '../siamese/dws_checkpoint_gray_v6.pth.tar'
    if device == 'cpu':
        state_dict = torch.load(model_name, map_location='cpu')
    else:
        state_dict = torch.load(model_name)

    pair_path = "../siamese/data/D_D/0/"

    img_s_path = pair_path + 'logo.png'
    img_o_path = pair_path + 'yolo_box.png'

    model = load_model()
    model.to(device)
    model.load_state_dict(state_dict)
    model.eval()

    input_o = process_image(img_o_path)
    input_s = process_image(img_s_path)

    grad_cam = GradCam(model=model, feature_module=model.base_net.layer4, target_layer_names=["2"])

    features, output_o = grad_cam.extractor(input_o)
    output_o = output_o.squeeze(0)
    output_s = model.forward(input_s).squeeze(0)

    grad_cam.feature_module.zero_grad()
    grad_cam.model.zero_grad()

    one_hot = output_o.dot(output_s)

    print(one_hot)

    one_hot.backward(retain_graph=True)

    grads_val = grad_cam.extractor.get_gradients()[-1].cpu().data.numpy()

    target = features[-1]
    target = target.cpu().data.numpy()[0, :]

    weights = np.mean(grads_val, axis=(2, 3))[0, :]
    cam = np.zeros(target.shape[1:], dtype=np.float32)

    for i, w in enumerate(weights):
        cam += w * target[i, :, :]

    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, input_o.shape[2:])
    cam = cam - np.min(cam)

    cam = cam / np.max(cam)

    processed = input_o.detach().cpu().numpy().squeeze(0).transpose((1, 2, 0))

    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(processed)
    cam = cam / np.max(cam)
    cv2.imwrite("cam.jpg", np.uint8(255 * cam))
