import cv2
import numpy as np
import torch

from grad_cam.GradCamModel import GradCamForNewSiamese
from siamese.NewSiamese import load_model, process_image
from siamese.NewSiamese_debug import l2_norm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def generate_grad_cam_result(grad_cam_model, input_o, input_s, output_path):
    features, output_o = grad_cam_model.extractor(input_o)
    output_o = output_o.squeeze(0)

    output_s = l2_norm(model.features(input_s)).squeeze(0)

    grad_cam_model.feature_module.zero_grad()
    grad_cam_model.model.zero_grad()

    one_hot = output_o.dot(output_s)

    print(one_hot)

    one_hot.backward(retain_graph=True)

    grads_val = grad_cam_model.extractor.get_gradients()[-1].cpu().data.numpy()

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
    cv2.imwrite(output_path, np.uint8(255 * cam))


if __name__ == '__main__':
    classes = 180
    model_name = '../siamese/rgb_ar.pth'  ## RGB model

    pair_path = "./data/SS/4/"

    img_s_path = pair_path + 'logo.png'
    img_o_path = pair_path + 'cropped.png'

    model = load_model(classes, model_name)
    model.to(device)
    model.eval()

    input_o = process_image(img_o_path)
    input_s = process_image(img_s_path)

    grad_cam_model = GradCamForNewSiamese(model=model, feature_module=model.body.block4, target_layer_names=["unit03"])

    generate_grad_cam_result(grad_cam_model, input_o, input_s, "grad_cam_from.png")
    generate_grad_cam_result(grad_cam_model, input_s, input_o, "grad_cam_to.png")
