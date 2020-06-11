import cv2
import numpy as np
import torch
from torch.autograd import Function, Variable
import matplotlib.pyplot as plt

from grad_cam.GuidedBackPropReLUModel import deprocess_image, GuidedBackPropReLUModel
from siamese.NewSiamese import load_model, process_image
from siamese.NewSiamese_debug import l2_norm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def generate_GB_result(gb_model, input_o, input_s, output_path):
    output_s = l2_norm(model.features(input_s)).squeeze(0)

    output_o = l2_norm(gb_model.features(input_s)).squeeze(0)

    one_hot = output_o.dot(output_s)

    print(one_hot)

    one_hot.backward(retain_graph=True)

    output = input_o.grad.cpu().data.numpy()
    gb = output[0, :, :, :]

    gb = gb.transpose((1, 2, 0))

    gb = deprocess_image(gb)

    # cv2.imwrite(output_path, gb)
    #
    # input_processed = np.uint8(input_o.cpu().data.numpy()[0, :, :, :].transpose((1, 2, 0)) * 255)
    #
    # cv2.imwrite('raw_'+ output_path, input_processed)

    plt.imshow(input_o.squeeze(0).squeeze(0).cpu().detach(), cmap='gray')
    plt.title('processed source')
    plt.show()

    plt.imshow(gb.squeeze(2), cmap='rainbow')
    plt.title('gradient for source')
    plt.show()

    plt.imshow(input_o.squeeze(0).squeeze(0).cpu().detach(), cmap='gray')
    plt.imshow(gb.squeeze(2), alpha=0.6, cmap='rainbow')
    plt.title('mixed for source')
    plt.show()


if __name__ == '__main__':
    classes = 180
    model_name = '../siamese/rgb_ar.pth'  ## RGB model
    if device == 'cpu':
        state_dict = torch.load(model_name, map_location='cpu')
    else:
        state_dict = torch.load(model_name)

    pair_path = "./data/DD/2/"

    img_s_path = pair_path + 'logo.png'
    img_o_path = pair_path + 'cropped.png'

    model = load_model(classes, model_name)
    model.to(device)
    # model.load_state_dict(state_dict)

    input_o = process_image(img_o_path)
    input_s = process_image(img_s_path)

    model.eval()
    gb_model = GuidedBackPropReLUModel(model=model)

    generate_GB_result(gb_model, input_o, input_s, "gb_from.png")

    generate_GB_result(gb_model, input_s, input_o, "gb_to.png")
