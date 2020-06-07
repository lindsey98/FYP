import cv2
import numpy as np
import torch
from torch.autograd import Function, Variable
import matplotlib.pyplot as plt

from siamese.Siamese import load_model, procees_image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class GuidedBackpropReLU(Function):
    @staticmethod
    def forward(self, input):
        positive_mask = (input > 0).type_as(input)
        output = torch.addcmul(torch.zeros(input.size()).type_as(input), input, positive_mask)
        self.save_for_backward(input, output)
        return output

    @staticmethod
    def backward(self, grad_output):
        input, output = self.saved_tensors

        positive_mask_1 = (input > 0).type_as(grad_output)
        positive_mask_2 = (grad_output > 0).type_as(grad_output)
        grad_input = torch.addcmul(torch.zeros(input.size()).type_as(input),
                                   torch.addcmul(torch.zeros(input.size()).type_as(input), grad_output,
                                                 positive_mask_1), positive_mask_2)

        return grad_input


class GuidedBackpropReLUModel:
    def __init__(self, model):
        self.model = model
        self.model = model.to(device)

        def recursive_relu_apply(module_top):
            for idx, module in module_top._modules.items():
                recursive_relu_apply(module)
                if module.__class__.__name__ == 'ReLU':
                    module_top._modules[idx] = GuidedBackpropReLU.apply

        # replace ReLU with GuidedBackpropReLU
        recursive_relu_apply(self.model)

    def forward(self, input):
        return self.model(input)


def deprocess_image(img):
    """ see https://github.com/jacobgil/keras-grad-cam/blob/master/grad-cam.py#L65 """
    img = img - np.mean(img)
    img = img / (np.std(img) + 1e-5)
    img = img * 0.1
    img = img + 0.5
    img = np.clip(img, 0, 1)
    return np.uint8(img * 255)


def generate_GB_result(gb_model, input_o, input_s, output_path):
    output_s = model.forward(input_s).squeeze(0)

    output_o = gb_model.forward(input_o).squeeze(0)

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
    model_name = '../siamese/dws_checkpoint_gray_v6.pth.tar'
    if device == 'cpu':
        state_dict = torch.load(model_name, map_location='cpu')
    else:
        state_dict = torch.load(model_name)

    pair_path = "../siamese/data/S_D/2/"

    img_s_path = pair_path + 'logo.png'
    img_o_path = pair_path + 'yolo_box.png'

    model = load_model()
    model.to(device)
    model.load_state_dict(state_dict)

    input_o = procees_image(img_o_path)
    input_s = procees_image(img_s_path)

    model.eval()
    gb_model = GuidedBackpropReLUModel(model=model)

    generate_GB_result(gb_model, input_o, input_s, "gb_from.png")

    generate_GB_result(gb_model, input_s, input_o, "gb_to.png")
