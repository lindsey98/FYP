import random

import torch
import torch.nn.functional as F
import os

from siamese.NewSiamese import load_model, process_image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def l2_norm(x):
    if len(x.shape):
        x = x.reshape((x.shape[0],-1))
    return F.normalize(x, p=2, dim=1)


def pred(img_path, model, imshow=False, title=None, path=True, grayscale=True):
    img = process_image(img_path, imshow=imshow, title=title, path=path, grayscale=grayscale)

    with torch.no_grad():
        logo_feat = model.features(img)
        logo_feat = l2_norm(logo_feat).squeeze(0).cpu().numpy()
    return logo_feat


if __name__ == "__main__":
    data_path = './data/classification_data/'
    image_map = dict()

    classes = 180
    model_name = './rgb_ar.pth'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(classes, model_name)
    model.to(device)
    model.eval()

    bad_feature_map = dict()
    good_feature_map = dict()
    FN = 0

    for a, b, c in os.walk(data_path):
        for i in range(0, len(b)):
            if not b[i].__contains__("."):
                image_dic = data_path + b[i]

                for j in os.listdir(image_dic):
                    if j.split(".")[0].isdecimal():
                        image_path = image_dic + "/" + j
                        if image_map.get(i) is None:
                            image_map[i] = [image_path]
                        else:
                            image_map[i].append(image_path)

        for i in range(1000):
            print(i)
            img_list_1 = image_map.get(random.randrange(0, len(image_map)))
            if img_list_1 is not None:
                img_s_path = img_list_1[random.randrange(0, len(img_list_1))]
                img_o_path = img_list_1[random.randrange(0, len(img_list_1))]

                input_o = process_image(img_o_path, grayscale=False)
                input_s = process_image(img_s_path, grayscale=False)

                output_o = l2_norm(model.features(input_o)).squeeze(0)
                output_s = l2_norm(model.features(input_s)).squeeze(0)
                similarity = output_o.dot(output_s).cpu().detach().numpy()
                if similarity > 0.8:
                    continue
                    # lst = list()
                    # for j in range(2048):
                    #     lst.append((j, output_o[j].item() * output_s[j].item()))
                    # lst.sort(key=lambda x: x[1], reverse=True)
                    #
                    # for j in range(128):
                    #     if good_feature_map.get(lst[j][0]) is None:
                    #         good_feature_map[lst[j][0]] = 1
                    #     else:
                    #         good_feature_map[lst[j][0]] += 1
                elif similarity < 0.7:
                    FN += 1
                    lst = list()
                    for j in range(2048):
                        lst.append((j, output_o[j].item() * output_s[j].item()))
                    lst.sort(key=lambda x: x[1], reverse=False)

                    idx = 0
                    j = 0
                    while j < 256:
                        if bad_feature_map.get(lst[idx][0]) is None:
                            bad_feature_map[lst[idx][0]] = 1
                        else:
                            bad_feature_map[lst[idx][0]] += 1
                        if lst[idx][1] != 0:
                            j += 1
                        idx += 1

                    for j in range(256):
                        if good_feature_map.get(lst[-j][0]) is None:
                            good_feature_map[lst[-j][0]] = 1
                        else:
                            good_feature_map[lst[-j][0]] += 1
        for i in range(0):
            print(i)
            img_list_1 = image_map.get(random.randrange(0, len(image_map)))
            if img_list_1 is not None:
                img_s_path = img_list_1[random.randrange(0, len(img_list_1))]

                img_list_2 = image_map.get(random.randrange(0, len(image_map)))
                if img_list_2 is not None:
                    img_o_path = img_list_2[random.randrange(0, len(img_list_2))]

                    input_o = process_image(img_o_path, grayscale=False)
                    input_s = process_image(img_s_path, grayscale=False)

                    output_o = l2_norm(model.features(input_o)).squeeze(0)
                    output_s = l2_norm(model.features(input_s)).squeeze(0)
                    similarity = output_o.dot(output_s).cpu().detach().numpy()
                    if similarity > 0.8:
                        lst = list()
                        for j in range(2048):
                            lst.append((j, output_o[j].item() * output_s[j].item()))
                        lst.sort(key=lambda x: x[1], reverse=True)

                        for j in range(128):
                            if bad_feature_map.get(lst[j][0]) is None:
                                bad_feature_map[lst[j][0]] = 1
                            else:
                                bad_feature_map[lst[j][0]] += 1
                    elif similarity < 0.5:
                        lst = list()
                        for j in range(2048):
                            lst.append((j, output_o[j].item() * output_s[j].item()))
                        lst.sort(key=lambda x: x[1], reverse=False)

                        idx = 0
                        j = 0
                        while j < 128:
                            if good_feature_map.get(lst[idx][0]) is None:
                                good_feature_map[lst[idx][0]] = 1
                            else:
                                good_feature_map[lst[idx][0]] += 1
                            if lst[idx][1] != 0:
                                j += 1
                            idx += 1

        fp1 = open("./good_feature.csv", "w")
        for i in range(2048):
            if good_feature_map.get(i) is not None:
                fp1.write("%d, %d\n" % (i, good_feature_map.get(i)))
            else:
                fp1.write("%d, %d\n" % (i, 0))

        fp2 = open("./bad_feature.csv", "w")
        for i in range(2048):
            if bad_feature_map.get(i) is not None:
                fp2.write("%d, %d\n" % (i, bad_feature_map.get(i)))
            else:
                fp2.write("%d, %d\n" % (i, 0))

        print("FN ", FN)
        break
