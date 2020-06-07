import torch
import torch.nn.functional as F

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
    
    cropped_path = '../grad-cam/data/SD/0/cropped.png' ## change this
    logo_path = '../grad-cam/data/SD/0/logo.png' ## change this
    
    ### model configuration
    classes = 180
    model_name = './rgb_ar.pth' ## RGB model
    grayscale = False
    t_s = 0.8 ## threshold = 0.8
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(classes, model_name)
    model.to(device)
    model.eval()    
    
    ## prediction
    crop_feat = pred(cropped_path, model, imshow=True, title='Cropped image', path=True, grayscale=grayscale)
    logo_feat = pred(logo_path, model, imshow=True, title='protected logo', path=True, grayscale=grayscale)
    print('Similarity: %.5f'%crop_feat.dot(logo_feat), " while threshold is %.2f"%t_s)
