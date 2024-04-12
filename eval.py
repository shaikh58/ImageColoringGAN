import numpy as np
import PIL
import os
from matplotlib import pyplot as plt

import torch
from torchvision import transforms
from skimage.color import rgb2lab

from models import MainModel, build_res_unet
from utils import lab_to_rgb

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def eval(args):
    if args['use_pretrained_G']:
        net_G = build_res_unet(n_input=1, n_output=2, size=256)
        net_G.load_state_dict(torch.load("res18-unet.pt", map_location=device))
        model = MainModel(net_G=net_G)   
    else:     
        model = MainModel()
    trained_model_weights = torch.load(args['model_path'])
    model.load_state_dict(trained_model_weights)
    model = model.to(device)
    for img_file in os.listdir(args['test_imgs_dir']):
        path = os.path.join(args['test_imgs_dir'],img_file)
        img = PIL.Image.open(path)
        img = img.resize((256, 256))
        img = np.array(img)
        if 'ground_truth' in args['test_imgs_dir']:
            img_lab = rgb2lab(img).astype("float32") 
            img_lab = transforms.ToTensor()(img_lab)
            img = img_lab[[0], ...] / 50. - 1. 
        else:
            img = transforms.ToTensor()(img)[:1] * 2. - 1.
        model.eval()
        with torch.no_grad():
            preds = model.get_color(img.unsqueeze(0).to(device))
        colorized = lab_to_rgb(img.unsqueeze(0), preds.cpu())[0]
        plt.imshow(colorized)
        plt.savefig('colorized_{}'.format(img_file))


if __name__ == "__main__":
    args = {
        'use_pretrained_G': False,
        'test_imgs_dir':'test_imgs/compare',
        'model_path':'trained_model.pt',
    }
    eval(args)