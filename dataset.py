import os
import numpy as np
from PIL import Image
from skimage.color import rgb2lab

import torch
from torchvision import transforms
from torch.utils.data import Dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


colored_imgs_folder = 'coloured'
original_imgs_folder = 'grayscale'
class ColorizationDataset(Dataset):
    def __init__(self, paths, mode='lightness', split='train'):
        if split == 'train':
            self.transforms = transforms.Compose([
                transforms.Resize((256, 256),  Image.BICUBIC),
                transforms.RandomHorizontalFlip(p=0.2),
            ])
        elif split == 'test':
            self.transforms = transforms.Resize((256, 256),  Image.BICUBIC)
        
        self.split = split
        self.size = 256
        self.paths = paths
        self.mode = mode
    
    def __getitem__(self, idx):
        img = Image.open(os.path.join(colored_imgs_folder, self.paths[idx])).convert("RGB")
        img = self.transforms(img)
        img = np.array(img)
        img_lab = rgb2lab(img).astype("float32")
        img_lab = transforms.ToTensor()(img_lab)
        if self.mode == 'lightness':
            L = img_lab[[0], ...] / 50. - 1. 
        elif self.mode == 'original':
            original_img = Image.open(os.path.join(original_imgs_folder, self.paths[idx]))
            original_img = self.transforms(original_img)
            original_img = np.array(original_img)
            original_img = transforms.ToTensor()(original_img)[:1] * 2. - 1.
            L = original_img
        else:
            raise NotImplementedError
        ab = img_lab[[1, 2], ...] / 110. 
        
        return {'L': L, 'ab': ab}
    
    def __len__(self):
        return len(self.paths)