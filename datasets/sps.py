# --------------------------------------------------
# Author: Youngwook Kwon
# Date: 2024-12-31
# --------------------------------------------------

import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

def voc_cmap(N=256, normalized=False):
    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)

    dtype = 'float32' if normalized else 'uint8'
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7-j)
            g = g | (bitget(c, 1) << 7-j)
            b = b | (bitget(c, 2) << 7-j)
            c = c >> 3

        cmap[i] = np.array([r, g, b])

    cmap = cmap/255 if normalized else cmap
    return cmap

class SPSegmentation(Dataset):
    cmap = voc_cmap()
    def __init__(
        self,
        root,
        image_set='train',
        transform=None, 
    ):
        super().__init__()
        self.root = os.path.expanduser(root)
        self.image_set = image_set
        self.transform = transform
        
        base_dir = "sps"
        voc_root = os.path.join(self.root, base_dir)
        if not os.path.isdir(voc_root):
            raise RuntimeError(f"Directory Error: {voc_root}")

        self.image_dir = os.path.join(voc_root, image_set + "_JPEGImages")
        self.mask_dir = os.path.join(voc_root, image_set + "_SegmentationClass")
        
        split_dir = os.path.join(voc_root, "ImageSets", "Segmentation")
        split_file = os.path.join(split_dir, f"{image_set}.txt")
        if not os.path.isfile(split_file):
            raise FileNotFoundError(f"{split_file} Files not exist.")

        with open(split_file, "r") as f:
            file_names = [x.strip() for x in f.readlines()]

        self.images = [os.path.join(self.image_dir, fn + ".jpg") for fn in file_names]
        self.masks = [os.path.join(self.mask_dir, fn + ".png") for fn in file_names]
        
    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = self.images[index]
        mask_path = self.masks[index]
        
        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path)

        mask_np = np.array(mask, dtype=np.uint8)
        mask_np[mask_np != 15] = 0
        mask_np[mask_np == 15] = 1

        mask = Image.fromarray(mask_np)
        if self.transform is not None:
            image, mask = self.transform(image, mask)

        return image, mask
    
    @classmethod
    def decode_target(cls, mask):
        """decode semantic mask to RGB image"""
        return cls.cmap[mask]