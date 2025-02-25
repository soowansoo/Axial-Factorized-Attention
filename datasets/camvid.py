import natsort
import os
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from PIL import Image
from .utils import *

color_dict = {
    'obj0' : [0, 128, 192], # Bicyclist
    'obj1' : [128, 0, 0], # Building
    'obj2' : [64, 0, 128], # Car
    'obj3' : [192, 192, 128], # Pole
    'obj4' : [64, 64, 128], # Fence
    'obj5' : [64, 64, 0], # Peestrian
    'obj6' : [128, 64, 128], # Road
    'obj7' : [0, 0, 192], # Pavement
    'obj8' : [192, 128, 128], # SignSymbol
    'obj9' : [128, 128, 128], # Sky
    'obj10' : [128, 128, 0] # Tree
}

class CamVid_Dataset():
  """CamVid <http://mi.eng.cam.ac.uk/research/projects/VideoRec/CamVid/, https://www.kaggle.com/datasets/carlolepelaars/camvid> Dataset.
    
    **Parameters:**
        - **root** (string): Root directory of dataset where directory 'train, val, test' and 'class_dict' are located.
        - **split** (string, optional): The image split to use, 'train', 'test' or 'val'
        - **transform** (callable, optional): A function/transform that takes in a PIL image and returns a transformed version. E.g, ``transforms.RandomCrop``
        - **target_transform** (callable, optional): A function/transform that takes in the target and transforms it.
    """
  def __init__(self, img_pth, mask_pth, transform):

    self.code2id, self.id2code, self.name2id, self.id2name = Color_map('/home/kws/data/camvid/Camvid/class_dict.csv')

    self.img_pth = img_pth
    self.mask_pth = mask_pth
    self.transform = transform
    all_imgs = os.listdir(self.img_pth)
    all_masks = [img_name[:-4] + '_L' + img_name[-4:] for img_name in all_imgs]
    self.total_imgs = natsort.natsorted(all_imgs)
    self.total_masks = natsort.natsorted(all_masks)
    
  def __len__(self):
    return len(self.total_imgs)

  def __getitem__(self, idx):
    img_loc = os.path.join(self.img_pth, self.total_imgs[idx])
    image = Image.open(img_loc).convert("RGB")
    mask_loc = os.path.join(self.mask_pth, self.total_masks[idx])
    mask = Image.open(mask_loc).convert("RGB")
    out_image, rgb_mask = self.transform(image, mask)
    out_mask = rgb_to_mask(torch.from_numpy(np.array(rgb_mask)), self.id2code)
    
    return out_image, out_mask#, rgb_mask.permute(0,1,2)
  
  def decode_segmap(cls, image):
    label_colours = np.array([
		color_dict['obj0'], color_dict['obj1'],
		color_dict['obj2'], color_dict['obj3'],
		color_dict['obj4'], color_dict['obj5'],
		color_dict['obj6'], color_dict['obj7'],
		color_dict['obj8'], color_dict['obj9'],
		color_dict['obj10']]).astype(np.uint8)
    r = np.zeros_like(image).astype(np.uint8)
    g = np.zeros_like(image).astype(np.uint8)
    b = np.zeros_like(image).astype(np.uint8)
    for l in range(0, 11):
        r[image == l] = label_colours[l, 0]
        g[image == l] = label_colours[l, 1]
        b[image == l] = label_colours[l, 2]
    rgb = np.zeros((image.shape[0], image.shape[1], 3)).astype(np.uint8)
    rgb[:, :, 0] = b
    rgb[:, :, 1] = g
    rgb[:, :, 2] = r
    return rgb

class Test():
  def __init__(self, img_pth, mask_pth, transform):
    self.img_pth = img_pth
    self.mask_pth = mask_pth
    self.transform = transform
    all_imgs = os.listdir(self.img_pth)
    all_masks = [img_name[:-4] + '_L' + img_name[-4:] for img_name in all_imgs]
    self.total_imgs = natsort.natsorted(all_imgs)
    self.total_masks = natsort.natsorted(all_masks)
  
  def __len__(self):
    return len(self.total_imgs)

  def __getitem__(self, idx):
    img_loc = os.path.join(self.img_pth, self.total_imgs[idx])
    image = Image.open(img_loc).convert("RGB")
    out_image = self.transform(image)

    mask_loc = os.path.join(self.mask_pth, self.total_masks[idx])
    mask = Image.open(mask_loc).convert("RGB")
    rgb_mask = self.transform(mask)
    
    return out_image, rgb_mask