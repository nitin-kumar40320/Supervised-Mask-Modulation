from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
from scipy.ndimage import distance_transform_edt as eucl_distance

def class2one_hot(seg: torch.Tensor, K: int) -> torch.Tensor:
    return torch.nn.functional.one_hot(seg.long(), num_classes=K).permute(2, 0, 1)

def one_hot2dist(seg, resolution=[1, 1], dtype=None):
    K = len(seg)
    res = np.zeros_like(seg, dtype=dtype)
    for k in range(K):
        posmask = seg[k].astype(bool)
        if posmask.any():
            negmask = ~posmask
            res[k] = eucl_distance(negmask, sampling=resolution) * negmask \
                   - (eucl_distance(posmask, sampling=resolution) - 1) * posmask
    return res

class Data_Loader(Dataset):
    def __init__(self, image_dir, mask_dir, K=2):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.K = K

        self.general_transform = A.Compose([
            A.Resize(512, 512),
            A.RandomBrightnessContrast(p=0.2, ensure_safe_range=True),
            A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ToTensorV2()
        ])

        self.image_names = sorted([
            img for img in os.listdir(mask_dir) if img.endswith(('png', 'jpg', 'jpeg', 'tif'))
        ])

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        mask_name = self.image_names[idx]
        img_name = mask_name.split('.')[0] + '_0000.' + mask_name.split('.')[1]
        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, mask_name)

        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"))  # class index mask

        augmented = self.general_transform(image=image, mask=mask)
        image = augmented['image']                    # Tensor [3, H, W]
        mask = augmented['mask'].long()               # Tensor [H, W]

        one_hot_mask = class2one_hot(mask, self.K)    # Tensor [K, H, W]

        dist_map = one_hot2dist(one_hot_mask.numpy(), resolution=[1, 1], dtype=np.float32)
        dist_map = torch.from_numpy(dist_map)     # Tensor [K, H, W]
        return image, mask, dist_map, mask_name


class Data_Loader_Test(Dataset):
    def __init__(self, image_dir):
        self.image_dir = image_dir
        
        self.general_transform = A.Compose([
            A.Resize(512, 512),
            A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ToTensorV2()
        ])

        self.image_names = sorted([img for img in os.listdir(image_dir) if img.endswith(('png', 'jpg', 'jpeg', 'tif'))])
    
    def __len__(self):
        return len(self.image_names)
    
    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        img_path = os.path.join(self.image_dir, img_name)
        
        image = np.array(Image.open(img_path).convert("RGB"))
        
        augmented = self.general_transform(image=image)
        image = augmented['image']           # [3, 512, 512]
        
        return image, img_name