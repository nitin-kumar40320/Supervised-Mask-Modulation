from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
from scipy.ndimage import distance_transform_edt as eucl_distance

def class2one_hot_3d(seg: torch.Tensor, K: int) -> torch.Tensor:
    # Input shape: [D, H, W] → Output shape: [K, D, H, W]
    one_hot = torch.nn.functional.one_hot(seg.long(), num_classes=K)  # [D, H, W, K]
    return one_hot.permute(3, 0, 1, 2)  # [K, D, H, W]

def one_hot2dist_3d(seg, resolution=[1, 1, 1], dtype=np.float32):
    K = len(seg)
    res = np.zeros_like(seg, dtype=dtype)
    for k in range(K):
        posmask = seg[k].astype(bool)
        if posmask.any():
            negmask = ~posmask
            res[k] = distance_transform_edt(negmask, sampling=resolution) * negmask \
                   - (distance_transform_edt(posmask, sampling=resolution) - 1) * posmask
    return res

class Data_Loader(Dataset):
    def __init__(self, image_dir, mask_dir, dist_map_dir):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.dist_map_dir = dist_map_dir

        self.image_names = sorted([
            f for f in os.listdir(image_dir) if f.endswith(".pt")
        ])
        self.mask_names = sorted([
            f for f in os.listdir(mask_dir) if f.endswith(".pt")
        ])
        assert len(self.image_names) == len(self.mask_names), "Mismatch between image and mask .pt files"

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_names[idx])
        msk_path = os.path.join(self.mask_dir, self.mask_names[idx])
        dist_map_path = os.path.join(self.dist_map_dir, self.mask_names[idx])

        image = torch.load(img_path)  # Tensor, shape: (1, H, W, D), dtype: float32
        mask = torch.load(msk_path)   # Tensor, shape: (1, H, W, D), dtype: long
        dist_map = torch.load(dist_map_path)  # Tensor, shape: (1, H, W, D), dtype: float32

        return {
            "image": image,
            "mask": mask,
            "dist_map": dist_map,
            "name": self.image_names[idx]
        }


class Data_Loader_Test(Dataset):
    def __init__(self, image_dir):
        self.image_dir = image_dir
        self.image_names = sorted([
            f for f in os.listdir(image_dir) if f.endswith(".pt")
        ])

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        img_path = os.path.join(self.image_dir, img_name)

        image = torch.load(img_path)  # Tensor, shape: (1, H, W, D), dtype: float32

        return image, img_name