import os
from torch.utils.data import Dataset
import torch
import numpy as np

class Data_Loader(Dataset):
    def __init__(self, image_dir, mask_dir, alter_mask_dir):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.alter_mask_dir = alter_mask_dir
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
        alter_mask_path = os.path.join(self.alter_mask_dir, f'{os.path.splitext(self.mask_names[idx])[0]}.npy')
        image = torch.load(img_path)  # Tensor, shape: (1, H, W, D), dtype: float32
        mask = torch.load(msk_path).long()   # Tensor, shape: (1, H, W, D), dtype: long
        alter_mask = torch.tensor(np.load(alter_mask_path))

        return {
            "image": image,
            "mask": mask,
            "alter_mask": alter_mask,
            "alter_mask_path": alter_mask_path
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