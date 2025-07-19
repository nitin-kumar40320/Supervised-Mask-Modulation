import os
from torch.utils.data import Dataset
import torch
import numpy as np

class Data_Loader(Dataset):
    def __init__(self, image_dir, mask_dir, processed_mask_dir):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.processed_mask_dir = processed_mask_dir

        self.image_names = sorted([f for f in os.listdir(mask_dir) if f.endswith(".pt")])

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_names[idx])
        msk_path = os.path.join(self.mask_dir, self.image_names[idx])
        processed_mask_path = os.path.join(self.processed_mask_dir, self.image_names[idx])

        image = torch.load(img_path)  # Tensor, shape: (1, H, W, D), dtype: float32
        mask = torch.load(msk_path)   # Tensor, shape: (1, H, W, D), dtype: long
        processed_mask = torch.load(processed_mask_path)

        return {
            "image": image,
            "mask": mask,
            "transformed_mask": processed_mask,
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
