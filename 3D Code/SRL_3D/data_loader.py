from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import torch

class Data_Loader(Dataset):
    def __init__(self, image_dir, mask_dir,processed_mask_dir):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.processed_mask_dir = processed_mask_dir
        self.mask_names = sorted([
            f for f in os.listdir(mask_dir) if f.endswith(".pt")
        ])
        # assert len(self.image_names) == len(self.mask_names), "Mismatch between image and mask .pt files"

    def __len__(self):
        return len(self.mask_names)

    def __getitem__(self, idx):
        mask_name = self.mask_names[idx]
        img_name = mask_name
        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, mask_name)
        processed_mask_path = os.path.join(self.processed_mask_dir, mask_name)

        image = torch.load(img_path)  # Tensor, shape: (1, H, W, D), dtype: float32
        mask = torch.load(mask_path)   # Tensor, shape: (1, H, W, D), dtype: long
        processed_mask = torch.load(processed_mask_path)
        return {
            "image": image,
            "mask": mask,
            "processed_mask" : processed_mask
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
