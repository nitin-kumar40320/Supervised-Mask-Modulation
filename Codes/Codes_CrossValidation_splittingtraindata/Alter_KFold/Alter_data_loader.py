from torch.utils.data import Dataset
from torchvision import transforms
import torch
from PIL import Image
import os
import numpy as np
import torch.nn.functional as F
import albumentations as A
from albumentations.pytorch import ToTensorV2

from torch.utils.data import Dataset
from torchvision import transforms
import torch
from PIL import Image
import os
import numpy as np
import torch.nn.functional as F
import albumentations as A
from albumentations.pytorch import ToTensorV2

class Data_Loader(Dataset):
    def __init__(self, image_paths, mask_paths, alter_mask_dir=None, num_classes=2, use_altered=True):
        """
        image_paths: list of image file paths
        mask_paths: list of mask file paths
        alter_mask_dir: directory of .npy altered masks (can be None)
        use_altered: if False, will not attempt to load altered masks
        """
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.alter_mask_dir = alter_mask_dir
        self.num_classes = num_classes
        self.use_altered = use_altered

        self.general_transform = A.Compose([
            A.RandomBrightnessContrast(p=0.2, brightness_limit=0.2, contrast_limit=0.2, ensure_safe_range=True),
            A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ToTensorV2()
        ], additional_targets={'alt_mask': 'mask'} if use_altered else None)

    def __len__(self):
        return len(self.mask_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]
        mask_filename = os.path.basename(mask_path)

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)

        image_np = np.array(image)
        mask_np = np.array(mask)

        if self.use_altered:
            assert self.alter_mask_dir is not None, "alter_mask_dir cannot be None when use_altered=True"
            alter_mask_path = os.path.join(self.alter_mask_dir, f"{os.path.splitext(mask_filename)[0]}.npy")
            alter_mask_np = np.load(alter_mask_path)

            transformed = self.general_transform(image=image_np, mask=mask_np, alt_mask=alter_mask_np)
            image_tensor = transformed['image']
            mask_tensor = transformed['mask'].long()
            alt_mask_tensor = transformed['alt_mask'].permute(2, 0, 1).long()

            return image_tensor, mask_tensor, alt_mask_tensor, alter_mask_path
        else:
            transformed = self.general_transform(image=image_np, mask=mask_np)
            image_tensor = transformed['image']
            mask_tensor = transformed['mask'].long()

            return image_tensor, mask_tensor # return dummy tensor and blank path

class Data_Loader_Test(Dataset):
    def __init__(self, image_dir):
        self.image_dir = image_dir
        self.image_names = sorted([
            img for img in os.listdir(image_dir) if img.endswith(('png', 'jpg', 'jpeg', 'tif'))
        ])

        self.general_transform = A.Compose([
            A.RandomBrightnessContrast(p=0.2, ensure_safe_range=True),
            A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ToTensorV2()
        ])

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        img_path = os.path.join(self.image_dir, img_name)

        image = np.array(Image.open(img_path).convert("RGB"))
        transformed = self.general_transform(image=image)
        return transformed['image'], img_name
