from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

class Data_Loader(Dataset):
    def __init__(self, image_dir, mask_dir, processed_mask_dir):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.processed_mask_dir = processed_mask_dir

        self.general_transform = A.Compose([
            A.Resize(512, 512),
            A.RandomBrightnessContrast(p=0.2, ensure_safe_range=True),
            A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ToTensorV2()
        ], additional_targets={'transformed_mask': 'mask'})

        self.image_names = sorted([img for img in os.listdir(mask_dir) if img.endswith(('png', 'jpg', 'jpeg'))])
    
    def __len__(self):
        return len(self.image_names)
    
    def __getitem__(self, idx):
        mask_name = self.image_names[idx]
        img_name = mask_name.split('.')[0] + '_0000.' + mask_name.split('.')[1]
        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, mask_name)
        processed_mask_path = os.path.join(self.processed_mask_dir, mask_name)
        
        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"))
        processed_mask = np.load(processed_mask_path).astype(np.int64)

        augmented = self.general_transform(image=image, mask=mask, transformed_mask=processed_mask)
        image = augmented['image']           # [3, 512, 512]
        mask = augmented['mask'].long()      # [512, 512], torch.int64
        processed_mask = augmented['transformed_mask'].long()      # [512, 512], torch.int64

        return image, mask, processed_mask, mask_name


class Data_Loader_Test(Dataset):
    def __init__(self, image_dir):
        self.image_dir = image_dir
        
        self.general_transform = A.Compose([
            A.Resize(512, 512),
            A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ToTensorV2()
        ])

        self.image_names = sorted([img for img in os.listdir(image_dir) if img.endswith(('png', 'jpg', 'jpeg'))])
    
    def __len__(self):
        return len(self.image_names)
    
    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        img_path = os.path.join(self.image_dir, img_name)
        
        image = np.array(Image.open(img_path).convert("RGB"))
        
        augmented = self.general_transform(image=image)
        
        image = augmented['image']           # [3, 512, 512] torch.int64
        
        return image, img_name