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
    def __init__(self, image_dir, mask_dir, alter_mask_dir, num_classes):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.alter_mask_dir = alter_mask_dir
        self.general_transform = A.Compose([
            # --- Photometric Augmentations ---
            A.RandomBrightnessContrast(p=0.2, brightness_limit=0.2, contrast_limit=0.2, ensure_safe_range=True),
            # A.HueSaturationValue(p=0.2),  # color distortions (safe for RGB natural images)
            # A.CLAHE(p=0.1),  # improves contrast locally

            # # --- Noise & Blur ---
            # A.GaussNoise(mean=0.0, std=(0.005, 0.01), per_channel=False, scale=1.0, p=0.3),
            # A.GaussianBlur(blur_limit=(3, 5), p=0.3),
            # A.MotionBlur(blur_limit=(3, 5), p=0.2),  # simulates motion artifacts
            
            # --- Normalize and ToTensor ---
            A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ToTensorV2()
            ], additional_targets={'alt_mask': 'mask'})
        self.image_names = sorted([img for img in os.listdir(mask_dir) if img.endswith(('png', 'jpg', 'jpeg', 'tif'))])
        self.num_classes = num_classes
    
    def __len__(self):
        return len(self.image_names)
    
    def __getitem__(self, idx):
        mask_name = self.image_names[idx]
        img_name = mask_name.split('.')[0] + '_0000.' + mask_name.split('.')[1]
        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, mask_name)
        alter_mask_path = os.path.join(self.alter_mask_dir, f'{os.path.splitext(mask_name)[0]}.npy')
        
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)
        
        image_np = np.array(image)               # shape: (H, W, 3)
        mask_np = np.array(mask)  

        # mask_np = np.array(mask)
        # unique_elements = np.unique(mask_np)
        # print(f"Unique elements in mask: {unique_elements}")
        # print(f"Number of unique elements: {len(unique_elements)}")
        alter_mask_np = np.load(alter_mask_path)
        # alt_mask_np = np.argmax(alter_mask_np, axis=0).astype(np.uint8)
        # alter_mask = Image.fromarray((alter_mask_np).astype(np.uint8))  # Convert to PIL Image
        # alter_mask=torch.from_numpy(alter_mask_np)
        # print(f'Image : {image_np.shape}, Mask : {mask_np.shape}, Alt Mask : {alter_mask_np.shape}')
        transformed = self.general_transform(image=image_np, mask=mask_np, alt_mask=alter_mask_np)

        image_tensor = transformed['image']
        mask_tensor = transformed['mask'].long()
        alt_mask_tensor = transformed['alt_mask']
        dims = list(range(alt_mask_tensor.ndim))
        alt_mask_tensor = alt_mask_tensor.permute(dims[-1], *dims[:-1]).long() #[C, H, W]
        # alt_mask_one_hot = F.one_hot(alt_mask_tensor, num_classes=self.num_classes).permute(2, 0, 1).float()
        # num_classes = torch.unique(mask).numel()
        # alter_mask = F.one_hot(alter_mask, num_classes=num_classes).permute(0, 3, 1, 2).float()
        # print(f'Image Shape : {image_tensor.shape}\nMask Shape : {mask_tensor.shape}\nAlt Shape : {alt_mask_tensor.shape}')
        
        return image_tensor, mask_tensor, alt_mask_tensor, alter_mask_path

class Data_Loader_Test(Dataset):
    def __init__(self, image_dir):
        self.image_dir = image_dir
        
        self.general_transform = A.Compose([
            # A.RandomBrightnessContrast(p=0.2, ensure_safe_range=True),
            # A.GaussNoise(var_limit=(10.0, 50.0), mean=0, p=0.3),
            # A.RandomBrightnessContrast(p=0.2, ensure_safe_range=True),
            # A.GaussianBlur(blur_limit=(3,5), p=0.3),
            A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            # A.Resize(512, 512),
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