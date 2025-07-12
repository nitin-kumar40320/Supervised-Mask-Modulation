from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np

class Data_Loader(Dataset):
    def __init__(self, image_lst):
        self.image_lst = image_lst

        self.general_transform = A.Compose([
            A.RandomBrightnessContrast(p=0.2, ensure_safe_range=True),
            A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ToTensorV2()
            ], additional_targets={'alt_mask': 'mask'})
        
    def __len__(self):
        return len(self.image_lst)
    
    def __getitem__(self, idx):
        img_path = self.image_lst[idx][0]
        mask_path = self.image_lst[idx][1]
        processed_mask_path = self.image_lst[idx][2]
        mask_name = os.path.basename(mask_path)
        
        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"))
        processed_mask = np.array(Image.open(processed_mask_path).convert("L"))
        
        transformed = self.general_transform(image=image, mask=mask, alt_mask=processed_mask)

        image_tensor = transformed['image']
        mask_tensor = transformed['mask'].long()
        processed_mask_tensor = transformed['alt_mask']
        
        return image_tensor, mask_tensor, processed_mask_tensor, mask_name


class Data_Loader_Test(Dataset):
    def __init__(self, image_dir):
        self.image_dir = image_dir
        self.general_transform = A.Compose([
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