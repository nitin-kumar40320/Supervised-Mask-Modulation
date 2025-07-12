from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

class Data_Loader(Dataset):
    def __init__(self, image_lst):

        self.image_lst = image_lst

        self.general_transform = A.Compose([
            A.RandomBrightnessContrast(p=0.2, ensure_safe_range=True),
            A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ToTensorV2()
        ])

    def __len__(self):
        return len(self.image_lst)

    def __getitem__(self, idx):
        img_path = self.image_lst[idx][0]
        mask_path = self.image_lst[idx][1]
        mask_name = os.path.basename(mask_path)
        
        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"))  # assuming mask is grayscale

        augmented = self.general_transform(image=image, mask=mask)
        image = augmented['image']           # [3, 512, 512]
        mask = augmented['mask'].long()      # [512, 512], torch.int64

        # print(image.shape, mask.shape, mask_name)

        return image, mask, mask_name


class Data_Loader_Test(Dataset):
    def __init__(self, image_lst):
        self.image_lst = image_lst
        
        self.general_transform = A.Compose([
            A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ToTensorV2()
        ])

    
    def __len__(self):
        return len(self.image_lst)
    
    def __getitem__(self, idx):
        img_path = self.image_lst[idx][0]
        img_name = os.path.basename(img_path)
        
        image = np.array(Image.open(img_path).convert("RGB"))
        
        augmented = self.general_transform(image=image)
        image = augmented['image']           # [3, 512, 512]
        
        return image, img_name