import os
from torch.utils.data import Dataset
import torch

class Data_Loader(Dataset):
    def __init__(self, image_dir, mask_dir):
        self.image_dir = image_dir
        self.mask_dir = mask_dir

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

        image = torch.load(img_path)  # Tensor, shape: (1, H, W, D), dtype: float32
        mask = torch.load(msk_path)   # Tensor, shape: (1, H, W, D), dtype: long

        return {
            "image": image,
            "mask": mask,
            "name": self.image_names[idx]
        }



class Data_Loader_Test(Dataset):
    def __init__(self, image_dir):
        self.image_dir = image_dir
        self.image_names = sorted([
            f for f in os.listdir(image_dir) if f.lower().endswith(('png', 'jpg', 'jpeg', 'tif'))
        ])

        # self.transform = transforms.Compose([
        #     transforms.Resize((512, 512)),
        #     transforms.ToTensor(),
        #     transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        # ])

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        img_path = os.path.join(self.image_dir, img_name)

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            raise FileNotFoundError(f"Error loading image: {e}")

        image = self.transform(image)
        return image, img_name
