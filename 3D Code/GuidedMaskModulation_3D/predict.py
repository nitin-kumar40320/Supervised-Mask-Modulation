import torch
import os
from torch.utils.data import DataLoader
from data_loader import Data_Loader_Test
import sys
sys.path.append('/home/nitin1/segmentation')
from unet_model import UNet3D
# from enet_model import ENet

import numpy as np
# from color_scheme import ColorMask
from PIL import Image

# Settings
seeds = [1337, 999, 1234, 2024, 2025]
base_model_path = "/home/nitin1/segmentation/Results_unet/GMM_400/Amos"
test_images_dir = "/home/nitin1/segmentation/Dataset007_Amos/imagesTsPt"
base_output_mask_dir = "/home/nitin1/segmentation/Results_unet/GMM_400/Amos"
num_classes = 4  # Including background

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# color_mapper = ColorMask()

# Dataset and DataLoader are constant across seeds
dataset = Data_Loader_Test(test_images_dir)
dataloader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=8, prefetch_factor=4)

for seed in seeds:
    print(f"\nStarting prediction for seed {seed}...")

    # Load model checkpoint for the given seed
    model_path = os.path.join(base_model_path, f"seed_{seed}", "best_checkpoint_path.pth")
    if not os.path.isfile(model_path):
        print(f"Checkpoint not found for seed {seed} at {model_path}. Skipping...")
        continue

    model = UNet3D(in_channels=1, out_channels=num_classes, init_features=16)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint)
    model = model.to(device)
    model.eval()

    # Prepare output directory for the seed
    seed_output_dir = os.path.join(base_output_mask_dir, f"seed_{seed}", 'test_output')
    os.makedirs(seed_output_dir, exist_ok=True)

    for images, img_names in dataloader:
        images = images.to(device)
        with torch.no_grad():
            outputs = model(images)  # shape: [B, C, H, W, D]
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            predicted_masks = torch.argmax(probabilities, dim=1)  # shape: [B, H, W, D]

        for i in range(predicted_masks.size(0)):
            mask_3d = predicted_masks[i].cpu().numpy()  # shape: [H, W, D]
            base_name = os.path.splitext(img_names[i])[0]

            # Save full 3D mask
            mask_path = os.path.join(seed_output_dir, base_name + ".npy")
            np.save(mask_path, mask_3d)

            # # Save middle slice as RGB image
            # mid_slice = mask_3d[:, :, mask_3d.shape[2] // 2]  # shape: [H, W]
            # mid_slice_tensor = torch.from_numpy(mid_slice).long()
            # rgb_img = color_mapper(mid_slice_tensor)  # PIL Image

            # rgb_path = os.path.join(seed_output_dir, base_name + "_rgb.png")
            # rgb_img.save(rgb_path)
