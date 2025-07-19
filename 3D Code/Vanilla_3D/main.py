import os
import sys
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import numpy as np
import random

from torch.utils.data import DataLoader
from data_loader import Data_Loader 
from Vanilla_Trainer import Trainer
from vanilla_loss import VanillaLoss
sys.path.append('/home/ayush/segmentation')
from unet_model import UNet3D

# Hyperparameters
total_epochs = 500
batch_size_is = 4
learning_rate = 1e-4
num_classes = 4  # Including background
seeds = [1337, 1234, 999, 2024, 2025]
# seeds = [2025]

# Paths
base_image_dir = "/home/ayush/segmentation/Dataset007_Amos/imagesSubsetPt"
base_mask_dir = "/home/ayush/segmentation/Dataset007_Amos/labelsSubsetPt"
base_log_dir = f"/home/ayush/segmentation/Results/Vanilla_{total_epochs}/Amos"

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import nibabel as nib

def convert_nii_to_pt(image_dir, mask_dir):
    image_names = sorted([f for f in os.listdir(image_dir) if f.endswith('.nii.gz')])
    mask_names = sorted([f for f in os.listdir(mask_dir) if f.endswith('.nii.gz')])
    
    assert len(image_names) == len(mask_names), "Mismatch in images and masks."

    for img_name, msk_name in zip(image_names, mask_names):
        pt_img_path = os.path.join(image_dir, img_name.replace(".nii.gz", ".pt"))
        pt_msk_path = os.path.join(mask_dir, msk_name.replace(".nii.gz", ".pt"))

        # Skip if already converted
        if os.path.exists(pt_img_path) and os.path.exists(pt_msk_path):
            continue

        print(f"Converting: {img_name} and {msk_name} to .pt format")

        # Load NIfTI
        img = nib.load(os.path.join(image_dir, img_name)).get_fdata()
        msk = nib.load(os.path.join(mask_dir, msk_name)).get_fdata()

        # Add channel dim
        img = torch.tensor(np.expand_dims(img, axis=0)).float()  # (1, H, W, D)
        msk = torch.tensor(np.expand_dims(msk, axis=0)).long()   # (1, H, W, D)

        # Save
        torch.save(img, pt_img_path)
        torch.save(msk, pt_msk_path)

# Run once before training loop
convert_nii_to_pt(base_image_dir, base_mask_dir)

def set_seed(seed: int):
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

# Loop over all seeds
for seed in seeds:
    print(f"\n=== Running training with seed {seed} ===\n")
    set_seed(seed)

    # Dataset and DataLoader
    dataset = Data_Loader(image_dir=base_image_dir, mask_dir=base_mask_dir)
    g = torch.Generator()
    g.manual_seed(seed)

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size_is,
        shuffle=True,
        prefetch_factor=4,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        worker_init_fn=seed_worker,
        generator=g
    )

    # Model
    model = UNet3D(in_channels=1, out_channels=num_classes, init_features=16)
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)

    torch.cuda.empty_cache()
    model.to(device)

    # Log directory
    log_dir = os.path.join(base_log_dir, f"seed_{seed}")
    os.makedirs(log_dir, exist_ok=True)

    # Save hyperparameters
    with open(os.path.join(log_dir, 'hyperparams.txt'), 'w') as f:
        f.write(f'Seed : {seed}\n')
        f.write(f'Total Epochs : {total_epochs}\n')
        f.write(f'Batch Size : {batch_size_is}\n')
        f.write(f'Learning Rate : {learning_rate}\n')
        f.write(f'Device : {device}\n')
        f.write('-' * 30 + '\n')

    # Loss and Trainer
    criterion = VanillaLoss(num_class=num_classes)
    trainer = Trainer(
        model, data_loader, epochs=total_epochs, num_classes=num_classes,
        lr=learning_rate, device=device, loss=criterion, log_dir=log_dir
    )

    print(f"Starting training for seed {seed}...")
    trainer.train()
