import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import cv2
from PIL import Image
import numpy as np
import random

from data_loader import Data_Loader
from GMM_Trainer import Trainer
from combined_loss import combined_loss_function
import sys
sys.path.append('/home/nitin1/segmentation')
from unet_model import UNet3D


total_epochs = 400
pretrainingEpochs = 80
batch_size_is = 4
learning_rate = 1e-4
number_of_classes = 4 # Including background
classes_to_focus_on = [1,2,3]
seeds = [2025]
path_to_images_dir = "/home/nitin1/segmentation/Dataset007_Amos/imagesSubsetPt"
path_to_masks_dir = "/home/nitin1/segmentation/Dataset007_Amos/labelsSubsetPt"
log_dir = f"/home/nitin1/segmentation/Results_unet/GMM_{total_epochs}/Amos"


def set_seed(seed: int):
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


if __name__ == "__main__":

    for seed in seeds:

        set_seed(seed)

        path_to_log_dir = os.path.join(log_dir, f'seed_{seed}')
        os.makedirs(path_to_log_dir, exist_ok=True)

        path_to_processed_masks_dir = os.path.join(path_to_log_dir, "transformed_masks")
        path_to_pred_mask_dir = os.path.join(path_to_log_dir, "predicted_masks")
        os.makedirs(path_to_processed_masks_dir, exist_ok=True)
        os.makedirs(path_to_pred_mask_dir, exist_ok=True)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print("Processing masks to get just create a copy for pretraining...")
        for mask_name in sorted(os.listdir(path_to_masks_dir)):
            mask_path = os.path.join(path_to_masks_dir, mask_name)
            
            mask = torch.load(mask_path)  # shape: [H, W] or [D, H, W]
            
            # One-hot encode (assumes class indices)
            one_hot = F.one_hot(mask.long(), num_classes=number_of_classes)  # [..., C]
            
            # Move channel to front: [C, ...]
            one_hot = one_hot.permute(-1, *range(mask.ndim)).float()  # [C, H, W] or [C, D, H, W]

            processed_mask_path = os.path.join(path_to_processed_masks_dir, mask_name)
            torch.save(one_hot.to(torch.uint8), processed_mask_path)

        print("Masks processed.")

        print("Loading dataset...")

        g = torch.Generator()
        g.manual_seed(42)

        dataset = Data_Loader(image_dir=path_to_images_dir, mask_dir=path_to_masks_dir, processed_mask_dir=path_to_processed_masks_dir)
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size_is, shuffle=True, 
        prefetch_factor=4, num_workers=4, pin_memory=True, persistent_workers=True, worker_init_fn=seed_worker, generator=g)

        model = UNet3D(in_channels=1, out_channels=number_of_classes, init_features=16)
        torch.cuda.empty_cache()
        model.to(device)

        with open(os.path.join(path_to_log_dir, 'hyperparams.txt'), 'w') as f:
            f.write(f'Total Epochs : {total_epochs}\nBatch Size : {batch_size_is}\nLearning Rate : {learning_rate}\n{"-"*30}')
            
        criterion = combined_loss_function(number_of_classes)

        print("initializing trainer...")

        trainer = Trainer(model, train_loader, epochs=total_epochs, pretrain_epochs=pretrainingEpochs, num_classes=number_of_classes, focus_classes=classes_to_focus_on, lr=learning_rate, device=device, loss=criterion, log_dir=path_to_log_dir, pred_mask_dir=path_to_pred_mask_dir, gt_mask_dir=path_to_masks_dir, transformed_mask_dir=path_to_processed_masks_dir)

        print("Starting training...")

        trainer.train()
