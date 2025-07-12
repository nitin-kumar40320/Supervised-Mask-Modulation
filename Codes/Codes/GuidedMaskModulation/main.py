import torch
import torch.nn as nn
import os
import cv2
from PIL import Image
import numpy as np

from data_loader import Data_Loader
from GMM_Trainer import Trainer
from combined_loss import combined_loss_function
import sys
sys.path.append('/home/nitin1/segmentation')
from unet_model import UNet
from enet_model import ENet

# total_epochs = 500
# pretrainingEpochs = 100
# batch_size_is = 4
# learning_rate=0.1
# number_of_classes = 4 # Including background
# classes_to_focus_on = [0,1,2,3]
# path_to_images_dir = "/home/nitin1/segmentation/Dataset005_Bombr/imagesTr"
# path_to_masks_dir = "/home/nitin1/segmentation/Dataset005_Bombr/labelsTr"
# path_to_log_dir = f"/home/nitin1/segmentation/Results_enet/GMM_{total_epochs}/Bombr"

total_epochs = 250
pretrainingEpochs = 50
batch_size_is = 4
learning_rate=0.1
number_of_classes = 2 # Including background
classes_to_focus_on = [1]
path_to_images_dir = "/home/nitin1/segmentation/Dataset015_DRIVE/imagesTr"
path_to_masks_dir = "/home/nitin1/segmentation/Dataset015_DRIVE/labelsTr"
path_to_log_dir = f"/home/nitin1/segmentation/Results_unet/GMM_{total_epochs}/Bombr"

path_to_processed_masks_dir = os.path.join(path_to_log_dir, "transformed_masks")
path_to_pred_mask_dir = os.path.join(path_to_log_dir, "predicted_masks")
os.makedirs(path_to_processed_masks_dir, exist_ok=True)
os.makedirs(path_to_log_dir, exist_ok=True)
os.makedirs(path_to_pred_mask_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Processing masks to get just create a copy for pretraining...")
for mask_name in sorted(os.listdir(path_to_masks_dir)):
    mask_path = os.path.join(path_to_masks_dir, mask_name)
    mask = np.array(Image.open(mask_path))
    
    temp_onehot = np.eye(number_of_classes)[mask]
    dims = list(range(temp_onehot.ndim))
    dims = [temp_onehot.ndim - 1] + dims[:-1]
    mask = np.transpose(temp_onehot, axes=dims).astype(np.float32)
    del temp_onehot

    np.save(os.path.join(path_to_processed_masks_dir, mask_name[:-4]), mask)

print("Masks processed.")

print("Loading dataset...")

dataset = Data_Loader(image_dir=path_to_images_dir, mask_dir=path_to_masks_dir, processed_mask_dir=path_to_processed_masks_dir)
train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size_is, shuffle=True)

# model = UNet(in_channels=3, out_channels=number_of_classes, init_features=32)
model = ENet(num_classes=number_of_classes)
model.to(device)

with open(os.path.join(path_to_log_dir, 'hyperparams.txt'), 'w') as f:
    f.write(f'Total Epochs : {total_epochs}\nBatch Size : {batch_size_is}\nLearning Rate : {learning_rate}\n{"-"*30}\nData Processing:\n{dataset.general_transform}')
    
criterion = combined_loss_function(number_of_classes)

print("initializing trainer...")

trainer = Trainer(model, train_loader, epochs=total_epochs, pretrain_epochs=pretrainingEpochs, num_classes=number_of_classes, focus_classes=classes_to_focus_on, lr=learning_rate, device=device, loss=criterion, log_dir=path_to_log_dir, pred_mask_dir=path_to_pred_mask_dir, gt_mask_dir=path_to_masks_dir, transformed_mask_dir=path_to_processed_masks_dir)

print("Starting training...")

trainer.train()
