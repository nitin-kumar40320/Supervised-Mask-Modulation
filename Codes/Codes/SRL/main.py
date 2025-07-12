import torch
import torch.nn as nn
import os
import cv2
import sys
from PIL import Image
import numpy as np
from data_loader import Data_Loader
from SRL_Trainer import Trainer
from combined_loss import combined_loss_function
from tubed_skel_transform import tubed_skeleton_transform

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from unet_model import UNet

total_epochs = 250
batch_size_is = 4
learning_rate= 0.1
num_classes = 2    ## including background
path_to_images_dir = "/home/nitin1/segmentation/Dataset015_DRIVE/imagesTr"
path_to_masks_dir = "/home/nitin1/segmentation/Dataset015_DRIVE/labelsTr"
path_to_log_dir = f"/home/nitin1/segmentation/Results_unet/SRL_{total_epochs}/Drive"


path_to_processed_masks_dir = os.path.join(path_to_log_dir,"processed_masks")
os.makedirs(path_to_processed_masks_dir, exist_ok=True)
os.makedirs(path_to_log_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Processing masks to get tubed skeleton...")

for mask_name in sorted(os.listdir(path_to_masks_dir)):
    mask_path = os.path.join(path_to_masks_dir, mask_name)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    processed_mask = tubed_skeleton_transform(mask)

    cv2.imwrite(os.path.join(path_to_processed_masks_dir, mask_name), processed_mask)

    # processed_mask = processed_mask.astype(np.float32)
    # processed_mask*=255
    # processed_mask/= (num_classes-1)
    # cv2.imwrite(os.path.join(path_to_processed_masks_dir, mask_name[:-4]+"_visible"+mask_name[-4:]), processed_mask)

print("Masks processed.")

print("Loading dataset...")

dataset = Data_Loader(image_dir=path_to_images_dir, mask_dir=path_to_masks_dir, processed_mask_dir=path_to_processed_masks_dir)
train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size_is, shuffle=True)

model = UNet(in_channels=3, out_channels=num_classes, init_features=32)
model.to(device)

criterion = combined_loss_function(num_of_class=num_classes)
with open(os.path.join(path_to_log_dir, 'hyperparams.txt'), 'w') as f:
    f.write(f'Total Epochs : {total_epochs}\nBatch Size : {batch_size_is}\nLearning Rate : {learning_rate}\n{"-"*30}\nData Processing:\n{dataset.general_transform}')

trainer = Trainer(model, train_loader, epochs=total_epochs, lr=learning_rate, device=device, loss=criterion, log_dir=path_to_log_dir)

print("Starting training...")

trainer.train()
