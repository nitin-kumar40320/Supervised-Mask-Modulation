import torch
import torch.nn as nn
import os
import cv2
from PIL import Image
import numpy as np
import json

from data_loader import Data_Loader, Data_Loader_val
from GMM_Trainer import Trainer
from combined_loss import combined_loss_function
import sys
sys.path.append('/home/ayush/segmentation')
from unet_model import UNet
from enet_model import ENet


total_epochs = 1000
pretrainingEpochs = 200
batch_size_is = 4
learning_rate=0.1
number_of_classes = 2 # Including background
classes_to_focus_on = [1]
folds = [0,1,2,3,4]
dest_dir = f"/home/ayush/segmentation/Results_5cv_unet/GMM_{total_epochs}/Cracks"

log_dirs = [os.path.join(dest_dir, f'fold_{fold}') for fold in folds]

def main(train_images_list, val_images_list, gt_mask_list, path_to_log_dir):

    path_to_processed_masks_dir = os.path.join(path_to_log_dir, "transformed_masks")
    path_to_pred_mask_dir = os.path.join(path_to_log_dir, "predicted_masks")
    os.makedirs(path_to_log_dir, exist_ok=True)
    os.makedirs(path_to_pred_mask_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epoch_visual_img = os.path.basename(train_images_list[0][1]).split('.')[0]

    train_dataset = Data_Loader(image_lst=train_images_list)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size_is, shuffle=True, num_workers=8, pin_memory=True, persistent_workers=True)
    val_loader = torch.utils.data.DataLoader(Data_Loader_val(image_lst=val_images_list), batch_size=batch_size_is, shuffle=False, num_workers=8, pin_memory=True, persistent_workers=True)

    model = UNet(in_channels=3, out_channels=number_of_classes, init_features=32)
    model.to(device)

    with open(os.path.join(path_to_log_dir, 'hyperparams.txt'), 'w') as f:
        f.write(f'Total Epochs : {total_epochs}\nBatch Size : {batch_size_is}\nLearning Rate : {learning_rate}\n{"-"*30}\nData Processing:\n{train_dataset.general_transform}')
        
    criterion = combined_loss_function(number_of_classes)

    print("initializing trainer...")

    trainer = Trainer(model, train_dataloader=train_loader, val_dataloader=val_loader, epochs=total_epochs, pretrain_epochs=pretrainingEpochs, num_classes=number_of_classes, focus_classes=classes_to_focus_on, lr=learning_rate, device=device, loss=criterion, log_dir=path_to_log_dir, pred_mask_dir=path_to_pred_mask_dir, gt_maskpath_list=gt_mask_list, transformed_mask_dir=path_to_processed_masks_dir, epoch_vis_img=epoch_visual_img)

    print("Starting training...")

    trainer.train()


for log_dir in log_dirs:
    json_path = os.path.join(log_dir, 'train.json')
    with open(json_path, 'r') as f:
        data = json.load(f)

    train_image_mask_pairs = [(item["image"], item["mask"]) for item in data]
    train_image_mask_pairs=[]

    path_to_processed_masks_dir = os.path.join(log_dir, "transformed_masks")
    os.makedirs(path_to_processed_masks_dir, exist_ok=True)

    print(f"Processing masks for {log_dir}")

    for item in data:

        mask_path = item["mask"]
        mask = np.array(Image.open(mask_path))
        
        temp_onehot = np.eye(number_of_classes)[mask]
        dims = list(range(temp_onehot.ndim))
        dims = [temp_onehot.ndim - 1] + dims[:-1]
        mask = np.transpose(temp_onehot, axes=dims).astype(np.float32)
        del temp_onehot

        mask_name = os.path.basename(mask_path)
        processed_mask_path = os.path.join(path_to_processed_masks_dir, os.path.splitext(mask_name)[0] + '.npy')
        np.save(processed_mask_path, mask)

        train_image_mask_pairs.append((item["image"], item["mask"], processed_mask_path))

    print(f"Resersed mask space for {log_dir}")

    gt_maskpath_list = [item["mask"] for item in data]

    json_path = os.path.join(log_dir, 'val.json')
    with open(json_path, 'r') as f:
        data = json.load(f)
    val_image_mask_pairs = [(item["image"], item["mask"]) for item in data]

    main(train_images_list=train_image_mask_pairs, val_images_list=val_image_mask_pairs, gt_mask_list=gt_maskpath_list, path_to_log_dir=log_dir)