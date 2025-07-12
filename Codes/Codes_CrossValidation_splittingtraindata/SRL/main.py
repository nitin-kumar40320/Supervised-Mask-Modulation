import torch
import torch.nn as nn
import os
import cv2
import sys
from PIL import Image
import json
import numpy as np
from data_loader import Data_Loader
from SRL_Trainer import Trainer
from combined_loss import combined_loss_function

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from unet_model import UNet
from enet_model import ENet

total_epochs = 1000
batch_size_is = 4
learning_rate= 0.1
num_classes = 2    ## including background
folds = [0,1,2,3,4]
dest_dir = "/home/ayush/segmentation/Results_5cv_unet/SRL_1000/Cracks"

log_dirs = [os.path.join(dest_dir, f'fold_{fold}') for fold in folds]

def main(train_images_list, val_images_list, path_to_log_dir):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Loading dataset...")
    epoch_visual_img = os.path.basename(train_images_list[0][1]).split('.')[0]

    # print(train_images_list[0])

    train_dataset = Data_Loader(image_lst=train_images_list)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size_is, shuffle=True, num_workers=8, pin_memory=True, persistent_workers=True)
    val_loader = torch.utils.data.DataLoader(Data_Loader(image_lst=val_images_list), batch_size=batch_size_is, shuffle=False, num_workers=8, pin_memory=True, persistent_workers=True)

    model = UNet(in_channels=3, out_channels=num_classes, init_features=32)
    # model = ENet(num_classes=num_classes)
    model.to(device)

    criterion = combined_loss_function(num_of_class=num_classes)
    with open(os.path.join(path_to_log_dir, 'hyperparams.txt'), 'w') as f:
        f.write(f'Total Epochs : {total_epochs}\nBatch Size : {batch_size_is}\nLearning Rate : {learning_rate}\n{"-"*30}\nData Processing:\n{train_dataset.general_transform}')

    trainer = Trainer(model, train_dataloader=train_loader, val_dataloader=val_loader, epochs=total_epochs, lr=learning_rate, device=device, loss=criterion, log_dir=path_to_log_dir, number_of_classes=num_classes, epoch_vis_img=epoch_visual_img)

    print("Starting training...")

    trainer.train()


for log_dir in log_dirs:
    json_path = os.path.join(log_dir, 'train.json')
    with open(json_path, 'r') as f:
        data = json.load(f)
    train_image_mask_pairs = [(item["image"], item["mask"], item['processed_mask']) for item in data]

    json_path = os.path.join(log_dir, 'val.json')
    with open(json_path, 'r') as f:
        data = json.load(f)
    val_image_mask_pairs = [(item["image"], item["mask"], item['processed_mask']) for item in data]
    main(train_images_list=train_image_mask_pairs, val_images_list=val_image_mask_pairs, path_to_log_dir=log_dir)