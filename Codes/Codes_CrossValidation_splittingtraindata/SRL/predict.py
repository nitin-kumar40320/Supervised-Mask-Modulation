import torch
import os
import torchvision
import cv2
import numpy as np
from torch.utils.data import DataLoader
from color_scheme import ColorMask
from data_loader import Data_Loader_Test

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from unet_model import UNet
from enet_model import ENet


test_images_dir = "/home/ayush/segmentation/Dataset015_Drive/imagesTs"
results_dir = "/home/ayush/segmentation/Results_5cv_unet/SRL_250/Drive"
number_of_classes = 2 ## including background
folds = [0,1,2,3,4]

for fold in folds:
    output_dir = os.path.join(results_dir, f"fold_{fold}")

    model_path = os.path.join(output_dir, "best_checkpoint_path.pth")
    output_mask_dir = os.path.join(output_dir, "test_output")

    os.makedirs(output_mask_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = UNet(in_channels=3, out_channels=number_of_classes, init_features=32)
    # model = ENet(num_classes=number_of_classes)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint)
    model = model.to(device) 

    dataset = Data_Loader_Test(test_images_dir)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    color_mapper = ColorMask()

    print("Starting prediction...")

    for images, img_names in dataloader :
        images = images.to(device)
        with torch.no_grad():
            outputs = model(images)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            predicted_masks = torch.argmax(probabilities, dim=1)

            print(torch.unique(predicted_masks))
        
        for i in range(predicted_masks.size(0)):
            mask_path_rgb = os.path.join(output_mask_dir, img_names[i][:-4] + "_rgb" + img_names[i][-4:])
            mask_path = os.path.join(output_mask_dir, img_names[i])

            print(predicted_masks[i].shape)
            
            colored_image = color_mapper(predicted_masks[i])
            colored_image.save(mask_path_rgb)
            cv2.imwrite(mask_path, predicted_masks[i].cpu().numpy())