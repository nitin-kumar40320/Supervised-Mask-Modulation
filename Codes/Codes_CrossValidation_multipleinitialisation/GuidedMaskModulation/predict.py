import torch
import os
import torchvision
import cv2
import numpy as np
from torch.utils.data import DataLoader
from data_loader import Data_Loader_Test
from color_scheme import ColorMask
import sys
sys.path.append('/home/nitin1/segmentation')
from unet_model import UNet

model_path = "/home/devansh/Desktop/SkelRec/EWLenv/Test_UNet/Results/GMM_1000/Bombr/run1/best_checkpoint_path.pth"
test_images_dir = "/home/devansh/Desktop/SkelRec/EWLenv/Test_UNet/Dataset005_Bombr/imagesTs"
output_mask_dir = "/home/devansh/Desktop/SkelRec/EWLenv/Test_UNet/Results/GMM_1000/Bombr/run1/test_output"
num_classes = 4

os.makedirs(output_mask_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = UNet(in_channels=3, out_channels=number_of_classes, init_features=32)
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

