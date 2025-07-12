import torch
import os
import torchvision
import cv2
import numpy as np
from torch.utils.data import DataLoader
from data_loader import Data_Loader_Test

model_path = "/home/devansh/Desktop/SkelRec/EWLenv/Test_UNet/Results/SRL_250/run1/best_checkpoint_path.pth"
test_images_dir = "/home/devansh/Desktop/SkelRec/EWLenv/Test_UNet/Dataset015_DRIVE/imagesTs"
output_mask_dir = "/home/devansh/Desktop/SkelRec/EWLenv/Test_UNet/Results/SRL_250/run1/test_output"
os.makedirs(output_mask_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet', in_channels=3, out_channels=1, init_features=32, pretrained=False)
checkpoint = torch.load(model_path, map_location=device)
model.load_state_dict(checkpoint)
model = model.to(device) 

dataset = Data_Loader_Test(test_images_dir)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

print("Starting prediction...")

for images, img_names in dataloader :
    images = images.to(device)
    with torch.no_grad():
        outputs = model(images)
        outputs = torch.sigmoid(outputs)
        predicted_masks = (outputs > 0.7).float()
    
    for i in range(predicted_masks.size(0)):
        mask_path = os.path.join(output_mask_dir, img_names[i])
        # print(predicted_masks[i].shape)
        torchvision.utils.save_image(predicted_masks[i], mask_path)
