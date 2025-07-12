import torch
import shutil
import os
import cv2
from Alter_data_loader import Data_Loader, Data_Loader_Test
from Alter_Trainer import Trainer
from AlterLoss import NDLoss
from Alter_predict import predict
import numpy as np
from PIL import Image
import torch.nn.functional as F
import sys
sys.path.append('/home/ayush/segmentation')
from enet_model import ENet
# from monai.networks.nets import UNet
import albumentations as A

input_dir = '/home/ayush/segmentation/Dataset065_Cracks'
path_to_images_dir = os.path.join(input_dir, "imagesTr")
path_to_masks_dir = os.path.join(input_dir, "labelsTr")
test_images_dir = os.path.join(input_dir, "imagesTs")
total_epochs = 500
batch_size_is = 4
learning_rate = 0.1
num_classes = 2
print(f'Input Directory: {input_dir}')
print(f'Input Classes : {num_classes}')

path_to_alter_masks = path_to_masks_dir + "_altered"
if os.path.exists(path_to_alter_masks):
    shutil.rmtree(path_to_alter_masks)
    print("Existing altered masks directory removed.")

os.makedirs(path_to_alter_masks, exist_ok=True)

num_samples = 0
# Load original_mask as altered_mask
for fname in os.listdir(path_to_masks_dir):
    if fname.endswith(('.png', '.jpg', '.jpeg', '.tif')):
        mask_path = os.path.join(path_to_masks_dir, fname)
        alter_mask_path = os.path.join(path_to_alter_masks, os.path.splitext(fname)[0] + ".npy")

        # Load the mask as grayscale so each pixel is a class index
        mask_img = Image.open(mask_path)
        mask_np = np.array(mask_img)
        # print(f'Step 1: {mask_np.shape}')
        mask_np = np.eye(num_classes)[mask_np] # [H, W, C]
        # print(f'Step 2: {mask_np.shape}')
        np.save(alter_mask_path, mask_np.astype(np.float32))
        num_samples += 1
# del mask_img, mask_np

print("Altered masks directory created at:", path_to_alter_masks)
print(f'Training Samples : {num_samples}')
path_to_log_dir = f"/home/ayush/segmentation/Results_enet/Alter_{total_epochs}/Cracks"
os.makedirs(path_to_log_dir, exist_ok=True)

# Visulaize altered mask for one sample
path_to_visualizations = os.path.join(path_to_log_dir,'visulaizations')
os.makedirs(path_to_visualizations, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data Loader
dataset = Data_Loader(image_dir=path_to_images_dir, mask_dir=path_to_masks_dir, alter_mask_dir = path_to_alter_masks, num_classes=num_classes)
train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size_is, shuffle=True)


# model = UNet(
#     spatial_dims=2,
#     in_channels=3,
#     out_channels=num_classes,  # reconstruction
#     channels=(16, 32, 64, 128),
#     strides=(2, 2, 2),
#     num_res_units=2,
#     norm='batch'
# ).to(device)

# Load Model
model = ENet(num_classes=num_classes)
# model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet', in_channels=3, out_channels=num_classes, init_features=32, pretrained=False)
model.to(device)

# Defining Loss
criterion = NDLoss(num_samples = num_samples, num_classes=num_classes, batch_size = batch_size_is, device = device, viz_path = path_to_visualizations)
with open(os.path.join(path_to_log_dir, 'hyperparams.txt'), 'w') as f:
    f.write(f'Total Epochs : {total_epochs}\nBatch Size : {batch_size_is}\nLearning Rate : {learning_rate}\n{"-"*30}\nData Processing:\n{dataset.general_transform}')
# Create trainer and begin training
trainer = Trainer(model, train_loader, epochs=total_epochs, lr=learning_rate, device=device, loss=criterion, log_dir=path_to_log_dir)
print("Starting Training...")
trainer.train()

print('Ended Training....')

#------------------------------------------------------------------------------------------------#
output_mask_dir = os.path.join(path_to_log_dir, "test_output")
os.makedirs(output_mask_dir, exist_ok=True)
model_path = os.path.join(path_to_log_dir, "best_checkpoint_path.pth")
checkpoint = torch.load(model_path, map_location=device)
model.load_state_dict(checkpoint)
dataset = Data_Loader_Test(test_images_dir)

print('Starting Prediction on Test Set')
predict(dataset, output_mask_dir, model)
