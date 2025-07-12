import torch
import torch.nn as nn
from data_loader import Data_Loader
from Vanilla_Trainer import Trainer
from vanilla_loss import VanillaLoss
import sys
sys.path.append('/home/nitin1/segmentation')
from unet_model import UNet
from enet_model import ENet
from hrnet_model import HRNetSegmentation
import os


total_epochs = 500
batch_size_is = 4
learning_rate = 0.1
num_classes = 2 ## including background

path_to_images_dir = "/home/nitin1/segmentation/Dataset065_Cracks/imagesTr"
path_to_masks_dir = "/home/nitin1/segmentation/Dataset065_Cracks/labelsTr"
path_to_log_dir = f"/home/nitin1/segmentation/Results_hrnet/Vanilla_{total_epochs}/Cracks"
os.makedirs(path_to_log_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = Data_Loader(image_dir=path_to_images_dir, mask_dir=path_to_masks_dir)
train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size_is, shuffle=True)


# model = UNet(in_channels=3, out_channels=num_classes, init_features=32)
# model = VisionTransformer(in_dim=3, out_dim=num_classes)
model = HRNetSegmentation(num_classes = num_classes, in_chans = 3)
model.to(device)

with open(os.path.join(path_to_log_dir, 'hyperparams.txt'), 'w') as f:
    f.write(f'Total Epochs : {total_epochs}\nBatch Size : {batch_size_is}\nLearning Rate : {learning_rate}\nDevice : {device}\n{"-"*30}\nData Processing : \n{dataset.general_transform}')
criterion = VanillaLoss(num_class = num_classes)

trainer = Trainer(model, train_loader, epochs=total_epochs, num_classes=num_classes, lr=learning_rate, device=device, loss=criterion, log_dir=path_to_log_dir)

print("Starting training...")

trainer.train()
