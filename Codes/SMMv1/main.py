import torch
import torch.nn as nn
import os
import cv2
from PIL import Image
import numpy as np
import random

from data_loader import Data_Loader
from GMM_Trainer import Trainer
from GMM_Trainer_segformer import Segformer_Trainer
from combined_loss import combined_loss_function
import sys
sys.path.append('/Codes')
from unet_model import UNet
from segnet import SegNet
from transformers import SegformerForSemanticSegmentation

total_epochs = 1000
pretrainingEpochs = 200
batch_size_is = 2
learning_rate = 1e-5
number_of_classes = 8 # Including background
classes_to_focus_on = [0,1,2,3,4,5,6,7]
seeds = [1337, 1234]
path_to_images_dir = "/cityscapes/imagesTr"
path_to_masks_dir = "/cityscapes/labelsTr"
log_dir = f"/Results_segformer/GMM_{total_epochs}/Cityscapes"
model_path = '/Codes/segformer/num_classes_8'


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
            mask = np.array(Image.open(mask_path))
            
            temp_onehot = np.eye(number_of_classes)[mask]
            dims = list(range(temp_onehot.ndim))
            dims = [temp_onehot.ndim - 1] + dims[:-1]
            mask = np.transpose(temp_onehot, axes=dims).astype(np.float32)
            del temp_onehot

            processed_mask_path = os.path.join(path_to_processed_masks_dir, mask_name)

            np.save(processed_mask_path, cv2.resize(mask, (512, 512), interpolation=cv2.INTER_NEAREST))

        print("Masks processed.")

        print("Loading dataset...")

        g = torch.Generator()
        g.manual_seed(42)

        dataset = Data_Loader(image_dir=path_to_images_dir, mask_dir=path_to_masks_dir, processed_mask_dir=path_to_processed_masks_dir)
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size_is, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True, worker_init_fn=seed_worker, generator=g)

        # model = UNet(in_channels=3, out_channels=number_of_classes, init_features=32)
        # model = SegNet(in_chn=3, out_chn=number_of_classes)
        model = SegformerForSemanticSegmentation.from_pretrained(model_path)
        model.to(device)

        with open(os.path.join(path_to_log_dir, 'hyperparams.txt'), 'w') as f:
            f.write(f'Total Epochs : {total_epochs}\nBatch Size : {batch_size_is}\nLearning Rate : {learning_rate}\n{"-"*30}\nData Processing:\n{dataset.general_transform}')
            
        criterion = combined_loss_function(number_of_classes)

        print("initializing trainer...")

        trainer = Segformer_Trainer(model, train_loader, epochs=total_epochs, pretrain_epochs=pretrainingEpochs, num_classes=number_of_classes, focus_classes=classes_to_focus_on, lr=learning_rate, device=device, loss=criterion, log_dir=path_to_log_dir, pred_mask_dir=path_to_pred_mask_dir, gt_mask_dir=path_to_masks_dir, transformed_mask_dir=path_to_processed_masks_dir)

        print("Starting training...")

        trainer.train()
