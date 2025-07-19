import torch
import numpy as np
import random
import os
import cv2
import sys
from PIL import Image
import numpy as np
from data_loader import Data_Loader
from SRL_Trainer import Trainer
from combined_loss import combined_loss_function
from tubed_skel_transform import tubed_skeleton_transform

sys.path.append('/home/nitin1/segmentation')
from unet_model import UNet3D

total_epochs = 400
batch_size_is = 4
learning_rate= 1e-4
num_classes = 4    ## including background
# seeds = [1337, 1234, 999, 2024, 2025]
seeds = [2025]
path_to_images_dir = "/home/nitin1/segmentation/Dataset007_Amos/imagesSubsetPt"
path_to_masks_dir = "/home/nitin1/segmentation/Dataset007_Amos/labelsSubsetPt"
log_dir = f"/home/nitin1/segmentation/Results_unet/SRL_{total_epochs}/Amos"


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


        path_to_processed_masks_dir = os.path.join(path_to_log_dir,"processed_masks")
        os.makedirs(path_to_processed_masks_dir, exist_ok=True)
        os.makedirs(path_to_log_dir, exist_ok=True)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        g = torch.Generator()
        g.manual_seed(42)

        print("Processing masks to get tubed skeleton...")
        for mask_name in sorted(os.listdir(path_to_masks_dir)):
            if not mask_name.endswith('.pt'):
                continue

            mask_path = os.path.join(path_to_masks_dir, mask_name)
            
            # Load .pt tensor and convert to NumPy if needed
            mask_tensor = torch.load(mask_path)
            mask = mask_tensor.squeeze().numpy()

            # Apply your transformation (assumes it takes NumPy array and returns NumPy)
            processed_mask_np = tubed_skeleton_transform(mask)

            # Convert back to tensor
            processed_mask_tensor = torch.from_numpy(processed_mask_np)

            # Save as .pt
            save_path = os.path.join(path_to_processed_masks_dir, mask_name)
            torch.save(processed_mask_tensor, save_path)

        print("Masks processed.")

        print("Loading dataset...")

        dataset = Data_Loader(image_dir=path_to_images_dir, mask_dir=path_to_masks_dir, processed_mask_dir=path_to_processed_masks_dir)
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size_is, shuffle=True, num_workers=4, prefetch_factor=4, pin_memory=True, persistent_workers=True, worker_init_fn=seed_worker, generator=g)

        model = UNet3D(in_channels=1, out_channels=num_classes, init_features=16)
        model.to(device)

        criterion = combined_loss_function(num_of_class=num_classes)
        with open(os.path.join(path_to_log_dir, 'hyperparams.txt'), 'w') as f:
            f.write(f'Total Epochs : {total_epochs}\nBatch Size : {batch_size_is}\nLearning Rate : {learning_rate}\n{"-"*30}')

        trainer = Trainer(model, train_loader, epochs=total_epochs, lr=learning_rate, device=device, loss=criterion, log_dir=path_to_log_dir)

        print("Starting training...")

        trainer.train()
