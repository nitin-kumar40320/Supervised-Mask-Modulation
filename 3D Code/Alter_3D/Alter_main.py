import torch
import shutil
import os
import cv2
import random
import time
import numpy as np
from PIL import Image
import torch.nn.functional as F
import torch.multiprocessing as mp
import sys

sys.path.append('/home/ayush/segmentation')
from Alter_data_loader import Data_Loader #, Data_Loader_Test
from Alter_Trainer import Trainer
from AlterLoss import NDLoss
from Alter_predict import predict
# from segnet import SegNet
from unet_model import UNet3D
import albumentations as A

# === Paths & Config ===
input_dir = '/home/ayush/segmentation/Dataset007_Amos'
path_to_images_dir = os.path.join(input_dir, "imagesSubsetPt")
path_to_masks_dir = os.path.join(input_dir, "labelsSubsetPt")
# test_images_dir = os.path.join(input_dir, "imagesTs")
total_epochs = 400
batch_size_is = 4
learning_rate = 1e-4
num_classes = 4
# seeds = [1337, 1234, 999, 2024, 2025]
seeds = [2024]
log_dir_parent = f"/home/ayush/segmentation/Results/Alter_{total_epochs}/Amos"

# === Seed Functions ===
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

# === Training Function (Per Seed Per GPU) ===
def train_on_seed(seed, gpu_id):
    set_seed(seed)
    torch.cuda.set_device(gpu_id)
    device = torch.device(f"cuda:{gpu_id}")

    # === Logging and Mask Paths ===
    path_to_log_dir = os.path.join(log_dir_parent, f'seed_{seed}_v2')
    os.makedirs(path_to_log_dir, exist_ok=True)

    path_to_alter_masks = os.path.join(path_to_log_dir, "altered_masks")
    os.makedirs(path_to_alter_masks, exist_ok=True)

    # === Generate Altered Masks Specific to This Seed ===
    dims = []
    num_samples = 0
    for fname in os.listdir(path_to_masks_dir):
        if fname.endswith('.pt'):
            mask_path = os.path.join(path_to_masks_dir, fname)
            alter_mask_path = os.path.join(path_to_alter_masks, os.path.splitext(fname)[0] + ".npy")
            mask_np = torch.load(mask_path).squeeze().numpy()
            one_hot = np.eye(num_classes)[mask_np]  # Shape: H x W x C
            if len(dims) == 0:
                dims = list(range(one_hot.ndim))
            one_hot = np.transpose(one_hot, (dims[-1], *dims[:-1]))  # Now shape is C x H x W
            np.save(alter_mask_path, one_hot.astype(np.float32))
            num_samples += 1

    # === Data Loader ===
    g = torch.Generator()
    g.manual_seed(42)

    path_to_visualizations = os.path.join(path_to_log_dir, 'visualizations')
    os.makedirs(path_to_visualizations, exist_ok=True)

    dataset = Data_Loader(
        image_dir=path_to_images_dir,
        mask_dir=path_to_masks_dir,
        alter_mask_dir=path_to_alter_masks,
    )
    train_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size_is, shuffle=True, num_workers=4, prefetch_factor=4,
        pin_memory=True, persistent_workers=True, worker_init_fn=seed_worker, generator=g
    )
    
    # === Model, Loss, Trainer ===
    model = UNet3D(in_channels=1, out_channels=num_classes, init_features=16)
    criterion = NDLoss(
        num_samples=num_samples, num_classes=num_classes,
        num_epochs=total_epochs, batch_size=batch_size_is,
        device=device, viz_path=path_to_visualizations
    )

    with open(os.path.join(path_to_log_dir, 'hyperparams.txt'), 'w') as f:
        f.write(f'Total Epochs : {total_epochs}\nBatch Size : {batch_size_is}\n'
                f'Learning Rate : {learning_rate}\nSeed : {seed}\n{"-"*30}\n')

    trainer = Trainer(
        model, train_loader, epochs=total_epochs, lr=learning_rate,
        device=device, loss=criterion, log_dir=path_to_log_dir
    )

    print(f"Starting Training on GPU {gpu_id} for seed {seed}...")
    trainer.train()
    print(f"Ended Training for seed {seed} on GPU {gpu_id}.")

    # === Inference ===
    # output_mask_dir = os.path.join(path_to_log_dir, "test_output")
    # os.makedirs(output_mask_dir, exist_ok=True)
    # model_path = os.path.join(path_to_log_dir, "best_checkpoint_path.pth")

    # checkpoint = torch.load(model_path, map_location=device)
    # model.load_state_dict(checkpoint)

    # test_dataset = Data_Loader_Test(test_images_dir)
    # print(f"Starting Prediction for seed {seed} on GPU {gpu_id}...")
    # predict(test_dataset, output_mask_dir, model)

# === Main Execution (One Process Per GPU) ===
if __name__ == "__main__":
    num_gpus = torch.cuda.device_count()
    assert num_gpus > 0, "No CUDA GPUs available!"

    print(f"Available GPUs: {num_gpus}")
    print(f"Input Directory: {input_dir}")
    print(f"Input Classes : {num_classes}")

    processes = []
    current_gpu = 0

    for seed in seeds:
        # Wait until any GPU is free
        while len(processes) >= num_gpus:
            for p in processes:
                if not p.is_alive():
                    p.join()
                    processes.remove(p)
            time.sleep(5)

        print(f"Launching seed {seed} on GPU {current_gpu}")
        p = mp.Process(target=train_on_seed, args=(seed, current_gpu))
        p.start()
        processes.append(p)

        current_gpu = (current_gpu + 1) % num_gpus

    # Wait for all processes to complete
    for p in processes:
        p.join()
