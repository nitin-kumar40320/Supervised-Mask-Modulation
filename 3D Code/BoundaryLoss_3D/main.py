import torch
import numpy as np
import random
from data_loader import Data_Loader
from BoundaryLoss_Trainer import Trainer
from combined_loss import CombinedLoss
import sys
sys.path.append('/home/nitin1/segmentation')
from unet_model import UNet3D
import os
from scipy.ndimage import distance_transform_edt


total_epochs = 400
batch_size_is = 4
learning_rate = 1e-4
num_classes = 4 ## including background
classes_to_focus_on = [1,2,3]
# seeds = [1337, 1234, 999, 2024, 2025]
seeds = [2025]
path_to_images_dir = "/home/nitin1/segmentation/Dataset007_Amos/imagesSubsetPt"
path_to_masks_dir = "/home/nitin1/segmentation/Dataset007_Amos/labelsSubsetPt"
log_dir = f"/home/nitin1/segmentation/Results_unet/BoundaryLoss_{total_epochs}/Amos"


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

def class2one_hot_3d(seg: torch.Tensor, K: int) -> torch.Tensor:
    # Input: [D, H, W] → Output: [K, D, H, W]
    one_hot = torch.nn.functional.one_hot(seg.long(), num_classes=K)
    return one_hot.permute(3, 0, 1, 2).contiguous()

def one_hot2dist_3d(seg, resolution=[1, 1, 1], dtype=np.float32):
    K = len(seg)
    res = np.zeros_like(seg, dtype=dtype)
    for k in range(K):
        posmask = seg[k].astype(bool)
        if posmask.any():
            negmask = ~posmask
            res[k] = distance_transform_edt(negmask, sampling=resolution) * negmask \
                - (distance_transform_edt(posmask, sampling=resolution) - 1) * posmask
    return res



if __name__ == "__main__":

    output_dir = os.path.join(log_dir, 'dist_maps')
    os.makedirs(output_dir, exist_ok=True)

    for file_name in sorted(os.listdir(path_to_masks_dir)):
        if not file_name.endswith('.pt'):
            continue

        input_path = os.path.join(path_to_masks_dir, file_name)
        output_path = os.path.join(output_dir, file_name)

        if os.path.exists(output_path):
            continue

        # Load segmentation tensor
        seg = torch.load(input_path)  # Expected shape: [D, H, W]

        seg = seg.squeeze().long()  # Remove batch if needed, ensure int type
        one_hot = class2one_hot_3d(seg, num_classes)  # Shape: [K, D, H, W]
        dist_map = one_hot2dist_3d(one_hot.numpy())  # Apply distance transform
        dist_tensor = torch.from_numpy(dist_map).to(torch.float32)  # Convert back to tensor

        torch.save(dist_tensor, output_path)

    for seed in seeds:

        set_seed(seed)

        path_to_log_dir = os.path.join(log_dir, f'seed_{seed}')
        os.makedirs(path_to_log_dir, exist_ok=True)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        g = torch.Generator()
        g.manual_seed(42)

        dataset = Data_Loader(image_dir=path_to_images_dir, mask_dir=path_to_masks_dir, dist_map_dir=output_dir)
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size_is, shuffle=True, prefetch_factor=4, num_workers=4, pin_memory=True, persistent_workers=True, worker_init_fn=seed_worker, generator=g)

        model = UNet3D(in_channels=1, out_channels=num_classes, init_features=16)
        torch.cuda.empty_cache()
        model.to(device)

        with open(os.path.join(path_to_log_dir, 'hyperparams.txt'), 'w') as f:
            f.write(f'Total Epochs : {total_epochs}\nBatch Size : {batch_size_is}\nLearning Rate : {learning_rate}\nDevice : {device}\n{"-"*30}')
        criterion = CombinedLoss(num_class = num_classes, focus_classes = classes_to_focus_on)

        trainer = Trainer(model, train_loader, epochs=total_epochs, num_classes=num_classes, lr=learning_rate, device=device, loss=criterion, log_dir=path_to_log_dir)

        print("Starting training...")

        trainer.train()
