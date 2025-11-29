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

# sys.path.append('/home/ayush/segmentation')
from Alter_data_loader import Data_Loader, Data_Loader_Test
from Alter_Trainer import Trainer
from AlterLoss import NDLoss
from Alter_predict import predict
# from segnet import SegNet
import albumentations as A
from transformers import SegformerForSemanticSegmentation
from concurrent.futures import ProcessPoolExecutor, as_completed


# === Paths & Config ===
input_dir = '/home/nitin1/segmentation/cityscapes'
path_to_images_dir = os.path.join(input_dir, "imagesTr")
path_to_masks_dir = os.path.join(input_dir, "labelsTr")
test_images_dir = os.path.join(input_dir, "imagesTs")
total_epochs = 1000
batch_size_is = 2
learning_rate = 1e-5
num_classes = 8
# seeds = [1337, 1234, 999, 2024, 2025]
seeds = [2025]
log_dir_parent = f"/home/nitin1/segmentation/Results_segformer/Alter_{total_epochs}/Cityscapes"
model_path = '/home/nitin1/segmentation/Codes/segformer/num_classes_8'

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

# === Albumentations Transform (Resize) ===
resize_transform = A.Compose([
    A.Resize(512, 512)
])

def process_mask_file(args):
    fname, path_to_masks_dir, path_to_alter_masks, num_classes = args
    try:
        mask_path = os.path.join(path_to_masks_dir, fname)
        alter_mask_path = os.path.join(
            path_to_alter_masks, os.path.splitext(fname)[0] + ".npy"
        )
        mask_np = np.array(Image.open(mask_path))
        mask_resized = resize_transform(image=mask_np)["image"]
        # one_hot = np.eye(num_classes, dtype=np.float32)[mask_resized]
        os.makedirs(path_to_alter_masks, exist_ok=True)
        np.save(alter_mask_path, mask_resized)
        # np.save(alter_mask_path, one_hot)

        return fname
    except Exception as e:
        print(f"[ERROR] {fname}: {e}")
        return None


# === Parallel Execution ===
def generate_altered_masks(path_to_masks_dir, path_to_alter_masks, num_classes, num_workers=None):
    image_files = [
        f for f in os.listdir(path_to_masks_dir)
        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif'))
    ]
    total = len(image_files)
    print(f"Found {total} images to process.")

    tasks = [(fname, path_to_masks_dir, path_to_alter_masks, num_classes)
             for fname in image_files]

    processed = 0
    num_cpus = num_workers or len(os.sched_getaffinity(0))  # safer than os.cpu_count()
    with ProcessPoolExecutor(max_workers=num_cpus) as executor:
        futures = [executor.submit(process_mask_file, t) for t in tasks]
        for future in as_completed(futures):
            if future.result() is not None:
                processed += 1

    print(f"All {processed} altered masks generated.")

# === Training Function (Per Seed Per GPU) ===
def train_on_seed(seed, primary_gpu_id, split_loss_gpu=False):
    num_gpus_available = torch.cuda.device_count()
    model_device = torch.device(f"cuda:{primary_gpu_id}")

    # Decide whether to split model and loss
    if split_loss_gpu and (num_gpus_available > primary_gpu_id + 1):
        loss_gpu_id = primary_gpu_id + 1
        loss_device = torch.device(f"cuda:{loss_gpu_id}")
        print(f"[Multi-GPU Split Mode] Model on {model_device}, Loss on {loss_device}")
    else:
        loss_device = model_device
        print(f"[Single-GPU Mode] Model and Loss share {model_device}")
    
    # Set seed and device
    torch.cuda.set_device(primary_gpu_id)
    set_seed(seed)

    # === Logging and Paths ===
    path_to_log_dir = os.path.join(log_dir_parent, f'seed_{seed}')
    os.makedirs(path_to_log_dir, exist_ok=True)
    path_to_alter_masks = os.path.join(path_to_log_dir, "altered_masks")
    os.makedirs(path_to_alter_masks, exist_ok=True)
    path_to_visualizations = os.path.join(path_to_log_dir, 'visualizations')
    os.makedirs(path_to_visualizations, exist_ok=True)

    # === Data Loader ===
    g = torch.Generator()
    g.manual_seed(42)
    generate_altered_masks(path_to_masks_dir, path_to_alter_masks, num_classes, num_workers=8)
    dataset = Data_Loader(
        image_dir=path_to_images_dir, mask_dir=path_to_masks_dir,
        alter_mask_dir=path_to_alter_masks, num_classes=num_classes
    )
    train_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size_is, shuffle=True, num_workers=8,
        pin_memory=True, persistent_workers=True, worker_init_fn=seed_worker, generator=g
    )

    num_samples = len(dataset)

    # === Model & Loss ===
    model = SegformerForSemanticSegmentation.from_pretrained(model_path)
    model.to(model_device)
    criterion = NDLoss(
        num_samples=num_samples, num_classes=num_classes,
        num_epochs=total_epochs, batch_size=batch_size_is,
        primary_device=model_device, device=loss_device,
        viz_path=path_to_visualizations
    )

    trainer = Trainer(
        model, train_loader, epochs=total_epochs, lr=learning_rate,
        device=model_device, loss=criterion, log_dir=path_to_log_dir
    )

    print(f"Starting Training on {model_device} for seed {seed}...")
    trainer.train()
    print(f"Ended Training for seed {seed} on {model_device}.")

# === Main Execution ===
if __name__ == "__main__":
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        raise RuntimeError("No GPU available!")

    # --- Parse input argument ---
    split_loss_flag = False
    if len(sys.argv) > 1 and sys.argv[1].lower() in ['true', '1', 'yes']:
        split_loss_flag = True
        print("[CONFIG] Running in split model/loss GPU mode.")
    else:
        print("[CONFIG] Running in single-GPU per process mode.")

    processes = []
    current_gpu = 0

    for seed in seeds:
        # Wait if all GPUs are busy
        while len(processes) >= num_gpus:
            for p in processes:
                if not p.is_alive():
                    p.join()
                    processes.remove(p)
            time.sleep(5)

        print(f"Launching seed {seed} process on GPU {current_gpu}")
        p = mp.Process(target=train_on_seed, args=(seed, current_gpu, split_loss_flag))
        p.start()
        processes.append(p)

        # If in split mode, each process uses 2 GPUs
        current_gpu = (current_gpu + (2 if split_loss_flag else 1)) % num_gpus

    for p in processes:
        p.join()