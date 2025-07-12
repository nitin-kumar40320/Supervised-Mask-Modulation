import os
import sys
import json
import torch
import shutil
import argparse
import numpy as np
from PIL import Image
from multiprocessing import Process, set_start_method

from Alter_data_loader import Data_Loader, Data_Loader_Test
from Alter_Trainer import Trainer
from AlterLoss import NDLoss
from Alter_predict import predict

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from unet_model import UNet
from enet_model import ENet


# ----- Config -----
input_dir = '../Dataset005_Bombr'
alter_mask_dir = os.path.join(input_dir, "altered_masks")
test_images_dir = os.path.join(input_dir, "imagesTs")

total_epochs = 500
batch_size = 4
learning_rate = 0.1
num_classes = 4
dest_dir = f"../Results_5cv_enet/Alter_{total_epochs}/{input_dir.split('_')[1]}"

# ----- Train One Fold -----
def train_fold(fold_idx, fold_dir, gpu_id):

    torch.cuda.set_device(gpu_id)
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    print(f"[Fold {fold_idx}] Running on GPU {gpu_id} - {device}")

    # Load JSON splits
    with open(os.path.join(fold_dir, 'train.json')) as f:
        train_data = json.load(f)
    with open(os.path.join(fold_dir, 'val.json')) as f:
        val_data = json.load(f)

    train_images = [item['image'] for item in train_data]
    train_masks = [item['mask'] for item in train_data]
    val_images = [item['image'] for item in val_data]
    val_masks = [item['mask'] for item in val_data]


    # Generate Altered Masks
    alter_dir = os.path.join(alter_mask_dir, f'fold_{fold_idx}')
    if os.path.exists(os.path.join(alter_dir)):
        shutil.rmtree(os.path.join(alter_dir))
    os.makedirs(alter_dir, exist_ok=True)

    for mask_path in sorted(train_masks):
        fname = os.path.basename(mask_path)
        out_path = os.path.join(alter_dir, os.path.splitext(fname)[0] + ".npy")
        mask = np.array(Image.open(mask_path))
        one_hot = np.eye(num_classes)[mask]
        np.save(out_path, one_hot.astype(np.float32))
    print(f"[Fold {fold_idx}] Generated altered masks for training images.")

    # Dataloaders
    train_dataset = Data_Loader(train_images, train_masks, alter_dir, num_classes=num_classes)
    val_dataset = Data_Loader(val_images, val_masks, None, num_classes=num_classes, use_altered=False)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True, persistent_workers=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True, persistent_workers=True)

    # Model & Loss
    model = ENet(num_classes=num_classes).to(device)
    loss_fn = NDLoss(num_samples=len(train_dataset), num_classes=num_classes, batch_size=batch_size, device=device)

    # Save hyperparams
    with open(os.path.join(fold_dir, 'hyperparams.txt'), 'w') as f:
        f.write(f"Fold: {fold_idx}\nEpochs: {total_epochs}\nBatch Size: {batch_size}\nLearning Rate: {learning_rate}\n")

    trainer = Trainer(model=model,
                      train_loader=train_loader,
                      val_loader=val_loader,
                      epochs=total_epochs,
                      lr=learning_rate,
                      device=device,
                      loss=loss_fn,
                      log_dir=fold_dir,
                      epoch_vis=os.path.splitext(os.path.basename(train_masks[0]))[0])

    print(f"[Fold {fold_idx}] Training...")
    trainer.train()
    print(f"[Fold {fold_idx}] Training complete")

    # Load best checkpoint and predict
    # model_path = os.path.join(fold_dir, "best_checkpoint_path.pth")
    # model.load_state_dict(torch.load(model_path, map_location=device))

    output_dir = os.path.join(fold_dir, "test_output")
    os.makedirs(output_dir, exist_ok=True)

    test_dataset = Data_Loader_Test(test_images_dir)
    print(f"[Fold {fold_idx}] Predicting on test set...")
    predict(test_dataset, output_dir, model)
    print(f"[Fold {fold_idx}] Completed.")

# ----- Fold Dispatcher -----
def run_selected_folds(folds_to_run):

    try:
        set_start_method('spawn')
    except RuntimeError:
        pass

    all_fold_dirs = sorted([os.path.join(dest_dir, d) for d in os.listdir(dest_dir) if d.startswith('fold_')])
    gpus = list(range(torch.cuda.device_count()))
    if not gpus:
        raise RuntimeError("No GPUs available!")

    print(f"Found {len(gpus)} GPU(s). Dispatching folds: {folds_to_run}")

    fold_idx = 0
    selected_fold_dirs = [all_fold_dirs[i] for i in folds_to_run]

    while fold_idx < len(selected_fold_dirs):
        active_processes = []

        for gpu_id in gpus:
            if fold_idx >= len(selected_fold_dirs):
                break

            actual_fold_idx = folds_to_run[fold_idx]
            fold_dir = selected_fold_dirs[fold_idx]

            print(f"Starting Fold {actual_fold_idx} on GPU {gpu_id}")
            p = Process(target=train_fold, args=(actual_fold_idx, fold_dir, gpu_id))
            p.start()
            active_processes.append((p, actual_fold_idx, gpu_id))
            fold_idx += 1

        for p, f, g in active_processes:
            p.join()
            print(f"Fold {f} on GPU {g} completed.")

    print("\n🎉 All selected folds completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--folds', type=str, default='all', help="Comma-separated folds (e.g., 0,1,3) or 'all'")
    args = parser.parse_args()

    # Detect available folds
    existing_folds = sorted([int(d.split('_')[1]) for d in os.listdir(dest_dir) if d.startswith('fold_')])
    if args.folds.strip().lower() == "all":
        folds = existing_folds
    else:
        input_folds = sorted(set(int(f.strip()) for f in args.folds.split(',')))
        invalid = set(input_folds) - set(existing_folds)
        if invalid:
            raise ValueError(f"Invalid folds requested: {invalid}. Available: {existing_folds}")
        folds = input_folds

    run_selected_folds(folds)

