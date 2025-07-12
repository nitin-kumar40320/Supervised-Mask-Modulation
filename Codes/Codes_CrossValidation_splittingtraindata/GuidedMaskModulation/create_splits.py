import os
import json
import random
from sklearn.model_selection import KFold

source_images_dir = "/home/ayush/segmentation/Dataset065_Cracks/imagesTr"
source_masks_dir = "/home/ayush/segmentation/Dataset065_Cracks/labelsTr"
dest_dir = "/home/ayush/segmentation/Results_5cv_unet/GMM_1000/Cracks"
n_splits = 5
random_seed = 42

image_filenames = sorted(os.listdir(source_images_dir))

data_pairs = []
for fname in image_filenames:
    image_path = os.path.join(source_images_dir, fname)
    mask_path = os.path.join(source_masks_dir, fname.replace("_0000",""))
    
    if os.path.exists(mask_path):
        data_pairs.append({
            "image": image_path,
            "mask": mask_path
        })
    else:
        print(mask_path)
        raise ValueError(f"Mask not found for {fname}")

kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_seed)

for fold_idx, (train_idx, val_idx) in enumerate(kf.split(data_pairs)):
    fold_dir = os.path.join(dest_dir, f"fold_{fold_idx}")
    os.makedirs(fold_dir, exist_ok=True)

    train_data = [data_pairs[i] for i in train_idx]
    val_data   = [data_pairs[i] for i in val_idx]

    with open(os.path.join(fold_dir, "train.json"), 'w') as f:
        json.dump(train_data, f, indent=4)

    with open(os.path.join(fold_dir, "val.json"), 'w') as f:
        json.dump(val_data, f, indent=4)


