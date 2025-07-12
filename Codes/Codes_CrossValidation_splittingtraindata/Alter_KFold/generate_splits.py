import os
import json
import argparse
from sklearn.model_selection import KFold

def make_kfold_splits(dataset_id, n_splits):
    source_dir = f'../{dataset_id}'
    source_images_dir = os.path.join(source_dir, "imagesTr")
    source_masks_dir = os.path.join(source_dir, "labelsTr")
    dataset_id = dataset_id.split('_')[1]
    dest_dir = f"../Results_5cv_enet/Alter/Vanilla_500/{dataset_id}"

    os.makedirs(dest_dir, exist_ok=True)

    image_filenames = sorted(os.listdir(source_images_dir))

    data_pairs = []
    for fname in image_filenames:
        image_path = os.path.join(source_images_dir, fname)
        mask_path = os.path.join(source_masks_dir, fname.replace("_0000", ""))
        if os.path.exists(mask_path):
            data_pairs.append({"image": image_path, "mask": mask_path})
        else:
            print(f"❌ Mask not found: {mask_path}")
            raise ValueError(f"Mask not found for {fname}")

    print(f"✅ Found {len(data_pairs)} valid image-mask pairs.")
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(data_pairs)):
        fold_dir = os.path.join(dest_dir, f"fold_{fold_idx}")
        os.makedirs(fold_dir, exist_ok=True)

        train_data = [data_pairs[i] for i in train_idx]
        val_data = [data_pairs[i] for i in val_idx]

        with open(os.path.join(fold_dir, "train.json"), 'w') as f:
            json.dump(train_data, f, indent=4)
        with open(os.path.join(fold_dir, "val.json"), 'w') as f:
            json.dump(val_data, f, indent=4)

        print(f"✅ Saved fold_{fold_idx} with {len(train_data)} train and {len(val_data)} val samples.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate K-fold splits for a dataset.")
    parser.add_argument('--dataset', type=str, required=True, help="Dataset ID (e.g., Dataset065_Cracks)")
    parser.add_argument('--folds', type=int, default=5, help="Number of folds (default: 5)")
    args = parser.parse_args()

    make_kfold_splits(args.dataset, args.folds)
