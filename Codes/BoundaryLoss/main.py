import torch
import numpy as np
import random
from data_loader import Data_Loader
from BoundaryLoss_Trainer import Trainer
from BoundaryLoss_Trainer_segformer import Segformer_Trainer
from combined_loss import CombinedLoss
import sys
sys.path.append('/Codes')
from unet_model import UNet
from segnet import SegNet
from transformers import SegformerForSemanticSegmentation
import os


total_epochs = 1000
batch_size_is = 2
learning_rate = 1e-5
num_classes = 8 ## including background
classes_to_focus_on = [0,1,2,3,4,5,6,7]
# seeds = [1337, 1234, 999, 2024, 2025]
seeds = [999, 2024]
path_to_images_dir = "/cityscapes/imagesTr"
path_to_masks_dir = "/cityscapes/labelsTr"
log_dir = f"/Results_segformer/BoundaryLoss_{total_epochs}/cityscapes"
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

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        g = torch.Generator()
        g.manual_seed(42)

        dataset = Data_Loader(image_dir=path_to_images_dir, mask_dir=path_to_masks_dir, K=num_classes)
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size_is, shuffle=True, prefetch_factor=4, num_workers=4, pin_memory=True, persistent_workers=True, worker_init_fn=seed_worker, generator=g)

        # model = UNet(in_channels=3, out_channels=num_classes, init_features=32)
        # model = SegNet(in_chn=3, out_chn=num_classes)
        model = SegformerForSemanticSegmentation.from_pretrained(model_path)
        model.to(device)

        with open(os.path.join(path_to_log_dir, 'hyperparams.txt'), 'w') as f:
            f.write(f'Total Epochs : {total_epochs}\nBatch Size : {batch_size_is}\nLearning Rate : {learning_rate}\nDevice : {device}\n{"-"*30}\nData Processing : \n{dataset.general_transform}')
        criterion = CombinedLoss(num_class = num_classes, focus_classes = classes_to_focus_on)

        trainer = Segformer_Trainer(model, train_loader, epochs=total_epochs, num_classes=num_classes, lr=learning_rate, device=device, loss=criterion, log_dir=path_to_log_dir)

        print("Starting training...")

        trainer.train()
