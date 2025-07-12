import torch
from PIL import Image
import cv2
import numpy as np
import torch.optim as optim
import time
import matplotlib.pyplot as plt
import os
from mask_modulation_transform import transform
from matplotlib import cm
import matplotlib.colors as mcolors

from dice import DiceLoss

# Define a list of RGB tuples (in 0-1 range)
color_list = [
    (0.0, 0.0, 0.0),     # black
    (1.0, 0.0, 0.0),     # red
    (1.0, 1.0, 0.0),     # yellow (optional extra)
    (0.0, 1.0, 0.0),     # green
    (0.0, 0.0, 1.0),     # blue
]

custom_cmap = mcolors.ListedColormap(color_list)

class Trainer:
    def __init__(self, model, train_dataloader, val_dataloader, epochs, pretrain_epochs, num_classes, focus_classes, lr, device, loss, log_dir, pred_mask_dir, gt_maskpath_list, transformed_mask_dir, epoch_vis_img):
        self.model = model.to(device)
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.epochs = epochs
        self.pretrain_epochs = pretrain_epochs
        self.device = device
        self.criterion = loss
        self.num_classes = num_classes
        self.epoch_vis_img = epoch_vis_img
        self.focus_classes = focus_classes
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.scheduler = optim.lr_scheduler.LinearLR(self.optimizer, start_factor=1.0, end_factor=0.0, total_iters=epochs)
        self.img_save_dir = os.path.join(log_dir,"epoch_outputs")

        self.latest_checkpoint_path = os.path.join(log_dir, "latest_checkpoint_path.pth")
        self.best_checkpoint_path = os.path.join(log_dir, "best_checkpoint_path.pth")
        self.pred_mask_dir = pred_mask_dir
        self.gt_mask_list = gt_maskpath_list
        self.transformed_mask_dir = transformed_mask_dir

        self.best_dice = 0.0
        self.train_losses = []
        self.train_dice_scores = []
        self.val_dice_scores = []
        
        # Create log directory if it doesn't exist
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(self.img_save_dir, exist_ok=True)
        self.log_file = os.path.join(log_dir, "training_log.csv")
        self.plot_file = os.path.join(log_dir, "training_plot.png")

        # Initialize log file
        with open(self.log_file, 'w') as f:
            f.write("Epoch, Loss, , Val Dice Score, Train Dice Score, Dice Loss, CE Loss, rsl Loss, Time\n")

    def _mask_modulation(self):

        for gt_mask_path in self.gt_mask_list:
            mask_name = os.path.basename(gt_mask_path)
            pred_mask_path = os.path.join(self.pred_mask_dir, mask_name)
            output_path = os.path.join(self.transformed_mask_dir, os.path.splitext(mask_name)[0] + '.npy')

            pred_mask = np.array(Image.open(pred_mask_path).convert('L'))
            gt_mask = np.array(Image.open(gt_mask_path).convert('L'))

            modulated_mask = transform(pred_mask, gt_mask, self.num_classes, self.focus_classes)

            np.save(output_path, modulated_mask)


    def save_comparison_image(self, rgb_image, gt_mask, pred_mask, output_path, num_classes):

        # Define unnormalize (ImageNet example — adjust if different)
        imagenet_mean = [0.485, 0.456, 0.406]
        imagenet_std = [0.229, 0.224, 0.225]

        # Convert from tensor if needed
        if hasattr(rgb_image, 'numpy'):
            rgb_image = rgb_image.detach().cpu().numpy()
        if hasattr(gt_mask, 'numpy'):
            gt_mask = gt_mask.detach().cpu().numpy()
        if hasattr(pred_mask, 'numpy'):
            pred_mask = pred_mask.detach().cpu().numpy()

        # Handle shape [3, H, W] to [H, W, 3]
        if rgb_image.ndim == 3 and rgb_image.shape[0] == 3:
            # Unnormalize
            for c in range(3):
                rgb_image[c] = rgb_image[c] * imagenet_std[c] + imagenet_mean[c]
            rgb_image = np.transpose(rgb_image, (1, 2, 0))

        # Clip and scale to [0, 255]
        rgb_image = np.clip(rgb_image * 255, 0, 255).astype(np.uint8)

        h, w = gt_mask.shape

        # Colormap
        # colormap = cm.get_cmap('tab10', num_classes)
        colormap = custom_cmap
        def colorize_mask(mask):
            colored = colormap(mask / (num_classes - 1))[:, :, :3]
            return (colored * 255).astype(np.uint8)

        gt_colored = colorize_mask(gt_mask)
        pred_colored = colorize_mask(pred_mask)

        # Add banner
        def add_title(image, title):
            banner_height = 30
            banner = np.ones((banner_height, image.shape[1], 3), dtype=np.uint8) * 255
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(banner, title, (10, int(banner_height * 0.75)), font, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
            return np.vstack((banner, image))

        rgb_image = add_title(rgb_image, "Input")
        gt_colored = add_title(gt_colored, "Ground Truth")
        pred_colored = add_title(pred_colored, "Predicted")

        # Concatenate and save
        combined = np.concatenate([rgb_image, gt_colored, pred_colored], axis=1)
        combined_bgr = cv2.cvtColor(combined, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path, combined_bgr)

    def validate(self):

        self.model.eval()

        epoch_dice = 0.0

        self.dice = DiceLoss(epsilon=1e-6, num_classes=self.num_classes)

        for images, masks, mask_names in self.val_dataloader:
            images, masks = images.to(self.device), masks.to(self.device)
            
            outputs = self.model(images)

            dice_loss = self.dice(outputs, masks)             
            
            dice_score = 1-dice_loss.item()
            epoch_dice += dice_score

        epoch_dice /= len(self.train_dataloader)
        self.val_dice_scores.append(epoch_dice)
        
        if epoch_dice > self.best_dice:
            self.best_dice = epoch_dice
            torch.save(self.model.state_dict(), self.best_checkpoint_path)

        return epoch_dice

    def train(self):

        for epoch in range(1, self.epochs + 1):
            self.model.train()
            epoch_loss = 0.0
            epoch_dice_loss = 0.0
            epoch_ce_loss = 0.0
            epoch_rsl_loss = 0.0
            epoch_dice = 0.0
            start_time = time.time()


            for images, masks, transformed_masks, mask_name in self.train_dataloader:
                                    
                images, masks, transformed_masks = images.to(self.device), masks.to(self.device), transformed_masks.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(images)

                loss_tuple = self.criterion(outputs, masks, transformed_masks, epoch, self.pretrain_epochs)
                loss = loss_tuple[0]
                loss.backward()
                self.optimizer.step()

                #saving the outputs
                pred_mask_paths = [os.path.join(self.pred_mask_dir, mask_name[i]) for i in range(len(mask_name))]

                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                predicted_masks = torch.argmax(probabilities, dim=1)
                
                for i in range(predicted_masks.size(0)):
                    mask_path = pred_mask_paths[i]
                    cv2.imwrite(mask_path, predicted_masks[i].cpu().detach().numpy())
                #saving end
                
                dice_loss = loss_tuple[1]
                dice_score = 1-dice_loss
                ce_loss = loss_tuple[2]
                rsl_loss = loss_tuple[3]
                epoch_loss += loss.item()
                epoch_dice_loss += dice_loss.item()
                epoch_ce_loss += ce_loss.item()
                epoch_rsl_loss += rsl_loss.item()
                epoch_dice += dice_score.item()

                for i, path in enumerate(mask_name):
                        if self.epoch_vis_img in path:
                            probabilities = torch.nn.functional.softmax(outputs, dim=1)
                            predicted_masks = torch.argmax(probabilities, dim=1)
                            output_path = os.path.join(self.img_save_dir,f"{epoch}epoch_{mask_name[i]}")

                            self.save_comparison_image(images[i].float().cpu().numpy(), masks[i].float().cpu().numpy(), predicted_masks[i].cpu().detach().numpy(), output_path, self.num_classes)

            self.scheduler.step()       #step the learning rate scheduler

            if epoch >= self.pretrain_epochs:       #tranforming the mask for each epoch after pretraining
                self._mask_modulation()      
            
            epoch_loss /= len(self.train_dataloader)
            epoch_dice_loss /= len(self.train_dataloader)
            epoch_ce_loss /= len(self.train_dataloader)
            epoch_rsl_loss /= len(self.train_dataloader)
            epoch_dice /= len(self.train_dataloader)
            self.train_losses.append(epoch_loss)
            self.train_dice_scores.append(epoch_dice)
            epoch_time = time.time() - start_time
            
            torch.save(self.model.state_dict(), self.latest_checkpoint_path)
            epoch_dice_val = self.validate()

            with open(self.log_file, 'a') as f:
                f.write(f"{epoch}, {epoch_loss:.4f}, {epoch_dice_val:.4f}, {epoch_dice:.4f}, {epoch_dice_loss:.4f}, {epoch_ce_loss:.4f}, {epoch_rsl_loss:.4f}, {epoch_time:.2f}\n")
            
        
        self.plot_metrics()
    
    def plot_metrics(self):
        # Plot training loss
        plt.figure(figsize=(8, 5))
        plt.plot(range(1, self.epochs + 1), self.train_losses, label='Training Loss', color='red')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training Loss Over Epochs')
        plt.legend()
        loss_plot_file = self.plot_file.replace('.png', '_loss.png')
        plt.savefig(loss_plot_file)
        plt.show()

        # Plot dice scores (training + validation)
        plt.figure(figsize=(8, 5))
        plt.plot(range(1, self.epochs + 1), self.train_dice_scores, label='Training Dice Score', color='blue')
        plt.plot(range(1, self.epochs + 1), self.val_dice_scores, label='Validation Dice Score', color='green')
        plt.xlabel('Epochs')
        plt.ylabel('Dice Score')
        plt.title('Dice Scores Over Epochs')
        plt.legend()
        dice_plot_file = self.plot_file.replace('.png', '_dice.png')
        plt.savefig(dice_plot_file)
        plt.show()
