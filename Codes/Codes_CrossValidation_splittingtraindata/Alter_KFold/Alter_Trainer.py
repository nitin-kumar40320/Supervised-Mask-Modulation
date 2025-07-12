import torch
import torch.nn as nn
import torch.optim as optim
import time
import matplotlib.pyplot as plt
import os
from Alter_dice import DiceScore
from PIL import Image, ImageDraw, ImageFont
from torchvision.transforms.functional import to_pil_image

class Trainer:
    def __init__(self, model, train_loader, val_loader, epochs, lr, device, loss, log_dir, epoch_vis):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.epochs = epochs
        self.device = device
        self.criterion = loss
        self.epoch_viz_mask = epoch_vis
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.scheduler = optim.lr_scheduler.LinearLR(self.optimizer, start_factor=1.0, end_factor=0.0, total_iters=epochs)

        # self.scaler = torch.amp.GradScaler("cuda")

        self.latest_checkpoint_path = os.path.join(log_dir, "latest_checkpoint_path.pth")
        self.best_checkpoint_path = os.path.join(log_dir, "best_checkpoint_path.pth")
        self.best_ema_score = 0.0

        self.train_losses = []
        self.train_dice_scores = []
        self.val_dice_scores = []

        os.makedirs(log_dir, exist_ok=True)
        self.log_file = os.path.join(log_dir, "training_log.csv")
        self.plot_file = os.path.join(log_dir, "training_plot.png")
        self.save_dir = os.path.join(log_dir, "epoch_visualizations")
        os.makedirs(self.save_dir, exist_ok=True)

        with open(self.log_file, 'w') as f:
            f.write("Epoch,Train Loss,Train Dice Loss,Train CE Loss,Train Dice,Val Dice,Time,Slope\n")

        self.font = ImageFont.load_default()

    def train(self):
        
        for epoch in range(1, self.epochs + 1):
            self.model.train()
            t_loss, t_dice, t_dice_loss, t_ce = 0.0, 0.0, 0.0, 0.0
            start_time = time.time()

            for images, masks, alter_masks, alter_mask_paths in self.train_loader:
                images = images.to(self.device)
                masks = masks.to(self.device)
                alter_masks = alter_masks.to(self.device)

                self.optimizer.zero_grad()

                with torch.autocast(device_type="cuda", enabled = False):
                    outputs = self.model(images)
                    loss, dice, ce = self.criterion(outputs, masks, alter_masks, alter_mask_paths)

                loss.backward()
                self.optimizer.step()

                dice_score = DiceScore(outputs, masks)
                t_loss += loss.item()
                t_dice_loss += dice.item()
                t_ce += ce.item()
                t_dice += dice_score.item()

                with torch.no_grad():
                    preds = torch.argmax(outputs, dim=1, keepdim=True)
                    alter_masks_argmax = torch.argmax(alter_masks, dim=1, keepdim=True)
                    for i, path in enumerate(alter_mask_paths):
                        if self.epoch_viz_mask in path:
                            img = self.normalize_image(images[i].float().cpu())
                            pred_img = self.normalize_image(preds[i].expand(3, -1, -1).float().cpu())
                            gt_img = self.normalize_image(masks[i].float().cpu())
                            alt_img = self.normalize_image(alter_masks_argmax[i].float().cpu())

                            img_pil = to_pil_image(img)
                            pred_pil = to_pil_image(pred_img)
                            gt_pil = to_pil_image(gt_img)
                            alt_pil = to_pil_image(alt_img)

                            w, h = pred_pil.size
                            header_height = 30
                            combined = Image.new("RGB", (4 * w, h + header_height), (0, 0, 0, 0))
                            draw = ImageDraw.Draw(combined)

                            combined.paste(img_pil, (0, header_height))
                            combined.paste(pred_pil, (w, header_height))
                            combined.paste(gt_pil, (2 * w, header_height))
                            combined.paste(alt_pil, (3 * w, header_height))

                            draw.text((w // 2 - 30, 5), "Input", fill="white", font=self.font)
                            draw.text((w + w // 2 - 40, 5), "Prediction", fill="white", font=self.font)
                            draw.text((2 * w + w // 2 - 45, 5), "Ground Truth", fill="white", font=self.font)
                            draw.text((3 * w + w // 2 - 45, 5), "Altered Mask", fill="white", font=self.font)

                            save_path = os.path.join(self.save_dir, f"epoch_{epoch}.png")
                            combined.save(save_path)

            self.scheduler.step()
            t_loss /= len(self.train_loader)
            t_dice /= len(self.train_loader)
            t_dice_loss /= len(self.train_loader)
            t_ce /= len(self.train_loader)

            self.train_losses.append(t_loss)
            self.train_dice_scores.append(t_dice)

            # Validation Dice only
            v_dice = self.validate()
            self.val_dice_scores.append(v_dice)

            epoch_time = time.time() - start_time
            gamma = self.criterion.alter.Alter(epoch, self.epochs)

            with open(self.log_file, 'a') as f:
                f.write(f"{epoch},{t_loss:.4f},{t_dice_loss:.4f},{t_ce:.4f},{t_dice:.4f},{v_dice:.4f},{epoch_time:.2f},{gamma:.6f}\n")

            torch.save(self.model.state_dict(), self.latest_checkpoint_path)

            if epoch == 1:
                ema_score = v_dice
            else:
                ema_score = 0.9 * ema_score + 0.1 * v_dice

            if ema_score > self.best_ema_score:
                self.best_ema_score = ema_score
                torch.save(self.model.state_dict(), self.best_checkpoint_path)

        self.plot_metrics()

    def validate(self):
        self.model.eval()
        total_dice = 0.0

        with torch.no_grad():
            for images, masks in self.val_loader:
                images, masks = images.to(self.device), masks.to(self.device)
                outputs = self.model(images)
                dice_score = DiceScore(outputs, masks)
                total_dice += dice_score.item()

        return total_dice / len(self.val_loader)

    def plot_metrics(self):
        fig, ax1 = plt.subplots(figsize=(10, 5))
        epochs_range = range(1, self.epochs + 1)

        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Dice Score', color='blue')
        ax1.plot(epochs_range, self.train_dice_scores, label='Train Dice', color='blue')
        ax1.plot(epochs_range, self.val_dice_scores, label='Val Dice', color='green')
        ax1.tick_params(axis='y', labelcolor='blue')

        ax2 = ax1.twinx()
        ax2.set_ylabel('Loss', color='red')
        ax2.plot(epochs_range, self.train_losses, label='Train Loss', color='red')
        ax2.tick_params(axis='y', labelcolor='red')

        lines_1, labels_1 = ax1.get_legend_handles_labels()
        lines_2, labels_2 = ax2.get_legend_handles_labels()
        fig.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper center', ncol=3)

        plt.title('Train Dice, Val Dice, and Loss')
        fig.tight_layout()
        plt.savefig(self.plot_file)
        plt.show()

    def normalize_image(self, tensor_img):
        img = tensor_img.clone()
        img = (img - img.min()) / (img.max() - img.min() + 1e-5)
        img = (img * 255).byte()
        return img
