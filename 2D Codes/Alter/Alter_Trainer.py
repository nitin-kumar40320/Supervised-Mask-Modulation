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
    def __init__(self, model, dataloader, epochs, lr, device, loss, log_dir):
        self.model = model.to(device)
        self.dataloader = dataloader
        self.epochs = epochs
        self.device = device
        self.criterion = loss
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.scheduler = optim.lr_scheduler.LinearLR(self.optimizer, start_factor=1.0, end_factor=0.0, total_iters=epochs)
        self.latest_checkpoint_path = os.path.join(log_dir, "latest_checkpoint_path.pth")
        self.best_checkpoint_path = os.path.join(log_dir, "best_checkpoint_path.pth")
        self.best_dice = 0.0
        self.train_losses = []
        self.train_dice_scores = []
        # Create log directory if it doesn't exist
        os.makedirs(log_dir, exist_ok=True)
        self.log_file = os.path.join(log_dir, "training_log.csv")
        self.plot_file = os.path.join(log_dir, "training_plot.png")
        self.save_dir = os.path.join(log_dir, "epoch_visualizations")
        os.makedirs(self.save_dir, exist_ok=True)

        # Initialize log file
        with open(self.log_file, 'w') as f:
            f.write("Epoch, Loss, Dice Loss, Cross-Entropy Loss, Dice Score, Time, Slope\n")

    def train(self):
        for epoch in range(1, self.epochs + 1):
            self.model.train()
            epoch_loss = 0.0
            epoch_dice = 0.0
            epoch_dice_loss = 0.0
            epoch_ce = 0.0
            start_time = time.time()


            for images, masks, alter_masks, alter_mask_paths in self.dataloader:
                images, masks, alter_masks = images.to(self.device), masks.to(self.device), alter_masks.to(self.device)
                # print(f'Epoch {epoch}')
                # print(f"Unique values in images: {torch.unique(images)}")
                # print(f"Unique values in masks: {torch.unique(masks)}")
                # print(f"Unique values in alter_masks: {torch.unique(alter_masks)}")
                self.optimizer.zero_grad()
                outputs = self.model(images)
                # print(f"Unique values in output: {torch.unique(outputs)}")
                # print('-'*20)
                loss, dice, ce = self.criterion(outputs, masks, alter_masks, alter_mask_paths, epoch)
                loss.backward()
                self.optimizer.step()

                
                dice_score = DiceScore(outputs, masks)
                epoch_loss += loss.item()
                epoch_dice_loss += dice.item()
                epoch_ce += ce.item()
                epoch_dice += dice_score.item()

                # Collage of prediction, Ground Truth and Altered Mask
                with torch.no_grad():
                    preds = torch.argmax(outputs, dim=1, keepdim=True)  # Shape: (B, 1, H, W)
                    alter_masks = torch.argmax(alter_masks, dim = 1, keepdim = True)
                    for i, path in enumerate(alter_mask_paths):
                        if '_001' in path:
                            img = self.normalize_image(images[i].float().cpu())
                            pred_img = self.normalize_image(preds[i].expand(3, -1, -1).float().cpu())
                            gt_img = self.normalize_image(masks[i].float().cpu())
                            alt_img = self.normalize_image(alter_masks[i].float().cpu())
                            
                            img_pil = to_pil_image(img)
                            pred_pil = to_pil_image(pred_img)
                            gt_pil = to_pil_image(gt_img)
                            alt_pil = to_pil_image(alt_img)

                            w, h = pred_pil.size
                            header_height = 30

                            combined = Image.new("RGB", (4 * w, h + header_height), (0, 0, 0, 0))

                            draw = ImageDraw.Draw(combined)
                            font = ImageFont.load_default()

                            combined.paste(img_pil, (0, header_height))
                            combined.paste(pred_pil, (w, header_height))
                            combined.paste(gt_pil, (2 * w, header_height))
                            combined.paste(alt_pil, (3 * w, header_height))

                            draw.text((w // 2 - 30, 5), "Input", fill="white", font=font)
                            draw.text((w + w // 2 - 40, 5), "Prediction", fill="white", font=font)
                            draw.text((2 * w + w // 2 - 45, 5), "Ground Truth", fill="white", font=font)
                            draw.text((3 * w + w // 2 - 45, 5), "Altered Mask", fill="white", font=font)


                            base_name = f"epoch_{epoch}"
                            save_path = os.path.join(self.save_dir, f"{base_name}.png")
                            combined.save(save_path)
            self.scheduler.step()
            epoch_loss /= len(self.dataloader)
            epoch_dice /= len(self.dataloader)
            epoch_dice_loss /= len(self.dataloader)
            epoch_ce /= len(self.dataloader)
            self.train_losses.append(epoch_loss)
            self.train_dice_scores.append(epoch_dice)
            epoch_time = time.time() - start_time

            gamma = self.criterion.alter.Alter(epoch)

            # print(f"Epoch {epoch}: Loss={epoch_loss:.4f}, Dice Loss={epoch_dice_loss:.4f}, KL Div={epoch_ce:.4f}, Dice Score={epoch_dice:.4f}, Time={epoch_time:.2f}s, Slope={gamma:.2f}")
            with open(self.log_file, 'a') as f:
                f.write(f"{epoch}, {epoch_loss:.4f}, {epoch_dice_loss:.4f}, {epoch_ce:.4f}, {epoch_dice:.4f}, {epoch_time:.2f}, {gamma:2f}\n")
            
            torch.save(self.model.state_dict(), self.latest_checkpoint_path)
            if epoch_dice > self.best_dice:
                self.best_dice = epoch_dice
                torch.save(self.model.state_dict(), self.best_checkpoint_path)
        
        self.plot_metrics()
    
    def plot_metrics(self):
        plt.figure(figsize=(10,5))
        plt.plot(range(1, self.epochs+1), self.train_losses, label='Loss', color='red')
        plt.plot(range(1, self.epochs+1), self.train_dice_scores, label='Dice Score', color='blue')
        plt.xlabel('Epochs')
        plt.ylabel('Metric Value')
        plt.legend()
        plt.title('Training Metrics Over Epochs')
        plt.savefig(self.plot_file)  # Save the plot
        plt.show()
    
    def normalize_image(self, tensor_img):
        img = tensor_img.clone()
        img = (img - img.min()) / (img.max() - img.min() + 1e-5)
        img = (img * 255).byte()
        return img
