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
        self.scaler = torch.amp.GradScaler('cuda')
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


            for batch in self.dataloader:
                images = batch['image'].to(self.device)
                masks = batch['mask'].squeeze(1).to(self.device).long()
                alter_masks = batch['alter_mask'].to(self.device)
                alter_mask_paths = batch['alter_mask_path']
                # print(f'Epoch {epoch}')
                # print(f"Unique values in images: {torch.unique(images)}")
                # print(f"Unique values in masks: {torch.unique(masks)}")
                # print(f"Unique values in alter_masks: {torch.unique(alter_masks)}")
                self.optimizer.zero_grad()
                with torch.autocast('cuda', enabled = True):
                    outputs = self.model(images)
                    loss, dice, ce = self.criterion(outputs, masks, alter_masks, alter_mask_paths, epoch)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)      
                self.scaler.update()

                dice_score = DiceScore(outputs, masks)
                epoch_loss += loss.item()
                epoch_dice_loss += dice.item()
                epoch_ce += ce.item()
                epoch_dice += dice_score.item()

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
