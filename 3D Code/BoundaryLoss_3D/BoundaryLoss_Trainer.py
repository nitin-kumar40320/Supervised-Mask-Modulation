import torch
import torch.nn as nn
import torch.optim as optim
import time
import matplotlib.pyplot as plt
import os
import cv2
import numpy as np
from matplotlib import cm
import matplotlib.colors as mcolors

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
    def __init__(self, model, dataloader, epochs, num_classes, lr, device, loss, log_dir):
        self.model = model.to(device)
        self.dataloader = dataloader
        self.epochs = epochs
        self.device = device
        self.criterion = loss
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.scheduler = optim.lr_scheduler.LinearLR(self.optimizer, start_factor=1.0, end_factor=0.0, total_iters=epochs)
        self.img_save_dir = os.path.join(log_dir,"epoch_outputs")
        self.num_classes = num_classes


        self.latest_checkpoint_path = os.path.join(log_dir, "latest_checkpoint_path.pth")
        self.best_checkpoint_path = os.path.join(log_dir, "best_checkpoint_path.pth")
        self.best_dice = 0.0
        self.train_losses = []
        self.train_dice_scores = []

        self.scaler = torch.amp.GradScaler('cuda', enabled=True)
        
        # Create log directory if it doesn't exist
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(self.img_save_dir, exist_ok=True)
        self.log_file = os.path.join(log_dir, "training_log.csv")
        self.plot_file = os.path.join(log_dir, "training_plot.png")

        # Initialize log file
        with open(self.log_file, 'w') as f:
            f.write("Epoch, Loss, Dice Score, Dice Loss, CE Loss, Boundary Loss, Time\n")

    
    def train(self):
        for epoch in range(1, self.epochs + 1):
            self.model.train()
            epoch_loss = 0.0
            epoch_dice_loss = 0.0
            epoch_ce_loss = 0.0
            epoch_boundary_loss = 0.0
            epoch_dice = 0.0
            start_time = time.time()


            for batch in self.dataloader:
                images = batch['image'].to(self.device)
                masks = batch['mask'].squeeze(1).to(self.device).long()
                dist_map = batch['dist_map'].to(self.device).float()

                
                self.optimizer.zero_grad()

                with torch.autocast(device_type='cuda', enabled=True):
                    outputs = self.model(images)
                    loss_tuple = self.criterion(outputs, masks, dist_map)
                
                loss = loss_tuple[0]
                dice_loss = loss_tuple[1]
                ce_loss = loss_tuple[2]
                boundary_loss = loss_tuple[3]

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

                
                dice_score = 1-dice_loss.item()
                epoch_loss += loss.item()
                epoch_dice_loss += dice_loss.item()
                epoch_ce_loss += ce_loss.item()
                epoch_boundary_loss += boundary_loss.item()
                epoch_dice += dice_score

                # for i, path in enumerate(mask_names):
                #     if '_001' in path:
                #         probabilities = torch.nn.functional.softmax(outputs, dim=1)
                #         predicted_masks = torch.argmax(probabilities, dim=1)
                #         output_path = os.path.join(self.img_save_dir,f"{epoch}epoch_{mask_names[i]}")

                #         self.save_comparison_image(images[i].float().cpu().numpy(), masks[i].float().cpu().numpy(), predicted_masks[i].cpu().detach().numpy(), output_path, self.num_classes)


            self.scheduler.step()
            
            epoch_loss /= len(self.dataloader)
            epoch_dice_loss /= len(self.dataloader)
            epoch_ce_loss /= len(self.dataloader)
            epoch_boundary_loss /= len(self.dataloader)
            epoch_dice /= len(self.dataloader)
            self.train_losses.append(epoch_loss)
            self.train_dice_scores.append(epoch_dice)
            epoch_time = time.time() - start_time
            
            with open(self.log_file, 'a') as f:
                f.write(f"{epoch}, {epoch_loss:.4f}, {epoch_dice:.4f}, {epoch_dice_loss:.4f}, {epoch_ce_loss:.4f}, {epoch_boundary_loss:.4f}, {epoch_time:.2f}\n")
            
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
