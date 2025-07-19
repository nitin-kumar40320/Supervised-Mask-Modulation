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
    def __init__(self, model, dataloader, epochs, pretrain_epochs, num_classes, focus_classes, lr, device, loss, log_dir, pred_mask_dir, gt_mask_dir, transformed_mask_dir):
        self.model = model.to(device)
        self.dataloader = dataloader
        self.epochs = epochs
        self.pretrain_epochs = pretrain_epochs
        self.device = device
        self.criterion = loss
        self.num_classes = num_classes
        self.focus_classes = focus_classes
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.scheduler = optim.lr_scheduler.LinearLR(self.optimizer, start_factor=1.0, end_factor=0.0, total_iters=epochs)
        self.img_save_dir = os.path.join(log_dir,"epoch_outputs")

        self.latest_checkpoint_path = os.path.join(log_dir, "latest_checkpoint_path.pth")
        self.best_checkpoint_path = os.path.join(log_dir, "best_checkpoint_path.pth")
        self.pred_mask_dir = pred_mask_dir
        self.gt_mask_dir = gt_mask_dir
        self.transformed_mask_dir = transformed_mask_dir

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
            f.write("Epoch, Loss, Dice Score, Dice Loss, CE Loss, rsl Loss, Time\n")



    def train(self):

        for epoch in range(1, self.epochs + 1):
            self.model.train()
            epoch_loss = 0.0
            epoch_dice_loss = 0.0
            epoch_ce_loss = 0.0
            epoch_rsl_loss = 0.0
            epoch_dice = 0.0
            start_time = time.time()


            for batch in self.dataloader:
                                                    
                images = batch['image'].to(self.device)
                masks = batch['mask'].squeeze(1).to(self.device).long()
                transformed_masks = batch['transformed_mask'].squeeze(2).to(self.device)
                mask_name = batch['name']

                print(images.shape, masks.shape, transformed_masks.shape)
                
                self.optimizer.zero_grad()
                with torch.autocast(device_type='cuda', enabled=True):
                    outputs = self.model(images)
                    loss_tuple = self.criterion(outputs, masks, transformed_masks, epoch, self.pretrain_epochs)
                
                loss = loss_tuple[0]
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                predicted_masks = torch.argmax(probabilities, dim=1)

                print(f"kdjc {predicted_masks.shape}, {masks.shape}")
                
                if epoch >= self.pretrain_epochs:
                        modulated_masks = transform(predicted_masks.cpu().detach(), masks.cpu(), self.num_classes, self.focus_classes)
                        output_paths = [os.path.join(self.transformed_mask_dir, i) for i in mask_name]
                        print(output_paths)
                        for i, output_path in enumerate(output_paths):
                            torch.save(modulated_masks[i], output_path)
                        

                
                dice_loss = loss_tuple[1]
                dice_score = 1-dice_loss
                ce_loss = loss_tuple[2]
                rsl_loss = loss_tuple[3]
                epoch_loss += loss.item()
                epoch_dice_loss += dice_loss.item()
                epoch_ce_loss += ce_loss.item()
                epoch_rsl_loss += rsl_loss.item()
                epoch_dice += dice_score.item()

                # for i, path in enumerate(mask_name):
                #         if '_001' in path:
                #             probabilities = torch.nn.functional.softmax(outputs, dim=1)
                #             predicted_masks = torch.argmax(probabilities, dim=1)
                #             output_path = os.path.join(self.img_save_dir,f"{epoch}epoch_{mask_name[i]}")

                #             self.save_comparison_image(images[i].float().cpu().numpy(), masks[i].float().cpu().numpy(), predicted_masks[i].cpu().detach().numpy(), output_path, self.num_classes)

            self.scheduler.step()       #step the learning rate scheduler
            
            epoch_loss /= len(self.dataloader)
            epoch_dice_loss /= len(self.dataloader)
            epoch_ce_loss /= len(self.dataloader)
            epoch_rsl_loss /= len(self.dataloader)
            epoch_dice /= len(self.dataloader)
            self.train_losses.append(epoch_loss)
            self.train_dice_scores.append(epoch_dice)
            epoch_time = time.time() - start_time
            
            print(f"Epoch {epoch}: Loss={epoch_loss:.4f}, Dice Score={epoch_dice:.4f}, Dice Loss={epoch_dice_loss:.4f}, CE Loss={epoch_ce_loss:.4f}, rsl Loss={epoch_rsl_loss:.4f}, Time={epoch_time:.2f}s")

            with open(self.log_file, 'a') as f:
                f.write(f"{epoch}, {epoch_loss:.4f}, {epoch_dice:.4f}, {epoch_dice_loss:.4f}, {epoch_ce_loss:.4f}, {epoch_rsl_loss:.4f}, {epoch_time:.2f}\n")
            
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
