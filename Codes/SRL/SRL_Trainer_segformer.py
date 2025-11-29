import torch
import torch.nn as nn
import torch.optim as optim
import time
import matplotlib.pyplot as plt
import os

class Segformer_Trainer:
    def __init__(self, model, dataloader, epochs, lr, device, loss, log_dir, save_checkpoint=500):
        self.model = model.to(device)
        self.dataloader = dataloader
        self.epochs = epochs
        self.device = device
        self.criterion = loss
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.scheduler = optim.lr_scheduler.LinearLR(self.optimizer, start_factor=1.0, end_factor=0.0, total_iters=epochs)
        self.save = save_checkpoint
        
        self.latest_checkpoint_path = os.path.join(log_dir, "latest_checkpoint_path.pth")
        self.best_checkpoint_path = os.path.join(log_dir, "best_checkpoint_path.pth")
        self.checkpoints = os.path.join(log_dir, "checkpoints")
        os.makedirs(self.checkpoints, exist_ok=True)
        self.best_dice = 0.0
        self.train_losses = []
        self.train_dice_scores = []
        
        # Create log directory if it doesn't exist
        os.makedirs(log_dir, exist_ok=True)
        self.log_file = os.path.join(log_dir, "training_log.csv")
        self.plot_file = os.path.join(log_dir, "training_plot.png")

        # Initialize log file
        with open(self.log_file, 'w') as f:
            f.write("Epoch, Loss, Dice Score, Dice Loss, CE Loss, SkelRec Loss, Time\n")

        # self.img_save_dir = ''

    # def save_comparison_image(self, rgb_image, gt_mask, pred_mask, output_path, num_classes):

    #     # Define unnormalize (ImageNet example — adjust if different)
    #     imagenet_mean = [0.485, 0.456, 0.406]
    #     imagenet_std = [0.229, 0.224, 0.225]

    #     # Convert from tensor if needed
    #     if hasattr(rgb_image, 'numpy'):
    #         rgb_image = rgb_image.detach().cpu().numpy()
    #     if hasattr(gt_mask, 'numpy'):
    #         gt_mask = gt_mask.detach().cpu().numpy()
    #     if hasattr(pred_mask, 'numpy'):
    #         pred_mask = pred_mask.detach().cpu().numpy()

    #     # Handle shape [3, H, W] to [H, W, 3]
    #     if rgb_image.ndim == 3 and rgb_image.shape[0] == 3:
    #         # Unnormalize
    #         for c in range(3):
    #             rgb_image[c] = rgb_image[c] * imagenet_std[c] + imagenet_mean[c]
    #         rgb_image = np.transpose(rgb_image, (1, 2, 0))

    #     # Clip and scale to [0, 255]
    #     rgb_image = np.clip(rgb_image * 255, 0, 255).astype(np.uint8)

    #     h, w = gt_mask.shape

    #     # Colormap
    #     # colormap = cm.get_cmap('tab10', num_classes)
    #     colormap = custom_cmap
    #     def colorize_mask(mask):
    #         colored = colormap(mask / (num_classes - 1))[:, :, :3]
    #         return (colored * 255).astype(np.uint8)

    #     gt_colored = colorize_mask(gt_mask)
    #     pred_colored = colorize_mask(pred_mask)

    #     # Add banner
    #     def add_title(image, title):
    #         banner_height = 30
    #         banner = np.ones((banner_height, image.shape[1], 3), dtype=np.uint8) * 255
    #         font = cv2.FONT_HERSHEY_SIMPLEX
    #         cv2.putText(banner, title, (10, int(banner_height * 0.75)), font, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
    #         return np.vstack((banner, image))

    #     rgb_image = add_title(rgb_image, "Input")
    #     gt_colored = add_title(gt_colored, "Ground Truth")
    #     pred_colored = add_title(pred_colored, "Predicted")

    #     # Concatenate and save
    #     combined = np.concatenate([rgb_image, gt_colored, pred_colored], axis=1)
    #     combined_bgr = cv2.cvtColor(combined, cv2.COLOR_RGB2BGR)
    #     cv2.imwrite(output_path, combined_bgr)

    def train(self):
        for epoch in range(1, self.epochs + 1):
            self.model.train()
            epoch_loss = 0.0
            epoch_dice_loss = 0.0
            epoch_ce_loss = 0.0
            epoch_skelrec_loss = 0.0
            epoch_dice = 0.0
            start_time = time.time()

            for images, masks, transformed_masks, mask_names in self.dataloader:

                transformed_masks = transformed_masks.long()
                images, masks, transformed_masks = images.to(self.device), masks.to(self.device), transformed_masks.to(self.device)
                
                self.optimizer.zero_grad()
                logits = self.model(images).logits
                # logits: [N,C,H_model,W_model], masks: [N,H_target,W_target]
                outputs = torch.nn.functional.interpolate(logits, size=(masks.shape[1], masks.shape[2]), mode='bilinear', align_corners=False)
                loss_tuple = self.criterion(outputs, masks, transformed_masks)
                loss = loss_tuple[0]
                loss.backward()
                self.optimizer.step()

                
                dice_loss = loss_tuple[1]
                dice_score = 1-dice_loss
                ce_loss = loss_tuple[2]
                skelrec_loss = loss_tuple[3]
                epoch_loss += loss.item()
                epoch_dice_loss += dice_loss.item()
                epoch_ce_loss += ce_loss.item()
                epoch_skelrec_loss += skelrec_loss.item()
                epoch_dice += dice_score.item()

                # for i, path in enumerate(mask_names):
                #     with torch.no_grad():
                #         if '_001' in path:
                #             probabilities = torch.nn.functional.softmax(outputs, dim=1)
                #             predicted_masks = torch.argmax(probabilities, dim=1)
                #             output_path = os.path.join(self.img_save_dir,f"{epoch}epoch_{mask_names[i]}")
                #             self.save_comparison_image(images[i].float().cpu().numpy(), masks[i].float().cpu().numpy(), predicted_masks[i].cpu().detach().numpy(), output_path, self.num_classes)


            self.scheduler.step()
            
            epoch_loss /= len(self.dataloader)
            epoch_dice_loss /= len(self.dataloader)
            epoch_ce_loss /= len(self.dataloader)
            epoch_skelrec_loss /= len(self.dataloader)
            epoch_dice /= len(self.dataloader)
            self.train_losses.append(epoch_loss)
            self.train_dice_scores.append(epoch_dice)
            epoch_time = time.time() - start_time
            
            print(f"Epoch {epoch}: Loss={epoch_loss:.4f}, Dice Score={epoch_dice:.4f}, Dice Loss={epoch_dice_loss:.4f}, CE Loss={epoch_ce_loss:.4f}, SkelRec Loss={epoch_skelrec_loss:.4f}, Time={epoch_time:.2f}s")
            with open(self.log_file, 'a') as f:
                f.write(f"{epoch}, {epoch_loss:.4f}, {epoch_dice:.4f}, {epoch_dice_loss:.4f}, {epoch_ce_loss:.4f}, {epoch_skelrec_loss:.4f}, {epoch_time:.2f}\n")
            
            torch.save(self.model.state_dict(), self.latest_checkpoint_path)
            if epoch_dice > self.best_dice:
                self.best_dice = epoch_dice
                torch.save(self.model.state_dict(), self.best_checkpoint_path)
            if epoch % self.save == 0:
                checkpoint_path = os.path.join(self.checkpoints, f"checkpoint_epoch_{epoch}.pth")
                torch.save(self.model.state_dict(), checkpoint_path)

        
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
