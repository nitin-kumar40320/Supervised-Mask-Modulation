import torch
import torch.optim as optim
import time
import matplotlib.pyplot as plt
import os
from Alter_dice import DiceScore
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor

# from PIL import Image, ImageDraw, ImageFont
# from torchvision.transforms.functional import to_pil_image


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
        self.checkpoints = os.path.join(log_dir, "checkpoints")
        self.checkpoint_futures = []
        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(self.checkpoints, exist_ok=True)
        self.executor = ThreadPoolExecutor(max_workers=8)

        # Initialize log file
        with open(self.log_file, 'w') as f:
            f.write("Epoch, Loss, Dice Loss, Cross-Entropy Loss, Dice Score, Time, Slope\n")

    def _save_checkpoint_async(self, state_dict, path):
        """Worker function to save the model state dictionary."""
        torch.save(state_dict, path)

    def train(self):
        for epoch in range(1, self.epochs + 1):
            self.model.train()
            epoch_loss = 0.0
            epoch_dice = 0.0
            epoch_dice_loss = 0.0
            epoch_ce = 0.0
            start_time = time.time()

            for images, masks, alter_masks, alter_mask_paths in self.dataloader:

                # t0 = time.perf_counter()
                images, masks, alter_masks = images.to(self.device), masks.to(self.device), alter_masks.to(self.device)
                if self.device.type == 'cuda': torch.cuda.synchronize()
                # print(f'Data Loading and transfer : {(time.perf_counter() - t0)}')

                self.optimizer.zero_grad()
                # t0 = time.perf_counter()
                logits = self.model(images).logits
                outputs = torch.nn.functional.interpolate(logits, size=(masks.shape[1], masks.shape[2]), mode='bilinear', align_corners=False)
                # print(f'Model Output : {(time.perf_counter() - t0)}')

                # t0 = time.perf_counter()
                loss, dice, ce = self.criterion(outputs, masks, alter_masks, alter_mask_paths)
                # print(f'Loss Calculation : {(time.perf_counter() - t0)}')

                loss.backward()
                self.optimizer.step()

                
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
            
            # t0 = time.perf_counter()
            gamma = self.criterion.alter.Alter(epoch)
            # print(f'Gamma Calculation : {(time.perf_counter() - t0)}')
            # print(f"Epoch {epoch}: Loss={epoch_loss:.4f}, Dice Loss={epoch_dice_loss:.4f}, KL Div={epoch_ce:.4f}, Dice Score={epoch_dice:.4f}, Time={epoch_time:.2f}s, Slope={gamma:.2f}")
            
            # t0 = time.perf_counter()
            with open(self.log_file, 'a') as f:
                f.write(f"{epoch}, {epoch_loss:.4f}, {epoch_dice_loss:.4f}, {epoch_ce:.4f}, {epoch_dice:.4f}, {epoch_time:.2f}, {gamma:2f}\n")
            
            state_dict = self.model.state_dict()

            if self.checkpoint_futures:
                concurrent.futures.wait(self.checkpoint_futures)
                self.checkpoint_futures.clear()

            # Latest Checkpoint Save
            future = self.executor.submit(self._save_checkpoint_async, state_dict, self.latest_checkpoint_path)
            self.checkpoint_futures.append(future)

            # Best Checkpoint Save (Conditional)
            if epoch_dice > self.best_dice:
                self.best_dice = epoch_dice
                future = self.executor.submit(self._save_checkpoint_async, state_dict, self.best_checkpoint_path)
                self.checkpoint_futures.append(future)

            # Periodic Checkpoint Save (Conditional)
            if (epoch) % 500 == 0:
                checkpoint_path = os.path.join(self.checkpoints, f"checkpoint_epoch_{epoch+1}.pth")
                future = self.executor.submit(self._save_checkpoint_async, state_dict, checkpoint_path)
                self.checkpoint_futures.append(future)

            # print(f'Logging and Saving : {(time.perf_counter() - t0)}')
            self.criterion.synchronize_io()

        if self.checkpoint_futures:
                concurrent.futures.wait(self.checkpoint_futures)
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
