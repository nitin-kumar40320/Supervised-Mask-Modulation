import cv2
import numpy as np
import os
import torch
import torch.nn as nn
from Alter_vanilla_loss import VanillaLoss
# from KLDivergenceLoss import KLDivergenceLoss
import torch.nn.functional as F
from collections import deque
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor


class AlterLoss(nn.Module):
    def __init__(self, num_samples, num_classes, batch_size, num_epochs, device):
        super(AlterLoss, self).__init__()
        self.num_samples = num_samples
        print(f'Number of training samples: {self.num_samples} ')
        self.batch_first = (batch_size > 1)
        self.device = device
        self.epochs = num_epochs

        self.original_th = 0
        self.check = True
        self.gamma = torch.tensor(0.0, device=self.device)
        self.queue = deque(maxlen=15)
        self.pretrain = 0.2*(num_epochs)
        self.threshold = - np.inf

        self.fn_buffer = None
    
    def update_mask(self, pred_mask, gt_mask):
        """
        pred_mask, gt_mask: tensors of shape (B, C, H, W) on the self.device.
        Returns:
            updated_masks: tensor of shape (B, C, H, W) on the self.device
        """

        # Updated mask is initialized with GT (shape B, C, H, W)
        updated_masks = gt_mask.clone()
        fn = torch.gt(gt_mask[:, 1:], pred_mask[:, 1:]).float() # 1.0 where FN occurs, 0.0 otherwise

        # Apply dilation using PyTorch MaxPool2d (GPU-accelerated)
        # MaxPool2d with kernel 3, padding 1, stride 1 is equivalent to dilation
        dilated_fn_result = F.max_pool2d(fn, kernel_size=3, padding=1, stride=1)
        
        # Updated mask for foreground classes: union of GT_FG and Dilated_FN
        # The union of two binary masks A and B is (A + B) > 0
        torch.clamp(updated_masks[:, 1:] + dilated_fn_result, 0.0, 1.0, out=updated_masks[:, 1:])

        return updated_masks

    def count_true_positives(self, pred_mask, gt_mask):
        '''
        Input: pred_maks, gt_mask: Tensors of size [B, C, H, W]
        Output: Count of average true positives per class -> type: float
        '''
        tp = pred_mask * gt_mask
        dim = tuple(range(2, pred_mask.ndim))
        tp = tp.sum(dim=dim).mean(dim=1).sum()  # Sum over H, W; mean over batch; sum over classes
        return tp
    
    def compute_slope(self):
        n = len(self.queue)
        if n < 2:
            return 0.0  # Not enough data

        x = torch.arange(n).float()
        # print(self.queue)
        y = torch.tensor(list(self.queue)).float()

        sum_x = torch.sum(x)
        sum_y = torch.sum(y)
        sum_xy = torch.sum(x * y)
        sum_x2 = torch.sum(x * x)

        numerator = n * sum_xy - sum_x * sum_y
        denominator = n * sum_x2 - sum_x * sum_x + 1e-8  # avoid divide-by-zero

        slope = numerator / denominator
        return slope.item()

    def forward(self, pred_mask, gt_mask):
        '''
        Input: pred_mask: Tensor of shape [B, C, H, W] 
        gt_mask: Tensor of shape [B, H, W]
        '''
        # assert gt_mask.ndim==3, f'Expected ground truth mask of shape [B, H, W] in AlterLoss, but got {gt_mask.shape}'
        targets = F.one_hot(gt_mask, num_classes=pred_mask.shape[1]).permute(0, gt_mask.ndim, *range(1, gt_mask.ndim)).float()
        self.gamma+=(self.count_true_positives(pred_mask, targets)/(torch.clamp(torch.sum(targets), min=1)))
        if self.check:
            return None
        else:
            alt_mask = self.update_mask(pred_mask, targets)
            return alt_mask
    
    # To be executed at end of each epoch
    def Alter(self, epoch):
        final_val = self.gamma.item()/self.num_samples
        self.queue.append(final_val)
        self.gamma.zero_()
        slope = self.compute_slope()
        if epoch < self.pretrain:
            self.original_th += slope
        elif epoch == self.pretrain:
            self.original_th += slope
            self.original_th /= self.pretrain
            self.original_th = np.abs(self.original_th)
            self.threshold = self.original_th
            print(f'Starting Threshold : {self.original_th}')
        elif epoch > self.pretrain:
            self.threshold = (1-((epoch-self.pretrain)/(self.epochs-self.pretrain)))*self.original_th
        
        if slope < self.threshold:
            self.check = False
        else:
            self.check = True
        return slope


class NDLoss(nn.Module):
    def __init__(self, num_samples, num_classes, num_epochs, batch_size, primary_device, device, viz_path, epsilon=1e-6):
        super(NDLoss, self).__init__()
        self.vanilla = VanillaLoss(epsilon = epsilon)
        self.alter = AlterLoss(num_samples, num_classes, batch_size, num_epochs, device)
        self.device = device
        self.viz_path = viz_path
        self.executor = ThreadPoolExecutor(max_workers=8)
        self.epoch_save_futures = []
        self.transfer_stream = torch.cuda.Stream(device=primary_device)

    def _save_mask_async(self, mask_data, save_path):
        """Worker function to save a single mask in a separate thread."""
        try:
            np.save(save_path, mask_data)
        except Exception as e:
            print(f"Error saving mask to {save_path}: {e}")

    def forward(self, preds, targets, alter_targets, alter_target_paths):
        '''
        preds: Model output, Shape: [B, C, H, W]
        targets: ground truth masks, Shape: [B, 1, H, W]
        alter_targets: Altered masks, Shape: [B, C, H, W]
        alter_target_paths: list of paths where altered masks are saved
        epoch: Current epoch index
        '''
        with torch.cuda.stream(self.transfer_stream):
            pred_loss = preds.to(self.device, non_blocking=True)
            target_loss = targets.to(self.device, non_blocking=True)
            alter_targets_loss = alter_targets.to(self.device, non_blocking=True)

        with torch.cuda.device(self.device):
            softm = torch.softmax(pred_loss, dim=1)
            pred_class = torch.argmax(softm, dim=1)  # (B, H, W)
            pred_class = F.one_hot(pred_class, num_classes=pred_loss.shape[1]).permute(0, pred_class.ndim, *range(1, pred_class.ndim)).float() # [B, C, H, W]
            target_loss = target_loss.squeeze(1) # [B, H, W]
            with torch.no_grad():
                alter_output = self.alter(pred_class, target_loss)
                if alter_output is not None:
                    alter_targets = alter_output.to(self.device)  # shape: (B, C, H, W)
                    if self.alter.batch_first:
                        
                        for i, (mask, mask_path) in enumerate(zip(alter_targets, alter_target_paths)):
                            
                            # pemute_mask = mask.permute(*range(1, mask.ndim), 0)
                            pemute_mask = torch.argmax(mask, dim=0).to(torch.uint8)  # single-channel, GPU tensor
                            cpu_tensor = pemute_mask.to('cpu', non_blocking=True).contiguous() # (H, W, C) tensor on CPU RAM (staging)
                            save_path = mask_path if mask_path.endswith('.npy') else f"{os.path.splitext(mask_path)[0]}.npy"
                            
                            # 2. Submit the I/O task. The worker thread will implicitly wait for the transfer
                            # to finish when it calls .numpy() and then saves.
                            # We submit a lambda/helper to ensure the .numpy() conversion is done off the main thread.
                            task = self.executor.submit(
                                lambda t, path: np.save(path, t.numpy()),  # The .numpy() call happens on the worker thread
                                cpu_tensor,
                                save_path
                            )
                            self.epoch_save_futures.append(task)
                    else:
                        # For the batch_first = False case (B=1), asynchronous I/O benefit is minimal,
                        # but we can still offload the entire .cpu().numpy() to the worker thread.
                        mask = alter_targets.squeeze(0)
                        cpu_tensor = mask.permute(*range(1, mask.ndim), 0).cpu(non_blocking=True)
                        save_path = alter_target_paths[0] if isinstance(alter_target_paths, list) else alter_target_paths
                        save_path = save_path if save_path.endswith('.npy') else f"{os.path.splitext(save_path)[0]}.npy"
                        
                        # Offload both conversion and saving
                        task = self.executor.submit(
                            lambda t, path: np.save(path, t.numpy()),
                            cpu_tensor,
                            save_path
                        )
                        self.epoch_save_futures.append(task)
            # --- Loss Calculation ---
            vanilla_loss, dice, ce = self.vanilla(pred_loss, softm, alter_targets_loss)
            return vanilla_loss, dice, ce

    
    def synchronize_io(self):
        """
        Blocks the main thread until all mask saving operations from the 
        current epoch are complete. Must be called at the end of each epoch.
        """
        
        # Wait for all futures to complete
        concurrent.futures.wait(self.epoch_save_futures)
        
        # Clear the list for the next epoch
        self.epoch_save_futures = []
        return
    
    def visualize(self, mask, epoch):
        pass
        # mask_np = mask.squeeze().cpu().numpy().astype(np.uint8)  # shape: (H, W), values = class indices
        # mask_color = cv2.applyColorMap(mask_np * int(255 / (mask_np.max() or 1)), cv2.COLORMAP_JET)
        # save_path = os.path.join(self.viz_path, f'Epoch_{epoch}.png')
        # cv2.imwrite(save_path, mask_color)

# --- Test Execution ---
if __name__ == "__main__":
    # --- Helper for Mocking and Timing ---
    import time
    class TimeRecorder:
        """Utility to record time for specific sections."""
        def __init__(self):
            self.records = {'TP_CALC': 0.0, 'LOSS_CALC': 0.0, 'IO_SYNC': 0.0}
            self.start_time = None
            self.current_section = None
        
        def start(self, section):
            self.start_time = time.perf_counter()
            self.current_section = section
        
        def stop(self):
            duration = time.perf_counter() - self.start_time
            self.records[self.current_section] += duration
            return duration

        def display(self, total_batches):
            print("\n--- Performance Breakdown ---")
            for key, value in self.records.items():
                avg_time = value / total_batches
                print(f"  {key:<12}: Total: {value:.6f}s | Avg/Batch: {avg_time:.6f}s")
            print("-----------------------------\n")


    class MockBatch:
        """Represents a single data batch."""
        def __init__(self, preds, targets, alter_targets, paths):
            self.preds = preds
            self.targets = targets
            self.alter_targets = alter_targets
            self.paths = paths

    def create_mock_batch(batch_idx, batch_size, num_classes, H, W, device):
        """Creates a single mock batch."""
        # Model Logits: [B, C, H, W]
        preds = torch.randn(batch_size, num_classes, H, W, device=device)
        
        # Original GT: [B, 1, H, W] (Indices)
        # Note: We ensure some non-zero mask areas for TP calculation
        targets = torch.randint(0, num_classes, (batch_size, 1, H, W), device=device).long()
        
        # Persisted Altered Mask: [B, C, H, W] (Initial One-Hot of GT)
        # The loss function logic will replace this if an update is triggered.
        targets_indices = targets.squeeze(1)
        alter_targets_init = F.one_hot(targets_indices, num_classes=num_classes).permute(0, 3, 1, 2).float().to(device)
        
        # I/O Paths (Ensure they are unique for I/O test)
        paths = [f"testing/mask_e00_b{batch_idx}_i{j}.npy" for j in range(batch_size)]
        
        return MockBatch(preds, targets, alter_targets_init, paths)

    # --- Configuration ---
    # Set for small, observable test
    NUM_EPOCHS = 10
    NUM_BATCHES_PER_EPOCH = 5
    BATCH_SIZE = 2
    NUM_CLASSES = 8
    HEIGHT, WIDTH = 512, 512
    
    # Total samples used for AlterLoss average calculation
    TOTAL_SAMPLES = NUM_BATCHES_PER_EPOCH * BATCH_SIZE 
    
    device = torch.device('cuda:1')
    primary_device = torch.device('cuda:0')

    viz_path = "testing/test_visualization"

    os.makedirs('testing', exist_ok=True)
    os.makedirs(viz_path, exist_ok=True)
    
    # Instantiate modules
    loss_module = NDLoss(TOTAL_SAMPLES, NUM_CLASSES, NUM_EPOCHS, BATCH_SIZE, primary_device, device, viz_path)
    recorder = TimeRecorder()
    
    # --- Setup AlterLogic for Test ---
    # We set a very low threshold (negative) for the first epoch to guarantee 
    # self.check remains TRUE (Pretrain/No Alteration).
    loss_module.alter.threshold = -100.0
    loss_module.alter.pretrain = 2.0 # End pretrain

    print(f"--- Starting Optimized Loss Test on {device} ---")
    print(f"Target Threshold (Pre-train): {loss_module.alter.threshold}")
    
    total_batches_processed = 0

    # --- Training Simulation Loop ---
    for epoch in range(NUM_EPOCHS):
        print(f"\n======== Epoch {epoch+1}/{NUM_EPOCHS} (Check: {loss_module.alter.check}) ========")
        
        # Manually set check to FALSE for the second epoch to trigger async I/O
        if epoch == 2:
            loss_module.alter.check = False
            print("--- ALTER MODE ACTIVATED (Manual Trigger) ---")

        for batch_idx in range(NUM_BATCHES_PER_EPOCH):
            total_batches_processed += 1
            
            # Create a fresh batch
            batch = create_mock_batch(batch_idx, BATCH_SIZE, NUM_CLASSES, HEIGHT, WIDTH, device)
            
            # 1. TP Calculation & Async I/O Submission (if triggered)
            # This measures the time for the synchronous TP calculation (non-deferred)
            recorder.start('TP_CALC')
            loss, dice, ce = loss_module(batch.preds, batch.targets, batch.alter_targets, batch.paths)
            recorder.stop()
            
            # 2. Loss Calculation (Simulated training work)
            # This represents the time spent on the core loss and backprop (should overlap with I/O)
            recorder.start('LOSS_CALC')
            # Simulate backprop and optimizer step here
            _ = loss.item() # Use loss item to force evaluation of the loss graph
            recorder.stop()
            
            is_saving = "ON" if loss_module.alter.check == False else "OFF"
            print(f"  Batch {batch_idx+1}: Loss={loss.item():.4f} | I/O: {is_saving} (Futures: {len(loss_module.epoch_save_futures)})")


        # --- END OF EPOCH SYNC ---
        
        # 3. Synchronize I/O (Blocking, should be fast if not saving, slow if saving)
        recorder.start('IO_SYNC')
        loss_module.synchronize_io() 
        recorder.stop()
        
        # 4. Update AlterLogic for next epoch
        slope = loss_module.alter.Alter(epoch + 1)
        print(f"  EPOCH {epoch+1} SYNC Time: {recorder.records['IO_SYNC'] - recorder.records['IO_SYNC']:.6f}s")
        print(f"  Logic Update: Slope={slope:.6f} | Check: {loss_module.alter.check}")
        
    # --- Final Cleanup and Timing Report ---
    loss_module.executor.shutdown(wait=True)
    recorder.display(total_batches_processed)