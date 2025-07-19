import cv2
import numpy as np
import os
import torch
import torch.nn as nn
from skimage.morphology import dilation
from Alter_vanilla_loss import VanillaLoss
# from KLDivergenceLoss import KLDivergenceLoss
import torch.nn.functional as F
from collections import deque

class AlterLoss(nn.Module):
    def __init__(self, num_samples, num_classes, batch_size, num_epochs, device):
        super(AlterLoss, self).__init__()
        self.gamma = 0 # accumulation factor
        self.original_th = 0
        # print(f'Original Threshold : {self.original_th}')
        self.check = True
        self.num_samples = num_samples
        print(f'Number of training samples: {self.num_samples} ')
        self.batch_first = (batch_size > 1)
        self.device = device
        self.queue = deque(maxlen=15)
        self.epochs = num_epochs
        self.pretrain = 0.2*(num_epochs)
        self.threshold = - np.inf
    
    def update_mask(self, pred_mask, gt_mask):
        """
        pred_mask, gt_mask: tensors of shape (B, C, ..), where C is the number of classes.
        Returns:
            updated_masks: tensor of shape (B, C, ..)
        """
        pred_mask_np = pred_mask.cpu().numpy()  # (B, C, ..)
        gt_mask_np = gt_mask.cpu().numpy()      # (B, C, ..)

        dims = pred_mask_np.shape[2:]

        # Initialize updated_masks array
        updated_masks = np.zeros_like(gt_mask_np, dtype=np.float32)

        # Background (class 0) remains unchanged
        updated_masks[:, 0] = gt_mask_np[:, 0].astype(np.float32)

        # Compute FN: where GT=1 and prediction=0 (shape: B, C-1, ..)
        fn = ((gt_mask_np[:, 1:] - pred_mask_np[:, 1:]) > 0).astype(np.uint8)

        # Flatten B and C-1 into one loop dimension
        B, C1 = fn.shape[:2]
        fn_flat = fn.reshape(-1, *dims)
        gt_flat = gt_mask_np[:, 1:].reshape(-1, *dims)

        # Apply dilation on each FN mask
        dilated_fn_flat = [dilation(f) for f in fn_flat]

        # Updated mask: union of GT and dilated FN
        updated_flat = [((gt + dil_fn) > 0).astype(np.float32)
                        for gt, dil_fn in zip(gt_flat, dilated_fn_flat)]

        # Reshape back to (B, C-1, ..)
        updated_masks[:, 1:] = np.stack(updated_flat).reshape(B, C1, *dims)
        return torch.tensor(updated_masks)

    
    def count_true_positives(self, pred_mask, gt_mask):
        '''
        Input: pred_maks, gt_mask: Tensors of size [B, C, ..]
        Output: Count of average true positives per class -> type: float
        '''
        tp = pred_mask * gt_mask
        dim = tuple(range(2, pred_mask.ndim))
        tp = tp.sum(dim=dim)
        tp_per_image = tp.mean(dim=1)
        tp_per_image = tp_per_image.sum().item()
        assert isinstance(tp_per_image, float), f"tp_per_image is {type(tp_per_image)}, with value {tp_per_image}"
        return tp_per_image
    
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
        Input: pred_mask: Tensor of shape [B, C, ..] 
        gt_mask: Tensor of shape [B, ..]
        '''
        # assert gt_mask.ndim==3, f'Expected ground truth mask of shape [B, ..] in AlterLoss, but got {gt_mask.shape}'
        targets = F.one_hot(gt_mask, num_classes=pred_mask.shape[1]).permute(0, gt_mask.ndim, *range(1, gt_mask.ndim)).float()
        self.gamma+=(self.count_true_positives(pred_mask, targets)/(torch.clamp(torch.sum(targets), min=1))).item()
        if self.check:
            return None
        else:
            alt_mask = self.update_mask(pred_mask, targets)
            return alt_mask
    
    # To be executed at end of each epoch
    def Alter(self, epoch):
        self.gamma = self.gamma/self.num_samples
        self.queue.append(self.gamma)
        # print(f'self.gamma after epoch {epoch} : {self.gamma}')
        self.gamma = 0
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
    def __init__(self, num_samples, num_classes, num_epochs, batch_size, device, viz_path, epsilon=1e-6):
        super(NDLoss, self).__init__()
        self.vanilla = VanillaLoss(epsilon = epsilon)
        self.alter = AlterLoss(num_samples, num_classes, batch_size, num_epochs, device)
        self.device = device
        self.viz_path = viz_path 
    
    def forward(self, preds, targets, alter_targets, alter_target_paths, epoch):
        '''
        preds: Model output, Shape: [B, C, ..]
        targets: ground truth masks, Shape: [B, 1, ..]
        alter_targets: Altered masks, Shape: [B, C, ..]
        alter_target_paths: list of paths where altered masks are saved
        epoch: Current epoch index
        '''
        with torch.autocast('cuda', enabled = False):
            softm = torch.softmax(preds.float(), dim=1)
            pred_class = torch.argmax(softm, dim=1)  # (B, ..)
            # print(f'Shape of preds : {preds.shape} -- ALterLoss.py')
            pred_class = F.one_hot(pred_class, num_classes=preds.shape[1]).permute(0, pred_class.ndim, *range(1, pred_class.ndim)).float() # [B, C, ..]
        # print(f'Shape of pred_class : {pred_class.shape} -- AlterLoss.py')
        # targets = F.one_hot(targets, num_classes=preds.shape[1]).permute(0, 3, 1, 2).float() # [B, C, ..]
        # alter_targets = alter_targets.squeeze(1)
        
        with torch.no_grad():
            alter_output = self.alter(pred_class, targets)

            if alter_output is not None:
                alter_targets = alter_output  # shape: (B, C, ..)
                if self.alter.batch_first:
                    for i, (mask, mask_path) in enumerate(zip(alter_targets, alter_target_paths)):
                        save_path = mask_path if mask_path.endswith('.npy') else f"{os.path.splitext(mask_path)[0]}.npy"
                        np.save(save_path, mask.cpu().numpy())
                        # if '_0001' in mask_path:
                        #     self.visualize(mask, epoch)
                else:
                    save_path = alter_target_paths[0] if alter_target_paths[0].endswith('.npy') else f"{os.path.splitext(alter_target_paths)[0][0]}.npy"
                    np.save(save_path, mask.squeeze().cpu().numpy())
                    # if '_0001' in alter_target_paths[0]:
                    #     self.visualize(mask, epoch)

        vanilla_loss, dice, ce = self.vanilla(preds.to(self.device), softm.to(self.device), alter_targets.to(self.device))
        return vanilla_loss, dice, ce

    def visualize(self, mask, epoch):
        pass
        # mask_np = mask.squeeze().cpu().numpy().astype(np.uint8)  # shape: (..), values = class indices
        # mask_color = cv2.applyColorMap(mask_np * int(255 / (mask_np.max() or 1)), cv2.COLORMAP_JET)
        # save_path = os.path.join(self.viz_path, f'Epoch_{epoch}.png')
        # cv2.imwrite(save_path, mask_color)


# Testcase
if __name__ == "__main__":
    import time
    # Configuration
    os.makedirs('testing', exist_ok=True)
    batch_size = 4
    num_classes = 4
    height, width, depth = 64, 64, 64  # 3D mask dimensions
    num_samples = 100
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    viz_path = "testing/test_visualization"
    os.makedirs(viz_path, exist_ok=True)

    # Instantiate loss
    loss_module = NDLoss(num_samples, num_classes, num_epochs=50, batch_size=batch_size, device=device, viz_path=viz_path)
    alter_module = loss_module.alter

    # Generate random predictions and ground truth
    preds = torch.randn(batch_size, num_classes, height, width, depth)
    targets = torch.randint(0, num_classes, (batch_size, height, width, depth)).long()
    alter_targets = torch.zeros_like(preds)
    alter_target_paths = [f"testing/mask_000{i}_0000.npy" for i in range(batch_size)]

    # Apply softmax + argmax + one-hot
    with torch.no_grad():
        softm = torch.softmax(preds, dim=1)
        pred_class = torch.argmax(softm, dim=1)  # Shape: (B, H, W, D)
        pred_class = F.one_hot(pred_class, num_classes=num_classes).permute(0, 4, 1, 2, 3).float()

    # Also one-hot encode ground truth
    targets_one_hot = F.one_hot(targets, num_classes=num_classes).permute(0, 4, 1, 2, 3).float()

    # Time the transformation
    torch.cuda.synchronize() if device.type == 'cuda' else None
    start = time.time()
    updated = alter_module.update_mask(pred_class.to(device), targets_one_hot.to(device))
    torch.cuda.synchronize() if device.type == 'cuda' else None
    end = time.time()

    total_time = end - start
    print(f"\nUpdated masks shape: {updated.shape}")
    print(f"Time taken for transformation (batch of {batch_size}): {total_time:.4f} seconds")
    print(f"Average time per mask: {total_time / batch_size:.4f} seconds")
