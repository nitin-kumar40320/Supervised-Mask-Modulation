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
    def __init__(self, num_samples, num_classes, batch_size, device):
        super(AlterLoss, self).__init__()
        self.gamma = 0 # accumulation factor
        # self.threshold = threshold
        self.original_th = np.log10(num_samples)/((num_samples)*np.exp(num_classes))
        print(f'Original Threshold : {self.original_th}')
        # self.prev_tp = 0
        self.check = True
        self.num_samples = num_samples
        print(f'Number of training samples: {self.num_samples} ')
        self.batch_first = (batch_size > 1)
        self.device = device
        self.queue = deque(maxlen=15)
    
    def update_mask(self, pred_mask, gt_mask):
        """
        pred_mask, gt_mask: tensors of shape (B, C, H, W), where C is the number of classes.
        Returns:
            updated_masks: tensor of shape (B, C, H, W)
        """
        pred_mask_np = pred_mask.cpu().numpy()  # (B, C, H, W)
        gt_mask_np = gt_mask.cpu().numpy()      # (B, C, H, W)

        # batch_size, num_classes = pred_mask_np.shape[0], pred_mask_np.shape[1]
        # updated_masks = []
        # for b in range(batch_size):
        #     updated_per_sample = []
        #     updated_per_sample.append(gt_mask_np[b, 0].astype(np.float32))
        #     for c in range(1, num_classes):
        #         pred_c = pred_mask_np[b, c]  # (H, W)
        #         gt_c = gt_mask_np[b, c]      # (H, W)

        #         fn = ((gt_c - pred_c) > 0).astype(np.uint8)
        #         dilated_fn = dilation(fn)
        #         updated_mask = ((gt_c + dilated_fn) > 0).astype(np.float32)
        #         updated_per_sample.append(updated_mask)

        #     updated_per_sample = np.stack(updated_per_sample)  # (C, H, W)
        #     updated_masks.append(updated_per_sample)

        # updated_masks = np.stack(updated_masks)  # (B, C, H, W)
        # updated_masks = torch.tensor(updated_masks, dtype=torch.float32)
        # updated_masks = updated_masks.argmax(dim=1)
        H, W = pred_mask_np.shape[2:]

        # Initialize updated_masks array
        updated_masks = np.zeros_like(gt_mask_np, dtype=np.float32)

        # Background (class 0) remains unchanged
        updated_masks[:, 0] = gt_mask_np[:, 0].astype(np.float32)

        # Compute FN: where GT=1 and prediction=0 (shape: B, C-1, H, W)
        fn = ((gt_mask_np[:, 1:] - pred_mask_np[:, 1:]) > 0).astype(np.uint8)

        # Flatten B and C-1 into one loop dimension
        B, C1 = fn.shape[:2]
        fn_flat = fn.reshape(-1, H, W)
        gt_flat = gt_mask_np[:, 1:].reshape(-1, H, W)

        # Apply dilation on each FN mask
        dilated_fn_flat = [dilation(f) for f in fn_flat]

        # Updated mask: union of GT and dilated FN
        updated_flat = [((gt + dil_fn) > 0).astype(np.float32)
                        for gt, dil_fn in zip(gt_flat, dilated_fn_flat)]

        # Reshape back to (B, C-1, H, W)
        updated_masks[:, 1:] = np.stack(updated_flat).reshape(B, C1, H, W)
        return torch.tensor(updated_masks)

    
    def count_true_positives(self, pred_mask, gt_mask):
        '''
        Input: pred_maks, gt_mask: Tensors of size [B, C, H, W]
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
        Input: pred_mask: Tensor of shape [B, C, H, W] 
        gt_mask: Tensor of shape [B, H, W]
        '''
        # assert gt_mask.ndim==3, f'Expected ground truth mask of shape [B, H, W] in AlterLoss, but got {gt_mask.shape}'
        targets = F.one_hot(gt_mask, num_classes=pred_mask.shape[1]).permute(0, gt_mask.ndim, *range(1, gt_mask.ndim)).float()
        self.gamma+=(self.count_true_positives(pred_mask, targets)/(torch.clamp(torch.sum(targets), min=1))).item()
        if self.check:
            return None
        else:
            alt_mask = self.update_mask(pred_mask, targets)
            return alt_mask
    
    # To be executed at end of each epoch
    def Alter(self, epoch, total_epochs):
        self.gamma = self.gamma/self.num_samples
        self.queue.append(self.gamma)
        # print(f'self.gamma after epoch {epoch} : {self.gamma}')
        self.gamma = 0
        slope = self.compute_slope()
        
        threshold = (1-(epoch/total_epochs))*self.original_th
        if slope < threshold and epoch > 0.2*(total_epochs):
            self.check = False
        else:
            self.check = True
        return slope


class NDLoss(nn.Module):
    def __init__(self, num_samples, num_classes, batch_size, device, viz_path, epsilon=1e-6):
        super(NDLoss, self).__init__()
        self.vanilla = VanillaLoss(epsilon = epsilon)
        self.alter = AlterLoss(num_samples, num_classes, batch_size, device)
        self.device = device
        self.viz_path = viz_path 
    
    def forward(self, preds, targets, alter_targets, alter_target_paths, epoch):
        '''
        preds: Model output, Shape: [B, C, H, W]
        targets: ground truth masks, Shape: [B, 1, H, W]
        alter_targets: Altered masks, Shape: [B, C, H, W]
        alter_target_paths: list of paths where altered masks are saved
        epoch: Current epoch index
        '''
        softm = torch.softmax(preds, dim=1)
        pred_class = torch.argmax(softm, dim=1)  # (B, H, W)
        # print(f'Shape of preds : {preds.shape} -- ALterLoss.py')
        pred_class = F.one_hot(pred_class, num_classes=preds.shape[1]).permute(0, pred_class.ndim, *range(1, pred_class.ndim)).float() # [B, C, H, W]
        # print(f'Shape of pred_class : {pred_class.shape} -- AlterLoss.py')
        
        targets = targets.squeeze(1) # [B, H, W]
        # targets = F.one_hot(targets, num_classes=preds.shape[1]).permute(0, 3, 1, 2).float() # [B, C, H, W]
        # alter_targets = alter_targets.squeeze(1)
        
        with torch.no_grad():
            alter_output = self.alter(pred_class, targets)

            if alter_output is not None:
                alter_targets = alter_output  # shape: (B, C, H, W)
                if self.alter.batch_first:
                    for i, (mask, mask_path) in enumerate(zip(alter_targets, alter_target_paths)):
                        save_path = mask_path if mask_path.endswith('.npy') else f"{os.path.splitext(mask_path)[0]}.npy"
                        np.save(save_path, mask.permute(*range(1, mask.ndim), 0).cpu().numpy())
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
        # mask_np = mask.squeeze().cpu().numpy().astype(np.uint8)  # shape: (H, W), values = class indices
        # mask_color = cv2.applyColorMap(mask_np * int(255 / (mask_np.max() or 1)), cv2.COLORMAP_JET)
        # save_path = os.path.join(self.viz_path, f'Epoch_{epoch}.png')
        # cv2.imwrite(save_path, mask_color)


# Testcase
if __name__ == "__main__":
    # Configuration
    os.makedirs('testing', exist_ok=True)
    batch_size = 2
    num_classes = 3
    height, width = 3, 3
    num_samples = 10
    batch_first = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # Use 'cuda' if GPU is available
    viz_path = "testing/test_visualization"

    os.makedirs(viz_path, exist_ok=True)

    # Dummy data
    preds = torch.nn.functional.one_hot(torch.randint(0, num_classes, (batch_size, height, width)), num_classes=num_classes).permute(0, 3, 1, 2).float() # Model logits
    targets = torch.randint(0, num_classes, (batch_size, 1, height, width)).long() # Ground truth
    alter_targets = preds.clone()  # Shape: (B, C, H, W)
    alter_target_paths = [f"testing/mask_000{i}_0000.npy" for i in range(batch_size)]  # Fake save paths
    epoch = 5
    # print(f'Alter_targets : {alter_targets}')
    print(f'Predictions : {preds.shape}, Targets : {targets.shape}, Alter_targets : {alter_targets.shape}')

    # Instantiate the loss module
    loss_module = NDLoss(num_samples, num_classes, batch_size, device, viz_path)
    for i in range(16, 29):
        loss_module.alter.queue.append(i)
    loss, dice, ce = loss_module(preds, targets, alter_targets, alter_target_paths, epoch)
    print("Loss computation successful.")
    print(f"Total Loss: {loss.item()}, Dice Loss: {dice.item()}, Cross Entropy Loss: {ce.item()}")
    
     
    slope = loss_module.alter.Alter(5, 50)
    print(f'Slope : {slope}; Queue : {loss_module.alter.queue}')
        
    from Alter_dice import DiceScore
    assert targets.shape[1]==1, f'Targets have changed shapes : {targets.shape}'
    print(targets.shape)
   
    dice_score = DiceScore(preds, alter_targets.long())
    print(f'Dice Score : {dice_score}')
    
    assert round(dice_score.item() + dice.item(), 6) == 1.0, f'There seems to be some error in the calculation. Dice Score : {dice_score}'