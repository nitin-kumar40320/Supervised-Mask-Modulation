import torch
import torch.nn as nn
import torch.nn.functional as F
from Alter_dice import DiceLoss

import torch
import torch.nn.functional as F

def classwise_bce_loss(logits, targets, ignore_index=None):
    """
    logits: (N, C, H, W) - raw output from model
    targets: (N, H, W) - ground truth class indices
    num_classes: int - number of classes
    ignore_index: optional index to ignore in loss

    Returns:
        average BCE loss across classes
    """
    C = logits.shape[1]
    total_loss = 0.0

    for cls in range(C):
        # Create binary ground truth mask for class `cls`
        gt_mask = targets[:, cls, :, :].float()  # (N, H, W)

        if ignore_index is not None:
            mask = (targets != ignore_index)
            gt_mask = gt_mask * mask.float()  # Optional masking

        # Logits for class `cls`
        pred_logits = logits[:, cls, :, :]  # (N, H, W)

        # Apply BCE with logits (handles sigmoid inside)
        loss = F.binary_cross_entropy_with_logits(pred_logits, gt_mask, reduction='mean')
        total_loss += loss

    return total_loss / C

class VanillaLoss(nn.Module):
    def __init__(self, epsilon=1e-6):
        super(VanillaLoss, self).__init__()
        self.dice = DiceLoss(epsilon=epsilon)
        self.CE = classwise_bce_loss
    
    def forward(self, preds, softm_preds, alter_targets):
        # preds: [N, C, H, W], raw logits
        # softm_preds: [N, C, H, W], softmax representation of output
        # targets: [N, H, W], class indices (LongTensor)
        # alter_targets: [N, C, H, W], class indices alter_mask
        
        # assert targets.ndim==3, f'Targets must be of the shape [B, H, W], but recieved {targets.shape}'
        assert alter_targets.ndim==4, f'Alter_targets must be of the shape [B, C, H, W], but recieved {alter_targets.shape}'
        # targets = torch.argmax(targets, dim=1)
        dice_loss = self.dice(softm_preds, alter_targets)
        ce_loss = self.CE(preds, alter_targets)
        
        return (dice_loss + ce_loss, dice_loss, ce_loss)
