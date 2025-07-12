import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self, epsilon=1e-6):
        super(DiceLoss, self).__init__()
        self.epsilon = epsilon
    
    def forward(self, preds, targets):
        '''
        Inputs : preds, gt_mask - Tensors of shape [B, C, H, W]
        '''
        assert targets.ndim == 4, f"Expected 4D tensor for targets (B, C, H, W) in DiceLoss, but got shape {targets.shape}"
        # assert targets.shape[1] >= 1, f"Expected channel dimension (C) >= 1 in targets, but got shape {targets.shape}"
        # print(f'targets: {targets}')
        # print(f'prediction : {preds}')
        # targets = F.one_hot(targets, num_classes=preds.shape[1]).permute(0, 3, 1, 2).float()
        # print(f'targets shape : {targets.shape} - in DiceLoss under alter_dice.py')
        # print(f'pred shapes : {preds.shape} - DiceLoss')
        dims = (2, 3)  # sum over batch, height, width
        intersection = torch.sum(preds * targets, dims)
        union = preds.sum(dim=dims) + targets.sum(dim=dims)
        dice = (2. * intersection + self.epsilon) / (union + self.epsilon)
        mean_dice = dice.mean(dim=1).mean(dim=0)
        return 1 - mean_dice


def DiceScore(preds, targets, epsilon=1e-6):
    """
    Computes Dice Score for multi-class segmentation.

    preds: Tensor of shape [B, C, H, W] (logits)
    targets: Tensor of shape [B, 1, H, W] or [B, H, W]
    """
    # print('Dice Score Calc')
    # Apply softmax to get class probabilities
    preds = torch.softmax(preds, dim=1)  # [B, C, H, W]
    # print(f'Predictions shape : {preds.shape} -- DiceScore')
    # print(f'Targets Shape : {targets.shape} -- DiceScore')
    if targets.shape[1] == 1:
        targets = targets.squeeze(1)
    targets = F.one_hot(targets, num_classes=preds.shape[1]).permute(0, 3, 1, 2).float()
    # print(f'One-Hot Targets Shape : {targets.shape} -- DiceScore')
    # Compute intersection and union
    dims = (2, 3)  # Sum over height, width
    intersection = torch.sum(preds * targets, dims)
    union = preds.sum(dim=dims) + targets.sum(dim=dims)
    dice = (2. * intersection + epsilon) / (union + epsilon)
    return dice.mean(dim=1).mean(dim=0)

