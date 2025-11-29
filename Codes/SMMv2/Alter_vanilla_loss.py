import torch
import torch.nn as nn
import torch.nn.functional as F
from Alter_dice import DiceLoss


class VanillaLoss(nn.Module):
    def __init__(self, epsilon=1e-6):
        super(VanillaLoss, self).__init__()
        self.dice = DiceLoss(epsilon=epsilon)
        self.bce = nn.BCEWithLogitsLoss(reduction='mean')

    def forward(self, preds, softm_preds, alter_targets):
        """
        preds: [N, C, H, W] — raw logits from model
        softm_preds: [N, C, H, W] — sigmoid/softmax representation (for dice)
        alter_targets: [N, C, H, W] — multi-hot target mask (float)
        """

        assert alter_targets.ndim == 4, f'Alter_targets must be [B, C, H, W], got {alter_targets.shape}'

        # --- BCE (vectorized, no for loop) ---
        bce_loss = self.bce(preds, alter_targets)

        # --- Dice Loss ---
        # if your model uses sigmoid instead of softmax for multi-label setup, replace `softm_preds` by `torch.sigmoid(preds)`
        dice_loss = self.dice(torch.sigmoid(preds), alter_targets)

        total_loss = dice_loss + bce_loss
        return total_loss, dice_loss, bce_loss
