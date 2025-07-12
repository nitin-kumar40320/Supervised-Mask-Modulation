import torch
import torch.nn as nn
import torch.nn.functional as F
from Alter_dice import DiceLoss


def classwise_bce_loss(logits, targets, ignore_index=None):
    """
    Computes binary cross-entropy loss for each class independently.

    Args:
        logits (Tensor): Raw model output of shape [N, C, H, W].
        targets (Tensor): One-hot or soft targets of shape [N, C, H, W].
        ignore_index (int, optional): If specified, pixels with this index are ignored.

    Returns:
        Tensor: Averaged binary cross-entropy loss across all classes.
    """
    C = logits.shape[1]
    total_loss = 0.0

    for cls in range(C):
        gt_mask = targets[:, cls, :, :].float()        # [N, H, W]
        pred_logits = logits[:, cls, :, :].float()             # [N, H, W]

        if ignore_index is not None:
            mask = (targets != ignore_index)
            gt_mask = gt_mask * mask[:, cls, :, :].float()

        loss = F.binary_cross_entropy_with_logits(pred_logits.float(), gt_mask.float(), reduction='mean')

        total_loss += loss

    return total_loss / C


class VanillaLoss(nn.Module):
    """
    Combines Dice loss and classwise BCE loss for segmentation.

    Args:
        epsilon (float): Small value to avoid division by zero in Dice computation.
    """
    def __init__(self, epsilon=1e-6):
        super(VanillaLoss, self).__init__()
        self.dice = DiceLoss(epsilon=epsilon)
        self.ce_loss_fn = classwise_bce_loss

    def forward(self, preds, softm_preds, alter_targets):
        """
        Compute combined loss.

        Args:
            preds (Tensor): Raw logits of shape [N, C, H, W].
            softm_preds (Tensor): Softmax probabilities of shape [N, C, H, W].
            alter_targets (Tensor): One-hot targets of shape [N, C, H, W].

        Returns:
            Tuple: (total_loss, dice_loss, ce_loss)
        """
        assert alter_targets.ndim == 4, f"alter_targets must be [B, C, H, W], got {alter_targets.shape}"

        dice_loss = self.dice(softm_preds, alter_targets)
        ce_loss = self.ce_loss_fn(preds, alter_targets)

        return dice_loss + ce_loss, dice_loss, ce_loss
