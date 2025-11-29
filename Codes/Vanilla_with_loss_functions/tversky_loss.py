import torch
import torch.nn as nn
import torch.nn.functional as F
from dice import DiceLoss

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F

class TverskyLoss(nn.Module):
    def __init__(self, num_classes, alpha=0.5, beta=0.5, eps=1e-7):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.eps = eps
        self.num_classes = num_classes 

    def forward(self, preds, targets):
        # preds: [N, C, H, W] raw logits
        # targets: [N, H, W] integer labels

        probs = F.softmax(preds, dim=1)

        targets_oh = F.one_hot(targets, self.num_classes).permute(0, 3, 1, 2).float()

        probs_flat = probs.reshape(-1, self.num_classes)
        targets_flat = targets_oh.reshape(-1, self.num_classes)

        TP = (probs_flat * targets_flat).sum(dim=0)
        FP = (probs_flat * (1 - targets_flat)).sum(dim=0)
        FN = ((1 - probs_flat) * targets_flat).sum(dim=0)

        tversky_index = TP / (TP + self.alpha * FP + self.beta * FN + self.eps)

        loss = 1 - tversky_index.mean()

        return loss


class TverskyCombinedLoss(nn.Module):
    def __init__(self, num_class, t_alpha=0.3, t_beta=0.7, epsilon=1e-6):
        super(TverskyCombinedLoss, self).__init__()
        self.dice = DiceLoss(epsilon=epsilon, num_classes=num_class)
        self.CE = nn.CrossEntropyLoss()
        self.tversky = TverskyLoss(num_classes=num_class, alpha=t_alpha, beta=t_beta)
    
    def forward(self, preds, targets):
        # preds: [N, C, H, W], raw logits
        # targets: [N, H, W], class indices (LongTensor)
        assert preds.dim() == 4, "Predictions should be of shape [N, C, H, W]"
        assert targets.dim() == 3, "Targets should be of shape [N, H, W]"
        
        ce_loss = self.CE(preds, targets)
        dice_loss = self.dice(preds, targets)
        tversky_loss = self.tversky(preds, targets)
        
        return (dice_loss + ce_loss + tversky_loss, dice_loss, ce_loss, tversky_loss)
