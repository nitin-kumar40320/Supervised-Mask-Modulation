import torch
import torch.nn as nn
import torch.nn.functional as F
from dice import DiceLoss

class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0, eps=1e-7):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps

    def forward(self, preds, targets):
        # preds: [N, C, H, W] raw logits
        # targets: [N, H, W] class indices
        
        # Convert logits to probabilities
        probs = F.softmax(preds, dim=1)  # [N, C, H, W]

        # Gather probabilities of the true class for each pixel
        pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)  # [N, H, W]

        pt = pt.clamp(min=self.eps, max=1.0 - self.eps)

        # Focal loss computation
        focal_term = (1 - pt) ** self.gamma

        loss = -self.alpha * focal_term * pt.log()

        return loss.mean()



class FocalCombinedLoss(nn.Module):
    def __init__(self, num_class, f_alpha=1.0, f_gamma=2.0, epsilon=1e-6):
        super(FocalCombinedLoss, self).__init__()
        self.dice = DiceLoss(epsilon=epsilon, num_classes=num_class)
        self.CE = nn.CrossEntropyLoss()
        self.focal = FocalLoss(alpha=f_alpha, gamma=f_gamma)
    
    def forward(self, preds, targets):
        # preds: [N, C, H, W], raw logits
        # targets: [N, H, W], class indices (LongTensor)
        assert preds.dim() == 4, "Predictions should be of shape [N, C, H, W]"
        assert targets.dim() == 3, "Targets should be of shape [N, H, W]"
        
        ce_loss = self.CE(preds, targets)
        dice_loss = self.dice(preds, targets)
        focal_loss = self.focal(preds, targets)
        
        return (dice_loss + ce_loss + focal_loss, dice_loss, ce_loss, focal_loss)
