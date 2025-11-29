import torch
import torch.nn as nn
import torch.nn.functional as F
from dice import DiceLoss
from BoundaryLoss import SurfaceLoss

class CombinedLoss(nn.Module):
    def __init__(self, num_class, focus_classes, epsilon=1e-6):
        super(CombinedLoss, self).__init__()
        self.dice = DiceLoss(epsilon=epsilon, num_classes=num_class)
        self.CE = nn.CrossEntropyLoss()
        self.boundary_loss = SurfaceLoss(num_classes=num_class, idc=focus_classes)
    
    def forward(self, preds, targets, dist_maps):
        # preds: [N, C, H, W], raw logits
        # targets: [N, H, W], class indices (LongTensor)
        assert preds.dim() == 4, "Predictions should be of shape [N, C, H, W]"
        assert targets.dim() == 3, "Targets should be of shape [N, H, W]"
        
        ce_loss = self.CE(preds, targets)
        dice_loss = self.dice(preds, targets)
        boundary_loss = self.boundary_loss(preds, dist_maps)
        
        return (dice_loss + ce_loss + boundary_loss, dice_loss, ce_loss, boundary_loss)
