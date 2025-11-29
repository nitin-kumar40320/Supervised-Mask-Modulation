import torch.nn as nn
import torch
import numpy as np
from dice import DiceLoss
from reduced_senstivity_loss import ReducedSenstivityLoss
import cv2


class combined_loss_function(nn.Module):
    def __init__(self, num_class, epsilon=1e-6):
        super(combined_loss_function, self).__init__()
        self.dice = DiceLoss(epsilon=epsilon, num_classes=num_class)
        self.CE = nn.CrossEntropyLoss()
        self.rsl = ReducedSenstivityLoss(num_classes=num_class, epsilon=epsilon)
    
    def forward(self, preds, targets, transformed_targets, epoch, pretrain_epochs):
        # preds: [N, C, H, W], raw logits
        # targets: [N, H, W], class indices (LongTensor)
        assert preds.dim() == 4, "Predictions should be of shape [N, C, H, W]"
        assert targets.dim() == 3, "Targets should be of shape [N, H, W]"

        ce_loss = self.CE(preds, targets)
        dice_loss = self.dice(preds, targets)
        rsl_loss = self.rsl(preds, transformed_targets)
        
        if epoch < pretrain_epochs:
            rsl_loss= rsl_loss * 0
        # else:
        #     rsl_loss = rsl_loss * 0.25
    
        return (dice_loss + ce_loss + rsl_loss, dice_loss, ce_loss, rsl_loss)