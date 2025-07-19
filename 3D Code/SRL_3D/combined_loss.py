import torch.nn as nn
import numpy as np
from dice import DiceLoss
from skelrec import SkeletonRecLoss
import torch

class combined_loss_function(nn.Module):
    def __init__(self, num_of_class, epsilon=1e-6):
        super(combined_loss_function, self).__init__()
        self.dice = DiceLoss(epsilon=epsilon, num_classes=num_of_class)
        self.CE = nn.CrossEntropyLoss()
        self.skelrec = SkeletonRecLoss(epsilon=epsilon, num_classes=num_of_class)
    
    def forward(self, preds, targets, transformed_targets):
        ce_loss = self.CE(preds, targets)
        with torch.autocast('cuda', enabled = False):
            preds = torch.nn.functional.softmax(preds.float(), dim=1)
        dice_loss = self.dice(preds, targets)
        skelrec_loss = self.skelrec(preds, transformed_targets)
    
        return (dice_loss + ce_loss + skelrec_loss, dice_loss, ce_loss, skelrec_loss)