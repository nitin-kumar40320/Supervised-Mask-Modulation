import torch
import torch.nn as nn
from math import prod

class ReducedSenstivityLoss(nn.Module):
    def __init__(self, num_classes, epsilon=1e-6):
        super(ReducedSenstivityLoss, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon

    def forward(self, preds, targets):

        assert isinstance(preds, torch.Tensor), "'preds' must be a PyTorch tensor"
        assert isinstance(targets, torch.Tensor), "'targets' must be a PyTorch tensor"

        with torch.autocast('cuda',enabled = False):
            ## convert raw logits to probabilities
            preds = torch.nn.functional.softmax(preds, dim=1)

        assert preds.shape == targets.shape, f"Predictions({preds.shape}) and targets({targets.shape}) must have the same shape"

        shp_x = preds.shape

        axes = list(range(2, len(shp_x)))

        with torch.no_grad():
            img_size = prod(shp_x[2:])

        tp = (preds.float() * targets.float()).sum(axes)
        fn = ((1-preds.float()) * targets.float()).sum(axes)

        denum = img_size + fn + self.epsilon
        red_sens = (tp + self.epsilon) / (torch.clip(denum, 1e-8))

        red_sens = red_sens.mean()

        return -red_sens
