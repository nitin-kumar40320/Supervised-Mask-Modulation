import torch
import torch.nn as nn
import numpy as np

class SkeletonRecLoss(nn.Module):
    def __init__(self, num_classes, epsilon=1e-6):
        super(SkeletonRecLoss, self).__init__()

        self.epsilon = epsilon
        self.num_classes = num_classes

    def forward(self, x, y):

        assert isinstance(x, torch.Tensor), "'preds' must be a PyTorch tensor"
        assert isinstance(y, torch.Tensor), "'targets' must be a PyTorch tensor"

        ## extracting classes of targets to convert from shape [B,H,W] to [B,C,H,W]
        temp_onehot = torch.nn.functional.one_hot(y, num_classes=self.num_classes)
        
        # Move class dimension to channel position: [..., C] → [N, C, ...]
        dims = list(range(temp_onehot.ndim))
        dims = [0, temp_onehot.ndim - 1] + dims[1:-1]
        y = temp_onehot.permute(*dims).float()
        del temp_onehot

        assert x.shape == y.shape, f"Predictions({x.shape}) and targets({y.shape}) must have the same shape"

        shp_x, shp_y = x.shape, y.shape

        # make everything shape (b, c)
        axes = list(range(2, len(shp_x)))

        with torch.no_grad():
              sum_gt = y.sum(axes)

        inter_rec = (x * y).sum(axes)

        rec = (inter_rec + self.epsilon) / (torch.clip(sum_gt+self.epsilon, 1e-8))

        rec = rec.mean()
        return -rec

