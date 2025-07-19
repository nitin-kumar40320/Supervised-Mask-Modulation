import torch
from typing import List

class SurfaceLoss(torch.nn.Module):
    def __init__(self, num_classes: int, idc: List[int]):
        super(SurfaceLoss, self).__init__()
        self.num_classes = num_classes
        self.idc = idc

    def forward(self, preds: torch.Tensor, dist_maps: torch.Tensor) -> torch.Tensor:
        assert isinstance(preds, torch.Tensor), "'preds' must be a PyTorch tensor"
        assert isinstance(dist_maps, torch.Tensor), "'dist_maps' must be a PyTorch tensor"

        with torch.autocast('cuda',enabled = False):

            # Convert raw logits to probabilities
            preds = torch.nn.functional.softmax(preds.float(), dim=1)

            # Shape check
            assert preds.shape == dist_maps.shape, \
                f"Shape mismatch: preds({preds.shape}) vs dist_maps({dist_maps.shape})"

        # Select only the specified classes
        preds = preds[:, self.idc, ...]
        dist_maps = dist_maps[:, self.idc, ...]

        # Element-wise multiplication and averaging
        surface_loss = (preds.float() * dist_maps.float()).mean()

        return surface_loss
