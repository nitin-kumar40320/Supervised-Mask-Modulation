import torch
import torch.nn.functional as F

def transform(pred_mask, gt_mask, num_classes, focus_classes):
    """
    pred_mask, gt_mask: [B, H, W, D] — class indices
    Returns: modulated one-hot mask: [B, C, H, W, D]
    """

    with torch.no_grad():
        # One-hot encode: F.one_hot gives shape [B, H, W, D, C]
        gt_onehot = F.one_hot(gt_mask.long(), num_classes=num_classes)  # [B, H, W, D, C]
        pred_onehot = F.one_hot(pred_mask.long(), num_classes=num_classes)

        # Move channel to 2nd dim: [B, C, H, W, D]
        gt_onehot = gt_onehot.permute(0, 4, 1, 2, 3).float()
        pred_onehot = pred_onehot.permute(0, 4, 1, 2, 3).float()

        # Missed pixel mask
        missed_pixel_mask = (gt_onehot > pred_onehot).float()  # [B, C, H, W, D]

        # Use 3D convolution to do dilation (morphological op) in 3D
        kernel = torch.ones((num_classes, 1, 3, 3, 3), device=missed_pixel_mask.device)
        padding = 1

        # Grouped convolution: apply kernel to each class channel
        dilated = F.conv3d(
            missed_pixel_mask, kernel, padding=padding, groups=num_classes
        )
        dilated = (dilated > 0).float()

        # Combine with gt
        final_mask = (gt_onehot.bool() | dilated.bool()).float()

        # Zero-out unfocused classes
        if focus_classes is not None:
            focus_mask = torch.zeros(num_classes, dtype=torch.bool, device=final_mask.device)
            focus_mask[focus_classes] = True
            class_mask = focus_mask.view(1, -1, 1, 1, 1)
            final_mask = torch.where(class_mask, final_mask, gt_onehot)

        return final_mask.to(torch.uint8)
