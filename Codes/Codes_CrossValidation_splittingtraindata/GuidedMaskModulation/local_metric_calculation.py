import numpy as np
from scipy.spatial.distance import directed_hausdorff
from skimage.morphology import skeletonize
from scipy.ndimage import binary_erosion
from scipy.spatial.distance import cdist

def calculate_metrics(gt, pred):
    
    cls=1

    this_class = []
    
    gt_bin = (gt == cls).astype(np.uint8)
    pred_bin = (pred == 255).astype(np.uint8)
    
    # Dice Score
    intersection = np.sum(gt_bin * pred_bin)
    dice = (2 * intersection) / (np.sum(gt_bin) + np.sum(pred_bin) + 1e-6)
    this_class.append(dice)

    # clDice
    gt_skel = skeletonize(gt_bin > 0)
    pred_skel = skeletonize(pred_bin > 0)
    prec = np.sum(pred_skel * gt_bin) / (np.sum(pred_skel) + 1e-6)
    sens = np.sum(gt_skel * pred_bin) / (np.sum(gt_skel) + 1e-6)
    cldice = 2 * prec * sens / (prec + sens + 1e-6)
    this_class.append(cldice)
    
    # IoU
    union = np.sum(gt_bin) + np.sum(pred_bin) - intersection
    iou = intersection / (union + 1e-6)
    this_class.append(iou)

    # HD95 (Hausdorff Distance at 95th percentile)
    if np.any(gt_bin) and np.any(pred_bin):
        gt_points = np.argwhere(gt_bin)
        pred_points = np.argwhere(pred_bin)
        hd95 = max(
            np.percentile([directed_hausdorff(gt_points, pred_points)[0] for _ in range(2)], 95),
            np.percentile([directed_hausdorff(pred_points, gt_points)[0] for _ in range(2)], 95)
        )
    else:
        hd95 = np.nan
    this_class.append(hd95)
    
    # ASD (Average Surface Distance)
    pred_boundary = pred_bin ^ binary_erosion(pred_bin)
    gt_boundary = gt_bin ^ binary_erosion(gt_bin)
    
    pred_coords = np.argwhere(pred_boundary)
    gt_coords = np.argwhere(gt_boundary)

    if len(pred_coords) == 0 or len(gt_coords) == 0:
        return float('inf')

    dist_pred_to_gt = cdist(pred_coords, gt_coords)
    dist_gt_to_pred = cdist(gt_coords, pred_coords)

    asd = (dist_pred_to_gt.min(axis=1).sum() + dist_gt_to_pred.min(axis=1).sum()) / (len(pred_coords) + len(gt_coords))
    this_class.append(asd)

    # False Negative Rate(FNR)
    TP = np.sum(gt_bin * pred_bin)
    FN = np.sum(gt_bin * (1-pred_bin))

    fnr = FN / (FN + TP + 1e-6)
    this_class.append(fnr)

    # False Positive Rate(FPR)
    FP = np.sum((1-gt_bin) * pred_bin)
    TN = np.sum((1-gt_bin) * (1-pred_bin))

    fpr = FP / (FP + TN + 1e-6)
    this_class.append(fpr)

    return this_class
