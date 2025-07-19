from tabulate import tabulate
import os
import numpy as np
from scipy.spatial.distance import directed_hausdorff
import cv2
from skimage.morphology import skeletonize
from scipy.ndimage import binary_erosion
from scipy.spatial.distance import cdist
import nibabel as nib
import matplotlib.pyplot as plt
from PIL import Image
from concurrent.futures import ProcessPoolExecutor


# Parameters
num_classes = 2  # Number of classes including background
is_0_background = True
gt_folder = "/home/nitin1/segmentation/Dataset065_Cracks/labelsTs"
results_folder = "/home/nitin1/segmentation/Results_segnet/Vanilla_1000/Cracks"
seeds = [1337, 1234, 999, 2024, 2025]
image_extension = ".png"

def calculate_metrics(gt, pred, num_classes):

    metric = []
    
    for cls in range(num_classes):

        this_class = []

        # print(f"GT unique: {np.unique(gt)}, Pred unique: {np.unique(pred)}")
        gt_bin = (gt == cls).astype(np.uint8)
        pred_bin = (pred == cls).astype(np.uint8)
        
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
            asd=0
        else:
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


        metric.append(this_class)

    if is_0_background:
        metric = metric[1:]

    mean = np.mean(metric, axis=0)
    metric.append(mean)

    return metric

def process_file(args):
    gt_path, pred_path, num_classes = args
    if(image_extension=='.nii' or image_extension=='.nii.gz'):
        gt = nib.load(gt_path).get_fdata()
        pred = nib.load(pred_path).get_fdata()
    else:        
        gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)

    if gt.shape != pred.shape:
        gt = cv2.resize(gt, (pred.shape[1], pred.shape[0]), interpolation=cv2.INTER_NEAREST)

    output = calculate_metrics(gt, pred, num_classes)
    
    print(f"done with {pred_path}")

    return output

def evaluate_segmentation(gt_folder, pred_folder, metric_output_folder, image_extension, num_classes):
    gt_files = sorted([f for f in os.listdir(gt_folder) if f.endswith(image_extension)])
    pred_files = sorted([f for f in os.listdir(pred_folder) if f.endswith(image_extension) and not f.endswith('_rgb'+ image_extension)])

    assert len(gt_files) == len(pred_files), "Number of ground truth and predicted files must match."

    args = [(os.path.join(gt_folder, gt), os.path.join(pred_folder, pr), num_classes) for gt, pr in zip(gt_files, pred_files)]

    with ProcessPoolExecutor(max_workers=8) as executor:
        metrics = list(executor.map(process_file, args))

    metrics = np.array(metrics)

    mean_metrics = np.nanmean(metrics, axis=0)

    if is_0_background:
        names = [f"Class {i}" for i in range(1,num_classes)]
    else:
        names = [f"Class {i}" for i in range(num_classes)]
    names.append("Mean Values")

    if is_0_background:
        combined_data = [[names[i]] + mean_metrics[i].tolist() for i in range(num_classes)]
    else:
        combined_data = [[names[i]] + mean_metrics[i].tolist() for i in range(num_classes+1)]

    headers = ["Class Name", "Dice scores", "clDice scores", "IoU scores", "HD95 scores", "ASD scores", "False Negative Rate", "False Positive Rate"]

    table = tabulate(combined_data, headers, tablefmt="grid")
        
    output_file = os.path.join(metric_output_folder, "metrics.txt")

    with open(output_file, "w") as file:
        file.write(table)

    return mean_metrics


if __name__=="__main__":

    results = []

    for fold in seeds:
        metric_output_folder = os.path.join(results_folder, f"seed_{fold}")

        pred_folder = os.path.join(metric_output_folder, "test_output")
        
        results.append(evaluate_segmentation(gt_folder, pred_folder, metric_output_folder, image_extension, num_classes))

    all_seeds_results = np.stack(results, axis=0)

    mean_metrics = np.mean(all_seeds_results, axis=0)
    std_metrics = np.std(all_seeds_results, axis=0)

    def format_metric(mean, std):
        return f"{mean:.4f} ± {std:.4f}"

    if is_0_background:
        names = [f"Class {i}" for i in range(1, num_classes)]
    else:
        names = [f"Class {i}" for i in range(num_classes)]
    names.append("Mean Values")

    combined_data = []
    for i in range(len(names)):
        row = [names[i]] + [format_metric(mean_metrics[i, j], std_metrics[i, j]) for j in range(7)]
        combined_data.append(row)

    headers = ["Class Name", "Dice scores", "clDice scores", "IoU scores", "HD95 scores", "ASD scores", "False Negative Rate", "False Positive Rate"]

    table = tabulate(combined_data, headers, tablefmt="grid")
    output_file = os.path.join(results_folder, "metrics.txt")

    with open(output_file, "w") as file:
        file.write(table)