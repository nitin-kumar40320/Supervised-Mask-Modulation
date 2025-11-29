import numpy as np
import torch
from skimage.morphology import dilation, ball, closing
import os

def transform(pred_mask, gt_mask, num_classes, focus_classes):

    temp_onehot = np.eye(num_classes)[gt_mask]
    dims = list(range(temp_onehot.ndim))
    dims = [temp_onehot.ndim - 1] + dims[:-1]
    gt_mask = np.transpose(temp_onehot, axes=dims).astype(np.float32)

    temp_onehot = np.eye(num_classes)[pred_mask]
    dims = list(range(temp_onehot.ndim))
    dims = [temp_onehot.ndim - 1] + dims[:-1]
    pred_mask = np.transpose(temp_onehot, axes=dims).astype(np.float32)
    del temp_onehot
    
    missed_pixel_mask = gt_mask - pred_mask
    missed_pixel_mask = (missed_pixel_mask > 0).astype(np.int16)

    modified_mask = dilation(missed_pixel_mask)
    modified_mask = (gt_mask.astype(bool) | modified_mask.astype(bool)).astype(float)

    for i in range(num_classes):
        if i not in focus_classes:
            modified_mask[i, :, :] = gt_mask[i, :, :]
    
    return modified_mask

# if __name__ == "__main__":
#     gt_image = cv2.imread("/home/devansh/Desktop/SkelRec/EWLenv/Test_UNet/Dataset015_DRIVE/labelsTs/retina_001.png", 0)
#     gt_image = (gt_image > 0).astype(np.int16)
#     gt_image = gt_image * 255
#     pred_image = cv2.imread("/home/devansh/Desktop/SkelRec/EWLenv/Test_UNet/Results/SRL_1000/test_output/retina_001_0000.png", 0)

#     print(pred_image.shape, gt_image.shape)
#     print(np.unique(pred_image), np.unique(gt_image))

#     modified_image = transform(pred_image, gt_image)*255

#     print("Modified Image Shape:", modified_image.shape)
#     print(np.unique(modified_image))

#     cv2.imwrite("/home/devansh/Desktop/SkelRec/EWLenv/Test_UNet/GuidedMaskModulation/sample_transform_withoutclosed.png", modified_image)
#     cv2.imwrite("/home/devansh/Desktop/SkelRec/EWLenv/Test_UNet/GuidedMaskModulation/sample_gt.png", gt_image)
#     cv2.imwrite("/home/devansh/Desktop/SkelRec/EWLenv/Test_UNet/GuidedMaskModulation/sample_pred.png", pred_image)