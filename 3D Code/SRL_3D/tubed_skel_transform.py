import numpy as np
from skimage.morphology import skeletonize, dilation
import cv2
    
def tubed_skeleton_transform(img_array):

    bin_seg = (img_array > 0)
    
    # Skeletonize
    if not np.sum(bin_seg) == 0:
        skel = skeletonize(bin_seg)
        skel = (skel > 0).astype(np.int16)
        skel = dilation(dilation(skel))
        skel *= img_array.astype(np.int16)

    return skel


if __name__=="__main__":

    img_path = "/home/nitin1/segmentation/Dataset005_Bombr/labelsTr/bombr_001.png"
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    skel = tubed_skeleton_transform(img)
    cv2.imwrite("bombr_tubed_skel.png", skel * 255)