import numpy as np
import os
from PIL import Image

# Assign a unique color per count value
def label_to_color(mask):
    unique_vals = np.unique(mask)
    np.random.seed(42)
    color_map = {
        val: tuple(np.random.randint(0, 256, size=3)) for val in unique_vals if val != 0
    }
    color_map[0] = (0, 0, 0)  # black for background/no overlap

    h, w = mask.shape
    color_img = np.zeros((h, w, 3), dtype=np.uint8)

    for val, color in color_map.items():
        color_img[mask == val] = color

    return color_img

def combine_masks(path_to_masks_dir, path_to_visualizations):
    # Load base mask with "_001"
    base_file = [file for file in os.listdir(path_to_masks_dir) if "_001" in file][0]
    image = Image.open(os.path.join(path_to_masks_dir, base_file)).convert('L')
    combined = (np.array(image) > 0).astype(np.uint16)  # binarize

    # Loop over visualizations and binarize + add
    for file in os.listdir(path_to_visualizations):
        img = Image.open(os.path.join(path_to_visualizations, file)).convert("L")
        img_np = (np.array(img) > 0).astype(np.uint16)
        combined += img_np  # just count overlaps
    
    # Convert to image and save
    colored = label_to_color(combined)
    Image.fromarray(colored).save(os.path.join(path_to_visualizations, "final_colored_overlay.png"))

# path_to_masks_dir = "/home/devansh/Desktop/SkelRec/EWLenv/Test_UNet/Dataset015_DRIVE/labelsTr"
# path_to_log_dir = "/home/devansh/Desktop/SkelRec/EWLenv/Test_UNet/Results/Alter"
# path_to_visualizations = os.path.join(path_to_log_dir,'visulaizations')

# combine_masks(path_to_masks_dir, path_to_visualizations)