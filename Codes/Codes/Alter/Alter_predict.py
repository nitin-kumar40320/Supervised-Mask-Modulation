import torch
import os
import cv2
from torch.utils.data import DataLoader
from color_scheme import ColorMask

# model_path = "/home/devansh/Desktop/SkelRec/EWLenv/Test_UNet/Results/Slope_method/Bombr/500/best_checkpoint_path.pth"
# test_images_dir = "/home/devansh/Desktop/SkelRec/EWLenv/Test_UNet/Dataset005_Bombr/imagesTs"
# output_mask_dir = "/home/devansh/Desktop/SkelRec/EWLenv/Test_UNet/Results/Slope_method/Bombr/500/"
# num_classes = 4
# os.makedirs(output_mask_dir, exist_ok=True)

# model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet', in_channels=3, out_channels=num_classes, init_features=32, pretrained=False)

# print("Starting prediction...")
def predict(dataset, output_mask_dir, model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device) 
    color_mapper = ColorMask()
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    for images, img_names in dataloader :
        images = images.to(device)
        with torch.no_grad():
            outputs = model(images)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            predicted_masks = torch.argmax(probabilities, dim=1)
        
        for i in range(predicted_masks.size(0)):
            mask_path_rgb = os.path.join(output_mask_dir, img_names[i][:-4] + "_rgb" + img_names[i][-4:])
            mask_path = os.path.join(output_mask_dir, img_names[i])
            
            colored_image = color_mapper(predicted_masks[i])
            colored_image.save(mask_path_rgb)
            cv2.imwrite(mask_path, predicted_masks[i].cpu().numpy())
        print(f'Done {img_names}')