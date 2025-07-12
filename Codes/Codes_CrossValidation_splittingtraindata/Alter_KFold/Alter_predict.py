import torch
import os
import cv2
import sys
from torch.utils.data import DataLoader
from Alter_data_loader import Data_Loader_Test
from color_scheme import ColorMask

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from unet_model import UNet

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


if __name__ == '__main__':
    test_images_dir = '/home/ayush/segmentation/Dataset065_Cracks/imagesTs'
    output_dir = '/home/ayush/segmentation/Results_5cv_unet/Alter/Vanilla_1000/Cracks/fold_4/test_output'
    test_dataset = Data_Loader_Test(test_images_dir)
    num_classes = 2

    model = UNet(in_channels=3, out_channels=num_classes, init_features=32).to('cuda')
    model_path = '/home/ayush/segmentation/Results_5cv_unet/Alter/Vanilla_1000/Cracks/fold_4/latest_checkpoint_path.pth'
    # model_path = os.path.join(output_dir.strip('/test_output'), "latest_checkpoint_path.pth")
    model.load_state_dict(torch.load(model_path, map_location='cuda'))

    predict(test_dataset, output_dir, model)