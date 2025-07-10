# Supervised-Mask-Modulation
Image segmentation often forms a popular task in computer vision. There have been several attempts at developing refined data-specific techniques for different segmentation tasks. In this repository, we present the code for a refined training algorithm which has proved to be effective in enahcning all kinds of segmentation tasks. The training strategy is architecture agnostic and works on various different kinds of data. The algorithm utilizes the False Negatives predicted by the model to enhance the training over the following epochs. 
This repository provides the code for implementation of Supervised Mask Modulation on UNet Architecture. The figure below presents the flow of the algorithm. 

---
## Mask Modulation
The methodology of the model revolves around a novel mask transformation technique. The technique involves segregating out the false negatives from the predicted samples, followed by dilation. This dilated mask is added back to the ground truth to evolve the modulated mask. This modulated mask is utilized for training in further epochs. 

---
## Training Strategies
As a part of this study, we have attmpted several ways to utilize this transformation. They have been presented as the hard and soft training strategies. In each strategy, the first 20% epochs are unaltered for vanilla pre-training of the model. Each of them utilizes the mask transformation in different ways to generate suitable outputs.
- **SMM***v1* : This version of SMM involves a special penalization function based on the sensitivity between the transformed mask and the predicted output. In this strategy, the mask is transformed in each epoch after pre-training. This is the hard-trianing strategy.
- **SMM***v2* : This version of SMM involves thresholding for conditional mask transformation. The model computes the recall loss for the generated outputs per epoch and calculates the gradient of the recall values with respect to a fixed number of epochs until the current recall value. If the slope value falls below a certain threshold, then the mask transformation is triggered, if the pre-training epochs are over.

---
## Dataset Format
The dataset directory format has to be similar to the one used in nnUNet. The module expects four directories within the parent dataset directory: 
- imagesTr : Training Samples
- labelsTr : Training Labels
- imagesTs : Testing Samples
- labelsTr : Testing Labels

Ensure that the image and lable names match as per the convention. A sample has been provided as follows:

    Dataset005_Bombr/
    ├── imagesTr
    │   ├── Bombr_001_0000.png
    │   ├── Bombr_002_0000.png
    │   ├── Bombr_003_0000.png
    │   └── ...
    ├── imagesTs
    │   ├── Bombr_485_0000.png
    │   ├── Bombr_486_0000.png
    │   └── ...
    ├── labelsTr
    │   ├── Bombr_001.png
    │   ├── Bombr_002.png
    │   ├── Bombr_003.png
    │   └── ...
    └── labelsTs
        ├── Bombr_485.png
        ├── Bombr_486.png
        └── ...

The code has been programmed to run KFold Cross Vaidation on the training dataset, with a default value of 5 folds. The training dataset, is therefore split up into training and validation sets for each fold, which is recorded as a json file in the respective results directory.

---
## Code usage
The code for both strategies has been included in separate directories, named as SMMv1 and SMMv2. Follow the corresponding steps to run the experiments:

### SMM***v1***
### SMM***v2***
- First create the KFold Splits using the command : `cd SMMv2; python generate_splits.py --dataset Datasetxxx_name --folds <int>` (To run 5 Folds, you may skip the `--folds <int>` part)
- Then, you may start trainng the model : `python main.py --folds [<int> or all]`. When selected `all`, the model starts training for 5 folds. The folds can also be specified as `<int1>,<int2>,<int3>` for running only certain specific folds.
- Once the model is trained, the model predictions on test set are automatically stored in directory called 'test_outputs'.

### Evaluation Statistics

