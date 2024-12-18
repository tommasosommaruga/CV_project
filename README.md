# Semantic Segmentation of Roads in the Cityscapes Dataset

This project involves performing semantic segmentation of the road class from the Cityscapes dataset using four different approaches. Each approach is implemented in a separate folder within the main project directory. The project also allows for adding personal images and videos for experimentation. Below are details on the approaches and dataset setup.

## Approaches

### 1. Single Pixel Classifier
This method treats each pixel independently. A binary classification model is trained to predict whether a given pixel belongs to the road class or not, without considering its spatial context. This simplistic approach serves as a baseline.

### 2. Patch Classification
In this approach, a binary classification model predicts the class of a central pixel in a small patch of the image. The surrounding pixels provide spatial context, enabling better predictions compared to the single pixel classifier.

### 3. Fully Convolutional Neural Networks (FCNNs)
FCNNs are deep learning models designed for dense predictions. They take an entire image as input and output a map where each pixel is classified. This approach leverages the full spatial context of the image and is computationally efficient for large-scale datasets.

### 4. U-NETs
U-NETs are a specialized type of convolutional neural network architecture for semantic segmentation. They utilize an encoder-decoder structure with skip connections, which helps preserve spatial information while capturing high-level features. U-NETs are particularly effective for pixel-wise segmentation tasks like this one.

## Dataset Setup
To set up the dataset, create a folder named `data` in the main project directory. This folder should contain subdirectories for the various components of the Cityscapes dataset and additional personal data:

```
main_project_folder/
    dumb_pixel_classifier/
    patch_classifier/
    fully_conv_nn/
    unet_classifier/
    data/
        disparity_trainvaltest/
        gtCoarse/
        gtFine_trainvaltest/
        leftimg8bit_trainvaltest/
        personal_images/
        personal_videos/
```

### Subdirectory Descriptions
1. **`disparity_trainvaltest/`**: Contains disparity maps for the Cityscapes dataset, used for depth information.
2. **`gtCoarse/`**: Contains coarse ground truth annotations.
3. **`gtFine_trainvaltest/`**: Contains fine-grained ground truth annotations, essential for training and evaluation.
4. **`leftimg8bit_trainvaltest/`**: Contains the left RGB images of the dataset, used as the primary input for the models.
5. **`personal_images/`**: Add your own images here for experimentation. Ensure they have a similar resolution and format as the Cityscapes dataset for consistency.
6. **`personal_videos/`**: Add personal video data here. These videos can be preprocessed into individual frames for testing the models.

## Adding Personal Data
- **Images**: Place additional images in the `personal_images/` folder. Ensure they are in a format compatible with the existing dataset (e.g., `.png` or `.jpg`) and preprocess them to match the input requirements of the models.
- **Videos**: Place videos in the `personal_videos/` folder. Use a video-to-frame extraction tool to generate individual frames if needed.

## Notes
- To obtain the cityscapes dataset it is necessary to have an account and login.


