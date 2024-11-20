# Pixel Classifier for Road Detection in Cityscapes Dataset

## Overview
This project implements a pixel-wise classifier that identifies whether each pixel in an image belongs to the "road" category. The model takes RGB pixel values as input and classifies each pixel as either `1` (road) or `0` (non-road). The model is trained using the Cityscapes dataset, which provides RGB images and corresponding pixel-level annotations. 

The primary goal of the code is to create a pipeline that:
1. Loads RGB images and their associated label images.
2. Samples pixels from the "road" class.
3. Trains a neural network model to classify these pixels.
4. Tests the model and validates performance on a subset of the data.

This project utilizes TensorFlow and Keras for model building and training, along with OpenCV for image processing.

---

## Requirements
- Python 3.x
- TensorFlow (for building and training the neural network)
- OpenCV (for image loading and processing)
- NumPy (for numerical operations)
- Matplotlib (for plotting images and visualizing results)
- Cityscapes dataset

---

## Cityscapes Dataset
The Cityscapes dataset provides high-resolution images of urban street scenes, annotated with pixel-level labels for various objects like cars, people, and roads. The dataset has several components:

1. **Images**: High-resolution RGB images stored in folders organized by city.
2. **Labels**: Label images where each pixel is associated with an integer representing a specific class (e.g., road, building, vehicle).

For this project, the main task is to focus on the "road" class, where the `TrainId` label for road pixels is `7` (as defined in the `labels.py` file).

---

## How the Code Works

### 1. **Loading the Cityscapes Images and Labels**

The code first loads the Cityscapes dataset, including the RGB images and corresponding label images. These images are organized in directories by city and labeled according to the Cityscapes annotation scheme.

```python
# Paths to the dataset directories
CITYSCAPES_DIR = "data"
IMAGES_DIR = os.path.join(CITYSCAPES_DIR, "leftImg8bit_trainvaltest/leftImg8bit/train")
LABELS_DIR = os.path.join(CITYSCAPES_DIR, "gtFine_trainvaltest/gtFine/train")
```

### 2. **Pixel Sampling**

To train the classifier, we need to extract pixel values from the "road" class. The program can either sample all road pixels or a random subset of pixels (to avoid large memory usage). The code then samples these pixels from a set of images.

Here is how the code works to select the pixels:

- **For each image**: The code loads the image and the corresponding label.
- **Random Sampling**: It randomly selects a fixed number of pixels from the "road" pixels for training.

```python
road_mask = (label_pixels == 7)
road_pixels = image_pixels[road_mask]
```

If you want to limit the number of pixels sampled for efficiency, the code allows you to specify a maximum number of pixels.

```python
sampled_road_pixel = road_pixels[np.random.choice(road_pixels.shape[0], 1)]
```

### 3. **Model Building**

The neural network (NN) model used for pixel classification is built using TensorFlow/Keras. The model takes RGB values as input and outputs a binary prediction: `1` for road pixels, and `0` for non-road pixels.

Here’s a basic outline of the model architecture:
- **Input Layer**: The model takes RGB values for each pixel.
- **Hidden Layers**: Several dense layers are used to process the pixel input.
- **Output Layer**: The model predicts whether the pixel is a road pixel or not.

### 4. **Training the Model**

The model is trained using the selected pixels. The following steps are involved:
- **Input**: Randomly sampled pixels from the dataset (RGB values).
- **Label**: The corresponding road or non-road label.
- **Training Process**: The neural network is trained to minimize the binary cross-entropy loss using the Adam optimizer.

```python
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=10, batch_size=32)
```

### 5. **Visualizing the Results**

Before normalizing the pixel values and passing them through the model, we can visualize the RGB values and the specific "road" pixel selected from the image.

```python
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.scatter(col, row, color='red', s=100, marker='x')  # Mark the road pixel
plt.title('Highlighted Road Pixel')
plt.show()
```

This allows us to confirm that the sampled pixels correspond to the road area, ensuring that the training data is correctly selected.

### 6. **Evaluation**

After training the model, the performance is evaluated on a set of test images. The model’s accuracy is reported, showing how well it can predict whether a given pixel belongs to the road class.

---

## Example Usage

### Step 1: Prepare the Dataset
Ensure that you have the Cityscapes dataset downloaded and available on your machine. Set the correct paths for your images and label files.

### Step 2: Sampling Pixels
Specify the number of images you want to use for training and how many pixels you want to sample from each image.

```python
pixels_per_image = 1000
```

### Step 3: Train the Model
Train the pixel classifier using the selected pixels.

```python
model.fit(x_train, y_train, epochs=10, batch_size=32)
```

### Step 4: Visualize the Pixels
Visualize the randomly selected road pixels.

```python
plot_random_road_pixel(image_path, label_path)
```

---

## Limitations and Further Work
- **Memory Usage**: The number of pixels in the dataset can be large, so limiting the number of sampled pixels helps control memory usage.
- **Model Complexity**: The current model is simple. A more complex model, such as a convolutional neural network (CNN), could improve accuracy.
- **Data Augmentation**: To improve model generalization, techniques like data augmentation (e.g., rotating, flipping) could be applied.

---

## Conclusion

This file provides a pipeline for training a pixel-wise classifier using the Cityscapes dataset. It allows random pixel sampling from the road class and trains a simple neural network to identify road pixels. While the current implementation is basic, it serves as a solid foundation for more advanced image segmentation tasks.