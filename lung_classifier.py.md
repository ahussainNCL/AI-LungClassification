# Pneumonia Detection App: Internal Code Documentation

[Linked Table of Contents](#linked-table-of-contents)

## <a name="linked-table-of-contents"></a>Linked Table of Contents

* [1. Introduction](#1-introduction)
* [2. Modules and Libraries](#2-modules-and-libraries)
* [3. User Interface (UI) Elements](#3-user-interface-ui-elements)
* [4. Model Loading and Selection](#4-model-loading-and-selection)
* [5. Image Preprocessing](#5-image-preprocessing)
* [6. Prediction Logic](#6-prediction-logic)
* [7. Output Presentation](#7-output-presentation)


## 1. Introduction

This document provides internal code documentation for the Pneumonia Detection application built using Streamlit and TensorFlow/Keras.  The application allows users to upload chest X-ray images (JPEG format) and receive predictions regarding the presence of pneumonia from one or more pre-trained convolutional neural network (CNN) models.


## 2. Modules and Libraries

The application utilizes the following Python modules and libraries:

| Module/Library     | Purpose                                                              |
|----------------------|--------------------------------------------------------------------------|
| `streamlit`         | Creates the web application interface.                                   |
| `tensorflow.keras` | Loads and utilizes pre-trained CNN models for image classification.     |
| `numpy`             | Provides numerical computing capabilities for image manipulation.        |


## 3. User Interface (UI) Elements

The Streamlit framework structures the application's UI with the following components:

* **Title:** Displays "Pneumonia Detection from Chest X-Ray Images" at the top.
* **Sidebar:** Contains:
    * A dropdown menu (`st.sidebar.selectbox`) to select a model ("ResNet50", "VGG16", "DenseNet", "InceptionV3", or "All Models").
    * A file uploader (`st.sidebar.file_uploader`) for users to select JPEG images.
    * A button (`st.sidebar.button`) labeled "Test" to trigger the prediction process.
* **Main Screen:** Displays the uploaded image and prediction results.


## 4. Model Loading and Selection

The application loads pre-trained Keras models based on the user's selection.  Model paths are defined in a dictionary:

| Model Name      | Path                                       |
|-----------------|--------------------------------------------|
| ResNet50        | `./models/lung_classifier_rn.keras`        |
| VGG16           | `./models/lung_classifier_vgg.keras`       |
| DenseNet         | `./models/lung_classifier_dn.keras`        |
| InceptionV3     | `./models/lung_classifier_inceptionv3.keras` |

The `load_model` function from TensorFlow/Keras loads the specified model. If "All Models" is selected, all models are loaded.


## 5. Image Preprocessing

Once an image is uploaded and the "Test" button is pressed, the image undergoes preprocessing:

1. **Target Size Determination:** The target image size is dynamically set based on the chosen model. InceptionV3 requires (299, 299) pixels, while the others use (224, 224).
2. **Image Loading:** `image.load_img` loads the image and resizes it to the determined `target_size`.
3. **Image to Array Conversion:** `image.img_to_array` converts the image into a NumPy array.
4. **Dimension Expansion:** `np.expand_dims` adds an extra dimension to the array to match the model's input shape (adding a batch dimension).
5. **Normalization:** The pixel values are normalized by dividing by 255.0, scaling them to the range 0-1.


## 6. Prediction Logic

The core prediction logic involves using the loaded model(s) to predict the probability of pneumonia.

* **Single Model Prediction:** If a single model is selected, the `predict` method is called on the selected model using the preprocessed image tensor.  The output is a single probability value representing the likelihood of pneumonia.
* **Multiple Model Predictions:** If "All Models" is selected, the prediction process iterates through each loaded model. For each model, the image is preprocessed with the appropriate `target_size` for that model. The result is a set of probabilities, one from each model.


## 7. Output Presentation

The application presents the prediction results on the main screen:

* **Uploaded Image:** The uploaded X-ray image is displayed using `st.image`.
* **Prediction Results:** The results are presented as text:
    * For a single model, it indicates whether pneumonia was detected and the associated probability.
    * For multiple models, it shows the predictions from each model individually. Probabilities are formatted to two decimal places using the `:.2%` format specifier.
