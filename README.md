# Image Classification with CNN

![Nature Scene](Colours-of-Nature-1.jpg)

This project implements a Convolutional Neural Network (CNN) model to classify images into six categories: buildings, forest, glacier, mountain, sea, and street.


## Project Overview


The model is trained on a dataset of labeled images and can predict the class of new, unseen images. The app includes scripts for data preprocessing, model training, evaluation, and prediction visualization.


## Features


- Loads and preprocesses image data
- Trains a CNN model using TensorFlow/Keras
- Evaluates model accuracy on a test dataset
- Visualizes predictions on sample images
- Supports batch prediction on new images


## Repository Structure
The repository structure includes the following files and directories:

- README.md: Markdown file with project documentation, instructions, or other relevant information.
- Report.pdf: PDF document providing a detailed report of the experiments for the project.
- seg_pred: Data used for predictions
- seg_test: Data used for testing model performance
- seg_train: Data used for model training
- app.py: Python file containing the main code for deployment.
- intel_image.h5: Pre-trained model file for model weights.
- image_class.ipynb: Jupyter notebook file for exploratory data analysis and model building.
- requirements.txt: File listing the required Python libraries for the implementation of this project.


### Dataset
This project uses the "Natural Scenes Image Classification" dataset, which includes around 25,000 images sized 150x150 pixels, categorized into six classes: buildings, forests, glaciers, mountains, seas, and streets.
The dataset is originally published on [Kaggle](https://www.kaggle.com/datasets/puneet6060/intel-image-classification) and was provided by Intel for an image classification challenge.


### Notebook Structure
The notebook is structured into the following sections:

1. Data Preprocessing: This section involves importing the necessary libraries, setting up the directory paths, and visualizing random images from the dataset.

2. Model Building: In this section, three pre-trained models (Xception, InceptionV3, and VGG16) are built and trained using the dataset. Each model is evaluated for accuracy and performance.

3. Comparing Results: The results from each model are compared by plotting the accuracy per iteration.

4. Saving the Model: The best model (Xception) is saved for future use.

5. Testing the Model: The saved model is loaded and used to predict and classify images from a test dataset. The predicted labels are compared with the actual labels to evaluate the model's performance.


### Results
The project demonstrates strong image classification performance, with the Xception model achieving slightly better validation results than InceptionV3 and VGG16. The test set accuracy reaches 94.65%.


### Deployment
The project would be deployed and accessible on the cloud. This web application will allow users to upload their own images and obtain predictions for the corresponding nature scenes.


### Conclusion
The project demonstrates the effectiveness of transfer learning techniques in building powerful image classification models. By leveraging pre-trained models such as Xception, InceptionV3, and VGG16, accurate predictions can be achieved even with limited training data. This project can serve as a starting point for more advanced image classification tasks and can be further improved by fine-tuning the pre-trained models or exploring other architectures.


### Contributing
- Contributions are welcome! Please open an issue or submit a pull request.

