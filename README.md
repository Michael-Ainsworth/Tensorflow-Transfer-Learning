# Tensorflow-Transfer-Learning

---------------------

### Project Description
This project uses the Cifar10 dataset, a small photo dataset containing 60,000 labeled color images separated into 10 classes. Each photo is size 32 x 32, and the dataset was directly imported from the Keras API.

The project originially followed a post on Machine Learning Mastery (linked below) on designing a Cifar10 CNN from scratch. My intention was to create a similar CNN built from scratch, and to compare the results to a separate model based off the VGG16, retrained through transfer learning.

https://machinelearningmastery.com/how-to-develop-a-cnn-from-scratch-for-cifar-10-photo-classification/

Future training is still needed. A stronger GPU is required to train the VGG16, as there is a large number of trainable parameters and a large amount of data to preprocess.

---------------------

### The files in this repository include:
- Main.ipynb: Runs all functions in a Jupyter Notebook and plots results
- Load-Data.py: Class for loading in and preprocessing the Cifar10 dataset
- Report.py: Function that plots and saves important variables of a trained model
- Model.py: Class that builds out a CNN from scratch
- vggModel.py: Function that imports VGG16 model and reshapes it to be used for Cifar data
- Testing.py: Testing suite designed to run all required functions to train CNN
- TestingVGG.py: Testing suite designed to run all required functions to retrain the VGG16 model
