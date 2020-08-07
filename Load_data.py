import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import math
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical


class Data(object):

    def __init__(self):
        self.data = cifar10

    # Load train and test dataset. Labels are letters. Use one hot encoding to convert labels to binary.

    def load_dataset(self):
        
        (X_train, y_train), (X_test, y_test) = self.data.load_data()

        uniqueLabels = set()
        for value in y_train:
            uniqueLabels.add(value[0])

        print('Unique labels include:')
        print(uniqueLabels)

        y_train = to_categorical(y_train)
        y_test = to_categorical(y_test)

        return X_train, y_train, X_test, y_test

    
    # Display size of data

    def dataSize(self, X_train, y_train, X_test, y_test):

        print('X Train = {},  y train = {}'.format(X_train.shape, y_train.shape))
        print('X Test = {},  y test = d{}'.format(X_test.shape, y_test.shape))
        

    # Visualize the first n images in the dataset where n is the variable numOfImages

    def showImages(self, X_train, y_train, X_test, y_test, numOfImages):

        val = math.ceil(np.sqrt(numOfImages))  # Determine shape of subplot based on images printed

        # Plot each image into the subplot
        for position in range(numOfImages):
            ax1 = plt.subplot(val,val,position+1)
            ax1 = plt.imshow(X_train[position])
        plt.show()


    # Visualize a specific image number in the dataset

    def showSingleImage(self, X_train, y_train, X_test, y_test, imageNumber):

        plt.imshow(X_train[imageNumber])
        plt.show()

    
    def normalize(self, X_train, X_test):

        normalizedTrain = X_train.astype(np.float)
        normalizedTest = X_test.astype(np.float)

        normalizedTrain /= 255
        normalizedTest /= 255

        return normalizedTrain, normalizedTest