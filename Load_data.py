import matplotlib.pyplot as plt
import numpy as np
import math
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from skimage.transform import resize


# Create a class to import all Cifar10 data. Design individual methods to load in dataset,
# reduce the size of the data, normalize the data, upscale the data, and to visualize
# specific parts of the dataset.

class Data(object):

    def __init__(self):
        self.data = cifar10


    # Load train and test dataset. Labels are letters. Use one hot encoding to convert 
    # labels to binary.

    def load_dataset(self):
        
        (X_train, y_train), (X_test, y_test) = self.data.load_data()

        uniqueLabels = set()
        for value in y_train:
            uniqueLabels.add(value[0])

        print('Unique labels include:')
        print(uniqueLabels)
        print('\n')

        y_train = to_categorical(y_train)
        y_test = to_categorical(y_test)

        return X_train, y_train, X_test, y_test


    # Method for testing with a fraction of the dataset. Splits entire dataset into
    # a certain fraction and only trains with that fraction. Reduces workload so model
    # can be trained on CPU.
    
    def reduceDataSize(self, X_train, y_train, X_test, y_test, fractionOfData):

        stopTrain = math.ceil(len(X_train) * fractionOfData)
        stopTest = math.ceil(len(X_test) * fractionOfData)

        X_train = X_train[:stopTrain]
        y_train = y_train[:stopTrain]
        X_test = X_test[:stopTest]
        y_test = y_test[:stopTest]

        return X_train, y_train, X_test, y_test

    
    # Display size of data

    def dataSize(self, X_train, y_train, X_test, y_test):

        print('X Train = {},  y train = {}'.format(X_train.shape, y_train.shape))
        print('X Test = {},  y test = {}'.format(X_test.shape, y_test.shape))
        print('\n')
        

    # Visualize the first n images in the dataset where n is the variable numOfImages

    def showImages(self, X_train, numOfImages):

        # Determine shape of subplot based on images printed
        val = math.ceil(np.sqrt(numOfImages))  

        for position in range(numOfImages):
            plt.subplot(val,val,position+1)
            plt.imshow(X_train[position])
        plt.show()


    # Visualize a specific image number in the dataset

    def showSingleImage(self, X_train, imageNumber):

        plt.imshow(X_train[imageNumber])
        plt.show()


    # Show label for given image number

    def showLabel(self, y_train, imageNumber):

        return y_train[imageNumber]


    # Normalize pixel data to have 0 mean and unit variance. Showed better results than
    # normalizing between 0 and 1.

    def normalize(self, X_train, X_test):
    
        mean = np.mean(X_train)
        std = np.std(X_train)
        
        X_train = (X_train - mean) / (std)
        X_test = (X_test - mean) / (std)
        
        return X_train, X_test
    
    
    # Upscale data in case of using model with different input shape
    
    def upscaleData(self, X_train, X_test, size):
    
        X_train_resized = resize(X_train, (X_train.shape[0], size, size, 3))
        X_test_resized = resize(X_test, (X_test.shape[0], size, size, 3))

        return X_train_resized, X_test_resized