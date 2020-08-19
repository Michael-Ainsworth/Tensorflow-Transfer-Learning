from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.optimizers import SGD


# Class to build out CNN model using various building blocks

class CNN_Model(object):

    # Initialize Sequential model

    def __init__(self):
        self.model = Sequential()
    

    # Method to add a single convolutional block to the model. Subsequent
    # convolutional blocks allow for a variable number of filters.

    def convolutionalBlock(self, number):

        self.model.add(Conv2D(32*number, (3, 3), activation = 'relu', padding='same', input_shape = (32,32,3)))
        self.model.add(Conv2D(32*number, (3, 3), activation = 'relu', padding='same'))
        self.model.add(MaxPooling2D((2,2)))
        self.model.add(Dropout(0.2))


    # Flatten convolutional output and add a dense layer with 10 nodes

    def denseLayers(self):
        
        # Flatten and add a dense layer to output classes
        self.model.add(Flatten())
        self.model.add(Dense(128, activation = 'relu'))
        self.model.add(Dropout(0.4))
        self.model.add(Dense(10, activation = 'softmax'))
        

    # Print model summary

    def modelSummary(self):
        
        self.model.summary()


    # Set optimizer and compile. Return model.

    def compileModel(self):
        opti = SGD(lr = 0.001, momentum = 0.9)
        self.model.compile(optimizer = opti, loss =  'categorical_crossentropy', metrics = ['accuracy'])

        return self.model