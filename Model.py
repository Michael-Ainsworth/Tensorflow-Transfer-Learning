from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.optimizers import SGD


class CNN_Model(object):

    def __init__(self):
        self.model = Sequential()
    
    def convolutionalBlock(self, number):

        # Add one convolutional block
        self.model.add(Conv2D(32*number, (3, 3), activation = 'relu', padding='same', input_shape = (32,32,3)))
        self.model.add(Conv2D(32*number, (3, 3), activation = 'relu', padding='same'))
        self.model.add(MaxPooling2D((2,2)))
        self.model.add(Dropout(0.2))

    def denseLayers(self):
        
        # Flatten and add a dense layer to output classes
        self.model.add(Flatten())
        self.model.add(Dense(128, activation = 'relu'))
        self.model.add(Dropout(0.4))
        self.model.add(Dense(10, activation = 'softmax'))
        
    def modelSummary(self):
        
        self.model.summary()

    def compileModel(self):
        # Set optimizer and compile
        opti = SGD(lr = 0.001, momentum = 0.9)
        self.model.compile(optimizer = opti, loss =  'categorical_crossentropy', metrics = ['accuracy'])

        return self.model