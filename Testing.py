from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd

import Load_data
from Load_data import Data

import Model
from Model import CNN_Model

import report
from report import report

class RunTesting(object):

    def __init__(self, fractionOfData, patience, epochs, imageSize):
        self.fractionOfData = fractionOfData
        self.patience = patience
        self.epochs = epochs
        self.imageSize = imageSize
        
    def run(self):

        load_cifar = Data()
        
        X_train, y_train, X_test, y_test = load_cifar.load_dataset()
        X_train, y_train, X_test, y_test = load_cifar.reduceDataSize(X_train,
                                                                     y_train, 
                                                                     X_test, 
                                                                     y_test, 
                                                                     self.fractionOfData)
        load_cifar.dataSize(X_train, y_train, X_test, y_test)
        
        if self.imageSize != 32:
            X_train, X_test = load_cifar.upscaleData(X_train, X_test, self.imageSize)
        
        X_train, X_test = load_cifar.normalize(X_train, X_test)
        
        model = CNN_Model()
        model.convolutionalBlock(1)
        model.convolutionalBlock(2)
        model.convolutionalBlock(4)
        model.denseLayers()
        model.modelSummary()
        
        modelA = model.compileModel()
        
#         model = CNN_Model()
#         modelA = model.threeVGG()

##        modelA = vggModel()

#         modelA = newModel
        
        early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=self.patience)
        
        datagen = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)
        it_train = datagen.flow(X_train, y_train, batch_size=64)

        steps = int(X_train.shape[0] / 64)
        history = modelA.fit(it_train,
                             steps_per_epoch=steps, 
                             epochs=self.epochs, 
                             validation_data=(X_test, y_test), 
                             verbose=1,
                             callbacks = [early_stop]
                            )
        
        report(history, modelA, X_test, y_test)

        return modelA
        