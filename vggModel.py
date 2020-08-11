from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

def vggModel():
    model = VGG16()
    print('Original Summary')
    model.summary()
    
    transfer = Sequential()
    for layer in model.layers[:-1]:
        transfer.add(layer)
    print('After adding layers')
    transfer.summary()
            
    for layer in transfer.layers:
        layer.trainable = False
    
    transfer.add(Dense(10,activation = 'relu'))
    print('Dense 10')
    transfer.summary()

    
    opti = SGD(lr = 0.001, momentum = 0.9)
    transfer.compile(optimizer = opti, loss =  'categorical_crossentropy', metrics = ['accuracy'])

    return transfer