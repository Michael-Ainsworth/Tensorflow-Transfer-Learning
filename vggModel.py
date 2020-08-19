from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam


# Class for building the VGG model

def vggModel():

    # Import VGG model, restructure input shape to be the size of Cifar images
    model = VGG16(
                weights=None, 
                include_top=True,
                classes=10,
                input_shape=(32,32,3)
            )
    print('Original Summary')
    model.summary()
    
    # Convert model to a Sequential model
    transfer = Sequential()
    for layer in model.layers:
        transfer.add(layer)
    print('After adding layers')
    transfer.summary()


    # To run model with original VGG weights, input image must be size 224 x 224.
    # Run below code instead and specify an input shape of 224 when running the testing
    # class:

    # model = VGG16()
    # print('Original Summary')
    # model.summary()
    
    # transfer = Sequential()

    # for layer in model.layers[:-1]:
    #     transfer.add(layer)
    # print('After adding layers')
    # transfer.summary()
            
    # for layer in transfer.layers:
    #     layer.trainable = False
    
    # transfer.add(Dense(10,activation = 'relu'))
    # print('Dense 10')
    # transfer.summary()

    
    # Compile model
    opti = SGD(lr = 0.001, momentum = 0.9)
    transfer.compile(optimizer = opti, loss =  'categorical_crossentropy', metrics = ['accuracy'])

    return transfer