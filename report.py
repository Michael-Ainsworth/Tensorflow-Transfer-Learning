import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
import os


def report(history, model, X_test, y_test): 
     
    losses = pd.DataFrame(model.history.history)
    a = np.array(history.history['accuracy'])
    b = np.array(history.history['val_accuracy'])
    data = np.concatenate((np.expand_dims(a, axis=1),np.expand_dims(b, axis=1)), axis = 1)
    accuracies = pd.DataFrame(data, columns = ['Accuracy', 'Validation Accuracy'])

    fig = plt.figure(figsize = (8,6))
    plt.plot(losses[['loss','val_loss']]['loss'], c = 'blue')
    plt.plot(losses[['loss','val_loss']]['val_loss'], c = 'red')
    plt.ylabel('Categorical Crossentropy Loss')
    plt.xlabel('Training Epochs')
    plt.title('Evolution of Losses')
    plt.legend(['Loss','Validation Loss'], loc = 'upper right', fontsize = 14)
    
    my_path = os.path.dirname(os.path.abspath("__file__"))
    my_path = my_path + '/Diagnostics'
    my_file = 'losses.png'
    fig.savefig(os.path.join(my_path, my_file)) 
    
    fig = plt.figure(figsize = (8,6))
    plt.plot(accuracies[['Accuracy','Validation Accuracy']]['Accuracy'], c = 'green')
    plt.plot(accuracies[['Accuracy','Validation Accuracy']]['Validation Accuracy'], c = 'purple')
    plt.ylabel('Model Accuracy')
    plt.xlabel('Training Epochs')
    plt.title('Evolution of Accuracies')
    plt.legend(['Accuracy','Validation Accuracy'], loc = 'lower right', fontsize = 14)

    my_path = os.path.dirname(os.path.abspath("__file__"))
    my_path = my_path + '/Diagnostics'
    my_file = 'accuracies.png'
    fig.savefig(os.path.join(my_path, my_file))        

    _, accuracy = model.evaluate(X_test, y_test, verbose=0)
    
    print('\n')
    print('Accuracy: {}%'.format(accuracy * 100.0))
    print('\n')