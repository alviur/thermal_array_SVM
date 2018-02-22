"""
===============================================================================
Author: Alexander Gomez Villa - SUPSI
Advisor: PhD Augusto Salazar - UdeA
email: alexander.gomezvilla@supsi.ch
-------------------------------------------------------------------------------
===============================================================================
"""

import numpy as np
import os
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras.optimizers import SGD
from keras.utils import np_utils
from keras import applications,callbacks
from keras import backend as K
from keras.models import Model

mainPath = '/media/lex/1005c5a7-fb93-42cf-b7e8-6ba1cc7f9917/CNN_folds/images/F1/'
savePath = '/media/lex/1005c5a7-fb93-42cf-b7e8-6ba1cc7f9917/CNN_folds/results/CNN/finetuning/VGG16/'

outputNames = ['F1','F2','F3','F4','F5']

for foldN in outputNames:
    
    print('============================================================')
    print('Training Fold '+foldN)
    print('============================================================')
    # build the VGG16 network
    modelRaw = applications.VGG19(include_top=False, weights='imagenet',input_shape = (224,224,3))
    last = modelRaw.output
    numClasses = 1
    
    x = Flatten()(last)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(numClasses, activation='linear')(x)
    
    model = Model(inputs=modelRaw.input, outputs=predictions)
    
    sgd = SGD(lr=0.0001)
    model.compile(optimizer='adam',
                  loss='mean_absolute_error',
                  metrics=['mean_absolute_error'])
    
    
    # set the first 25 layers (up to the last conv block)
    # to non-trainable (weights will not be updated)
    for layer in model.layers[:23]:
        layer.trainable = False   
       
    
    batch_size = 80
    nb_train_samples = 456960
    nb_validation_samples = 456960
    imageSize = 224
    
    
    epochs = 50
    
    
    datagen = ImageDataGenerator(rescale=1. / 255)
    trainPath = mainPath+foldN+'/train'
    list_of_values_train = os.listdir(trainPath)
    train_generator = datagen.flow_from_directory(
            trainPath,
            target_size=(imageSize, imageSize),
            batch_size=batch_size,
            class_mode='sparse',  # this means our generator will only yield batches of data, no labels
            shuffle=True)
    
    validation_generator = datagen.flow_from_directory(
            mainPath+foldN+'/test',
            target_size=(imageSize, imageSize),
            batch_size=batch_size,
            class_mode='sparse',
            shuffle=True)
    
    #def regression_flow_from_directory(flow_from_directory_gen, list_of_values):
    #    for x, y in flow_from_directory_gen:
    #        yield x, list_of_values[y]
    
    # fine-tune the model
    callbacks = [callbacks.EarlyStopping(monitor='val_loss',patience=2,verbose=0)]
    model.fit_generator(
            train_generator,        
            epochs=epochs,
            validation_data=validation_generator,
            steps_per_epoch= nb_train_samples // batch_size,
            validation_steps=100,
            callbacks=callbacks)
    
    model.save(savePath+'/VGG16_MAE_'+foldN+'.h5')
    del model

#plt.imshow(data[:,:,:,70100])
#plt.show()
#
#
#plt.plot(label)
#plt.show()