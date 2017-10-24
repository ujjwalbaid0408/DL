#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 16 13:25:44 2017
@author: ujjwal
Spyder editor
This is temporary script file
"""
#%%
from keras.datasets import mnist  # import the dataset
from keras.models import Sequential # import the type of model
from keras.layers import Dense, Dropout, Activation, Flatten # import layers
from keras.layers.convolutional import Convolution2D, MaxPooling2D # import convolution layers
from keras.utils import np_utils 
from keras import metrics
from keras import backend as K 
K.set_image_dim_ordering('th') 

#to plot figures and graphs import matplotlib
import matplotlib 
import matplotlib.pyplot as plt

#%%

#we are training the data in batches
batch_size = 128 

# number of output classes
nb_classes = 10 

# number of epochs to train
nb_epoch = 12

# input image dimension
img_rows, img_cols = 28, 28

# number of convilution filters to use
nb_filters = 32 

# size of pooling area for max pooling, here 2*2
nb_pool = 2 

# Convolution kernel size, here 3*3
nb_conv = 3

#%% 
# the data is loaded and split between train and test sets
# X_train and X_test are features/image features which are nothing but pixels
# y_train and y_test are labels
(X_train, y_train), (X_test, y_test) = mnist.load_data() 

# type X_train.shape in console you will length of database and size of image 
# There are 60k examples with size 28 by 28
# also type y_train.shape you will get how much labels it has

# reshape the data
#X_train = X_train.reshape(number of samples, channel, img_rows, img_cols) 
X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols) 
X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)

# Converting to float 32 format
X_train = X_train.astype('float32') 
X_test = X_test.astype('float32')
 
# Normalization with highest intensity
X_train /= 255 
X_test /= 255 


print('X_train shape:', X_train.shape) 
print(X_train.shape[0], 'train samples') 
print(X_test.shape[0], 'test samples')

# Convert class vectors to binary class matrices
# np_utils is to convert 2 as [0 0 1 0 0 0 0 0 0 0]
Y_train = np_utils.to_categorical(y_train, nb_classes) 
Y_test = np_utils.to_categorical(y_test, nb_classes) 

# for example print sample number 4600 with label
i = 0
plt.imshow(X_train[i, 0], interpolation="nearest") 
print('Label:', Y_train[i,:]) 

Y_train.shape # this shows each sample with label in binary class matrix
# now you can see all the variables in variable explorer and double click on 
# Y_test
#%% define model

model = Sequential() 

model.add(Convolution2D(nb_filters, nb_conv, nb_conv, 
                        border_mode='valid', 
                        input_shape=(1,img_rows,img_cols))) 

convout1 = Activation('relu') # rectified linear unit
model.add(convout1) 

model.add(Convolution2D(nb_filters, nb_conv, nb_conv)) 

convout2 = Activation('relu') 
model.add(convout2) 

model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool))) 
model.add(Dropout(0.25)) # for regularizing and avoid overfitting durinr training
 
model.add(Flatten()) 
# this will open up convolution  this is like fully connected layer


model.add(Dense(128)) 
model.add(Activation('relu')) 
model.add(Dropout(0.5)) 
model.add(Dense(nb_classes)) # last layer(output) will be equal to no of classes

model.add(Activation('softmax'))

# define optimizer and loss function
#model.compile(loss='categorical_crossentropy',optimizer='adadelta',metrics=[metrics.mae, metrics.categorical_accuracy])
model.compile(loss='categorical_crossentropy',optimizer='adadelta',metrics=['accuracy'])

#%% 

#model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch, 
#          show_accuracy=True, verbose=1, validation_data=(X_test, Y_test)) 

model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch, 
          verbose=1, validation_data=(X_test, Y_test)) 


# if we dont want to give validation data explicitly then we can split some 
# samples of training data to validation data
model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch, 
          verbose=1, validation_split=0.2) 

score = model.evaluate(X_test, Y_test) 

#print('Test score:', score) 
#print('Test accuracy:', score[1])
print(model.predict_classes(X_test[1:5]))
print(Y_test[1:5])



#%%

# To see entire congiguration of model ==> model.get_config() 
# in one curly brace we have one layer {}

# to see individual layer ==> model.layers[0].get_config()

# to see model parameters for complete model ==>model.count_params()  
# This will give total number of parametrs in the model

# to see individual layer parameters ==> model.layers[0].count_params()


model.summary()
from keras.utils.visualize_utils import plot
#graph = plot(model, to_file='my_model_cnn.png', show_shapes=True)



