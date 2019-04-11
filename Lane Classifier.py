#!/usr/bin/env python
# coding: utf-8

# ## Import all libraries we need

# In[1]:


import numpy as np
import keras 
from keras import backend as k 
from keras.models import Sequential
from keras.layers import Activation , Conv2D , MaxPooling2D , Dropout , Dense , Flatten
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import *


# ## Handeling our data

# In[2]:


train_path = 'lanes-and-nolanes/train' 
validation_path = 'lanes-and-nolanes/validation'
test_path = 'lanes-and-nolanes/test'


# In[3]:


train_batches = ImageDataGenerator().flow_from_directory(train_path,target_size=(80,160),classes=['lanes','nolanes'],batch_size=100)
validation_batches = ImageDataGenerator().flow_from_directory(validation_path,target_size=(80,160),classes=['lanes','nolanes'],batch_size=10)
test_batches = ImageDataGenerator().flow_from_directory(test_path,target_size=(80,160),classes=['lanes','nolanes'],batch_size=2)


# In[ ]:


imgs1 ,labels1 = next(train_batches)
imgs2 ,labels2 = next(validation_batches)
imgs3 ,labels3 = next(test_batches)


# ## Building our model and train it

# In[ ]:


# pool size and input shape are paramaters to fiddle with for optimization
pool_size = (2, 2)
input_shape = imgs1.shape[1:]

### Here is the actual neural network ###
model = Sequential()
# Normalizes incoming inputs. First layer needs the input shape to work
model.add(BatchNormalization(input_shape=input_shape))

# Below layers were re-named for easier reading of model summary; this not necessary
# Conv Layer 1
model.add(Conv2D(8, (3, 3), padding='valid', strides=(1,1), activation = 'relu', name = 'Conv1'))

# Conv Layer 2
model.add(Conv2D(16, (3, 3), padding='valid', strides=(1,1), activation = 'relu', name = 'Conv2'))

# Pooling 1
model.add(MaxPooling2D(pool_size=pool_size))

# Conv Layer 3
model.add(Conv2D(16, (3, 3), padding='valid', strides=(1,1), activation = 'relu', name = 'Conv3'))
model.add(Dropout(0.2))

# Conv Layer 4
model.add(Conv2D(32, (3, 3), padding='valid', strides=(1,1), activation = 'relu', name = 'Conv4'))
model.add(Dropout(0.2))

# Conv Layer 5
model.add(Conv2D(32, (3, 3), padding='valid', strides=(1,1), activation = 'relu', name = 'Conv5'))
model.add(Dropout(0.2))

# Pooling 2
model.add(MaxPooling2D(pool_size=pool_size))

# Conv Layer 6
model.add(Conv2D(64, (3, 3), padding='valid', strides=(1,1), activation = 'relu', name = 'Conv6'))
model.add(Dropout(0.2))

# Conv Layer 7
model.add(Conv2D(64, (3, 3), padding='valid', strides=(1,1), activation = 'relu', name = 'Conv7'))
model.add(Dropout(0.2))

# Pooling 3
model.add(MaxPooling2D(pool_size=pool_size))
          
# Flatten
model.add(Flatten())

# Fully connacted layer 1
model.add(Dense(30, activation = 'relu') , name = 'fcc1')

# Fully connacted layer 2
model.add(Dense(10, activation = 'relu' , name = 'fcc2'))

# Fully connacted layer 3
model.add(Dense(2, activation = 'softmax' , name = 'fcc3'))

# Summery of our model
model.summary()

# Training our model
model.compile(optimizer = 'Adam', loss = 'categorical_crossentropy', metrics=['accuracy'])
model.fit_generator(train_batches,steps_per_epoch = len(imgs1)/100,validation_data = validation_batches                     ,validation_steps = len(imgs2)/10 ,epochs = 20,verbose=2)

# Saving our model 
model.save('lane_classifier.h5')


# ## Predicting

# In[ ]:


predictions = model.predict_generator(test_batches,steps = len(imgs3)/2,verbose=0)
print(predictions)

