
# coding: utf-8

# In[9]:

from keras.models import model_from_json
from keras.layers import Input, Dense
from keras.models import Model
import csv
import pandas as pd
import numpy as np
import cv2
from numpy import newaxis
from PIL import Image
from numpy import array
import glob, os
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Lambda, ELU,Activation,MaxPooling2D
from keras.layers.convolutional import Convolution2D
from numpy import random
from keras.optimizers import SGD, Adam, RMSprop
import json
import os
import h5py
import math
from sklearn.cross_validation import train_test_split


# In[10]:

#Reading CSV file for image and steering angle details
file_name = 'driving_log.csv'
# data = read_data(file_name)
# data = pd.DataFrame(data)
# data

data = pd.read_csv(file_name)
# data = data[0:100][:]
data.head()


# In[11]:

# Crop the image to remove portion containing useless information like hood of car or top part of trees
def resize_image(image):
    shape = image.shape
    image = image[math.floor(shape[0]/5):shape[0]-25, 0:shape[1]]
    image1 = cv2.resize(image,(new_col, new_row),interpolation=cv2.INTER_AREA)    
    return image1


# Adding brightness to incorporate different level of available light situation while capturing pics
def add_random_brightness(image, rand_val=.25+np.random.uniform()):
    image_tran = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    image_tran[:,:,2] = image_tran[:,:,2]*rand_val
    return cv2.cvtColor(image_tran,cv2.COLOR_HSV2RGB)


# Function to randmly flip the image and reverse the angle
def flip_image(image,y_steer):
    if np.random.randint(2) == 0:
        image = cv2.flip(image,1)
        y_steer = -y_steer
    return image,y_steer

# Randomly selecting one of the three images and applying different preprocessing before using it for modeling
def preprocess_image(line):
    i = np.random.randint(3)
    if (i == 0):
        file = line['center'][0].strip()
        shift_ang = 0.0
    if (i == 1):
        file = line['left'][0].strip()
        shift_ang = 0.25        
    if (i == 2):
        file = line['right'][0].strip()
        shift_ang = -0.25
    y = line['steering'][0] + shift_ang
    image = cv2.cvtColor(cv2.imread(file),cv2.COLOR_BGR2RGB)
    image = add_random_brightness(image,.25+np.random.uniform())
    image = np.array(resize_image(image))
    return flip_image(image,y)



# In[12]:

# Batch Generator function for training
def generate_train(data,batch_size = 128):
    batch_steering = np.zeros(batch_size)
    batch_images = np.zeros((batch_size, new_col, new_row, 3))
    
    while 1:
        for i_batch in range(batch_size):
            i = np.random.randint(len(data))
            line = data.iloc[[i]].reset_index()
            x,y = preprocess_image(line)
            batch_images[i_batch] = x
            batch_steering[i_batch] = y
        yield batch_images, batch_steering
        
# Batch Generator function for validation
def generate_validate(data):
    while 1:
        for i in range(len(data)):
            line = data.iloc[[i]].reset_index()
            image = cv2.imread(line['center'][0].strip())
            image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
            image = np.array(resize_image(image))
            x = image.reshape(1, image.shape[0], image.shape[1], image.shape[2])
            y = np.array([[line['steering'][0]]])
            yield x, y




# In[22]:

plt.figure(figsize=(16,8))
for i_batch in range(10):
    i = np.random.randint(len(data))
    line = data.iloc[[i]].reset_index()
    x,y = preprocess_image(line)
#     plt.imshow(x)
    plt.subplot(2,5,i_batch+1)
    plt.imshow(x)
    plt.axis('off')
plt.show()


# In[14]:

# Load or define model
try:
    with open('model.json', 'r') as model_json:
        model = model_from_json(json.load(model_json))

    # Use adam optimizer and mean squared error for compiling the model
    model.compile("adam", "mse")

    # import weights
    model.load_weights('model.h5')

    print("Imported model and weights")


except:
    # New dimension of images
    new_col,new_row = 64, 64
    # Drop out layer prob
    drop_prob = 0.5
    # New dimension of input
    input_shape = (new_col, new_row, 3)
    # Filter size for Convolution
    filter_size = 3
    # Pool size for Max pooling
    pool_size = (2,2)

    model = Sequential()
    model.add(Lambda(lambda x: x/255.-0.5,input_shape=input_shape))

    model.add(Convolution2D(3,1,1,border_mode='valid', name='conv0', init='he_normal'))
    model.add(ELU())

    model.add(Convolution2D(24,filter_size,filter_size, border_mode='valid',name='conv1', init='he_normal'))
    model.add(ELU())

    model.add(Convolution2D(48,filter_size,filter_size,border_mode='valid',name='conv2', init='he_normal'))
    model.add(ELU())

    model.add(MaxPooling2D(pool_size=pool_size))
    model.add(Dropout(drop_prob))

    model.add(Convolution2D(64,filter_size,filter_size,border_mode='valid',name='conv3', init='he_normal'))
    model.add(ELU())

    model.add(MaxPooling2D(pool_size=pool_size))
    model.add(Dropout(drop_prob))


    model.add(Convolution2D(128,filter_size,filter_size, border_mode='valid', name='conv4', init='he_normal'))
    model.add(ELU())

    model.add(MaxPooling2D(pool_size=pool_size))
    model.add(Dropout(drop_prob))

    model.add(Flatten())

    model.add(Dense(512,name='conn1', init='he_normal'))
    model.add(ELU())
    model.add(Dropout(drop_prob))
    model.add(Dense(128,name='conn2', init='he_normal'))
    model.add(ELU())
    model.add(Dropout(drop_prob))
    model.add(Dense(64,name='conn3', init='he_normal'))
    model.add(ELU())
    model.add(Dropout(drop_prob))
    model.add(Dense(16,name='conn4',init='he_normal'))
    model.add(ELU())
    model.add(Dropout(drop_prob))
    # Final output layer 
    model.add(Dense(1, name='output', init='he_normal'))
    
    # Print model summary
#     model.summary()


# In[6]:

#Compile model using Adam optimizer
model.compile("adam", "mse")
# New size of images
new_col, new_row = 64, 64
# Batch size to be processed
batch_size = 256
# Validation set size
val_size = 1000

# Training model here using fit_generator to save memory by not loading all images at same time
history = model.fit_generator(generate_train(data,batch_size),nb_epoch=50,
            samples_per_epoch=batch_size*200, validation_data=generate_validate(data),
                        nb_val_samples=val_size)


# In[7]:

# Save model and weights
if 'model.json' in os.listdir():
    print("the model already exists")
else:
    # Save model as json file
    json_string = model.to_json()
    with open('model.json', 'w') as outfile:
        json.dump(json_string, outfile)

    # save model weights
    model.save_weights('./model.h5')
    print("Model saved")

