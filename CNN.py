'''If anyone wants to run this code, they can simply copy the code in Jupyter/Colab/any python IDE of their choice
Just have to change the 'train_path' variable
NOTE: If using Google Colab, upload the desired dataset on your google drive, right click on the dataset (in drive) will give you an
option of path. Copy that path and assign it to the 'train_path' vairable in the script'''
import numpy as np
import pandas as pd
import os
#import matplotlib.pyplot as plt
import cv2
#import csv
#from keras.utils import to_categorical
from keras.layers import Dense,Conv2D,Flatten,MaxPooling2D,Dropout, BatchNormalization, Activation
from keras.models import Sequential
from keras.optimizers import rmsprop
#import random
#from PIL import Image
from sklearn.model_selection import train_test_split

np.random.seed(1)


# Processing training data
# -> appending images in a list 'train_images'
# -> appending labels in a list 'train_labels'

train_images = []
train_labels = []
shape = (50, 50)
train_path = 'C://Users//Administrator//Desktop//ML_project//Training_set//GTSRB//Final_Training//Images'
for c in range(43):
        sub_dir = train_path + '//' + format(c, '05d')  # subdirectory for class
        for filename in os.listdir(sub_dir):
            if filename.split('.')[1] == 'ppm':
                img = cv2.imread(os.path.join(sub_dir, filename))
                train_labels.append(c)
                # Resize all images to a specific shape
                img = cv2.resize(img, shape)

                train_images.append(img)
# Converting train_images to array
train_images = np.array(train_images)
# Converting labels into One Hot encoded sparse matrix
train_labels = pd.get_dummies(train_labels).values


# Splitting Training data into train and validation dataset
x_train, x_val, y_train, y_val = train_test_split(train_images, train_labels, random_state=1)


# Creating a Sequential model
model = Sequential()
model.add(Conv2D(32, kernel_size=3, activation='relu', input_shape=(50, 50, 3)))
model.add(BatchNormalization())
model.add(MaxPooling2D(strides=(2, 2)))
model.add(Dropout(0.3))

model.add(Conv2D(64, kernel_size=3, activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(strides=(2, 2)))
model.add(Dropout(0.3))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(43, activation='softmax'))


model.compile(
    loss='categorical_crossentropy',
    metrics=['acc'],
    optimizer='RMSprop'
)

# Training the model
history = model.fit(train_images, train_labels, epochs=5,batch_size=50,validation_data=(x_val,y_val))






