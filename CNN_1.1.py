'''If anyone wants to run this code, they can simply copy the code in Jupyter/Colab/any python IDE of their choice
Just have to change the 'train_path' variable
NOTE: >Saved trained model is used in this code, that is, the model was first trained and then saved in the local machine. Saved model is used so as to save time spend in trainng
       the model repeatedly. 
      >Model was saved using joblib library, you can refer to it's documentaion for further information. 
      >The entire code is functional, however just to check if the model really predicts the correct class of a random image of traffic signs(downloaded from google), 
       therefore most of the code is commented. Feel free to uncomment it and run/ modify it.
      >If using Google Colab, upload the desired dataset on your google drive, right click on the dataset (in drive) will give you an
       option of path. Copy that path and assign it to the 'train_path' vairable in the script'''
import joblib
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import cv2
import csv
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


'''# Splitting Training data into train and validation dataset
x_train, x_val, y_train, y_val = train_test_split(train_images, train_labels, random_state=1)'''


# Processing testing data
# -> appending images in a list 'test_images'
# -> appending labels in a list 'test_labels'
# The test data contains labels as well also we are appending it to a list but we are'nt going to use it while training.

test_images = []
test_labels = []
shape = (50, 50)
test_path = 'C://Users//Administrator//Desktop//ML_project//Test_set//GTSRB//Final_Test//Images'

for filename in os.listdir(test_path):
    if filename.split('.')[1] == 'ppm':
        img = cv2.imread(os.path.join(test_path, filename))
        # Resize all images to a specific shape
        img = cv2.resize(img, shape)
        test_images.append(img)
    if filename.split('.')[1] == 'csv':
        csv_file = open(os.path.join(test_path, filename))
        csv_reader = csv.reader(csv_file, delimiter=';')
        i = 0
        for row in csv_reader:
            if i == 0:
                i += 1
                continue
            else:
                label = int(row[7])
                test_labels.append(label)
        csv_file.close()

# Converting test_images to array
test_images = np.array(test_images)
test_labels = np.array(test_labels)
test_labels = pd.get_dummies(test_labels).values
#print(test_labels)

#Since model has already beeen created and saved therefore no need to create it again
# Creating a Sequential model
'''model = Sequential()
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
    metrics=['accuracy'],
    optimizer='RMSprop'
)'''

# Training the model
#history = model.fit(train_images, train_labels, epochs=10,batch_size=50,validation_data=(test_images,test_labels))


#final_model = 'C://Users//Administrator//Desktop//ML_project//model//finalized_model.sav'
# save the model to disk
#joblib.dump(model, final_model)


'''index = np.random.randint(low=0, high=12629)
# Testing predictions and the actual label
checkImage = test_images[index -1 : index]
checklabel = test_labels[index -1 : index]'''


unkown_img_test = []
#Read the image(downloaded form internet) 
unkown_img = cv2.imread('C://Users//Administrator//Desktop//ML_project//20200622_231946-1.jpg')
#Resizing the image
unkown_img = cv2.resize(unkown_img, shape)
unkown_img_test.append(unkown_img)
checkImage = unkown_img_test[0 : 1]
# load the model from disk
loaded_model = joblib.load('C://Users//Administrator//Desktop//ML_project//model//finalized_model.sav')

#predicting the closest class it belongs to
predict = loaded_model.predict(np.array(checkImage))

print("Predicted class of the input image :- ",np.argmax(predict))

#For me the output came out to be 25, that is , according to this model that random image belongs to class 25
#opening any image of class 25 to compare if the input image(downloaded one) really belongs to class 25
unknown_img_show = []
img = cv2.imread('C://Users//Administrator//Desktop//ML_project//Training_set//GTSRB//Final_Training//Images//00025//00000_00001.ppm')
img = cv2.resize(img, shape)
unknown_img_show.append(img)
unknown_img_show = np.array(unknown_img_show)
plt.imshow(unknown_img_show[0])
plt.show()







