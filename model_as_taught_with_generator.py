import csv
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


lines = []

csv_file = './data/driving_log_small.csv'
car_images = []
steering_angles = []

FlipFlag = True
LeftCameraFlag = False
RightCameraFlag = False
LocalMachineFlag = True
Mfactor = 1

LeNetFlag = False
nVIDIAflag = True


if (LeftCameraFlag == True) and (RightCameraFlag == True):
    Mfactor = 3

def normalize(image_data):
    a = -0.5
    b = 0.5
    x_min = 0.0
    x_max = 255.0
    return a + (((image_data - x_min)*(b-a))/(x_max - x_min))

def process_image(img):
    new_img = img
    #new_img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    #new_img = normalize(new_img)
    return new_img


samples = []
with open(csv_file) as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

import cv2
import numpy as np
from sklearn.utils import shuffle

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                correcton = 0.25
                source_path_center = batch_sample[0]
                filename_center = source_path_center.split('\\')[-1]   
                if LocalMachineFlag == True:
                    current_path_center = 'd:/Users/SNP/CarND-Behavioral-Cloning-P3/data/IMG_small/' + filename_center 
                if LocalMachineFlag == False:
                    current_path_center = './data/IMG/' + filename_center                    
                #print('current file name = ',current_path_center)
                center_image = cv2.imread(current_path_center)
                center_image = process_image(center_image)
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)                
                
                if LeftCameraFlag == True:
                    
                    source_path_left = batch_sample[1]
                    filename_left = source_path_left.split('\\')[-1]   
                    if LocalMachineFlag == True:
                        current_path_left = 'd:/Users/SNP/CarND-Behavioral-Cloning-P3/data/IMG_small/' + filename_left 
                    if LocalMachineFlag == False:
                        current_path_left = './data/IMG/' + filename_left                    
                    #print('current file name = ',current_path_center)
                    left_image = cv2.imread(current_path_left)
                    left_image = process_image(left_image)
                    left_angle = float(batch_sample[3])  + 0.25
                    images.append(left_image)
                    angles.append(left_angle)                     
                    
                if RightCameraFlag == True:
                    correcton = 0.25
                    source_path_right = batch_sample[2]
                    filename_right = source_path_right.split('\\')[-1]   
                    if LocalMachineFlag == True:
                        current_path_right = 'd:/Users/SNP/CarND-Behavioral-Cloning-P3/data/IMG_small/' + filename_right 
                    if LocalMachineFlag == False:
                        current_path_right = './data/IMG/' + filename_right                   
                    #print('current file name = ',current_path_center)
                    right_image = cv2.imread(current_path_right)
                    right_image = process_image(right_image)
                    right_angle = float(batch_sample[3]) - 0.25    
                    images.append(right_image)
                    angles.append(right_angle)                     
                

                
                

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield shuffle(X_train, y_train)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

ch, row, col = 3, 90, 320  # Trimmed image format


JustDataCheck = False
#print(asd)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout
from keras.layers import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.layers import Cropping2D


if LeNetFlag == True:
    model = Sequential()
    model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(160,320,3)))
    #model.add(Lambda(lambda x: x/255.0-0.5))
    model.add(Convolution2D(6,5,5, activation='relu'))
    model.add(MaxPooling2D())
    model.add(Convolution2D(6,5,5, activation='relu'))
    model.add(MaxPooling2D())    
    model.add(Flatten())
    model.add(Dense(120))
    model.add(Dense(84))
    model.add(Dense(1))

if nVIDIAflag == True:
    model = Sequential()
    model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(160,320,3)))
    model.add(Lambda(lambda x: x/255.0-0.5))
    model.add(Convolution2D(24,5,5, border_mode ='valid', activation='relu'))
    model.add(MaxPooling2D())
    #model.add(Dropout(0.2))
    model.add(Convolution2D(36,5,5, border_mode ='valid', activation='relu'))
    model.add(MaxPooling2D())     
    #model.add(Dropout(0.2))
    model.add(Convolution2D(48,5,5, border_mode ='valid', activation='relu'))
    model.add(MaxPooling2D())
    #model.add(Dropout(0.2))
    model.add(Convolution2D(64,3,3, activation='relu'))
    model.add(MaxPooling2D())          
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dropout(0.5))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))
        
model.compile(loss = 'mse', optimizer='adam')
#model.fit(X_train,y_train,nb_epoch=1,validation_split=0.95, shuffle = True)
model.fit_generator(train_generator, 
                    samples_per_epoch= Mfactor*len(train_samples), 
                    validation_data=validation_generator, 
                    nb_val_samples=Mfactor*len(validation_samples), 
                    nb_epoch=1)   

model.save('model_AWS.h5')