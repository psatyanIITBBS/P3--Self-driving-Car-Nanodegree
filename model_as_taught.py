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
SideCameraFlag = False
LocalMachineFlag = True
LeNetFlag = False
nVIDIAflag = True

def normalize(image_data):
    a = -0.5
    b = 0.5
    x_min = 0
    x_max = 255
    return a + (((image_data - x_min)*(b-a))/(x_max - x_min))

def process_image(img):
    #new_img = img
    #new_img = normalize(img)
    #new_img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    return img

with open(csv_file, 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        steering_center = float(row[3])

        # create adjusted steering measurements for the side camera images
        correction = 0.27 # this is a parameter to tune
        steering_left = steering_center + correction
        steering_right = steering_center - correction
        
        source_path_center = row[0]
        filename_center = source_path_center.split('\\')[-1]
        source_path_left = row[1]
        filename_left = source_path_left.split('\\')[-1]
        source_path_right = row[2]
        filename_right = source_path_right.split('\\')[-1]   
        
        if LocalMachineFlag == True:
            #For running on local machine
            current_path_center = 'd:/Users/SNP/CarND-Behavioral-Cloning-P3/data/IMG_small/' + filename_center   
            current_path_left = 'd:/Users/SNP/CarND-Behavioral-Cloning-P3/data/IMG_small/' + filename_left  
            current_path_right = 'd:/Users/SNP/CarND-Behavioral-Cloning-P3/data/IMG_small/' + filename_right  
        if LocalMachineFlag == False:
            #For Submitting to AWS
            current_path_center = './data/IMG/' + filename_center   
            current_path_left = './data/IMG/' + filename_left  
            current_path_right = './data/IMG/' + filename_right          
    
        # read in images from center, left and right cameras
        #path = "..." # fill in the path to your training IMG directory
        
        img_center = process_image(np.asarray(Image.open(current_path_center)))
        img_left = process_image(np.asarray(Image.open(current_path_left)))
        img_right = process_image(np.asarray(Image.open(current_path_right)))        
        car_images.extend([img_center])
        steering_angles.extend([steering_center])
        
        if SideCameraFlag == True:
            car_images.extend([img_left])
            car_images.extend([img_right])
            steering_angles.extend([steering_left])
            steering_angles.extend([steering_right])
        if FlipFlag == True:
            car_images.extend([np.fliplr(img_center)])
            steering_angles.extend([-steering_center])
            if SideCameraFlag == True:
                car_images.extend([np.fliplr(img_left)])
                car_images.extend([np.fliplr(img_right)])
            if SideCameraFlag == True:
                steering_angles.extend([-steering_left])
                steering_angles.extend([-steering_right])       
  
X_train = np.array(car_images)
y_train = np.array(steering_angles)

JustDataCheck = False
#print(asd)
if ~JustDataCheck:
    from keras.models import Sequential
    from keras.layers import Flatten, Dense, Lambda, Dropout
    from keras.layers import Convolution2D
    from keras.layers.pooling import MaxPooling2D
    from keras.layers import Cropping2D
    
    if LeNetFlag == True:
        model = Sequential()
        model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(160,320,3)))
        model.add(Lambda(lambda x: x/255.0-0.5))
        model.add(Convolution2D(6,5,5, activation='relu'))
        model.add(MaxPooling2D())
        model.add(Dropout(0.25))
        model.add(Convolution2D(6,5,5, activation='relu'))
        model.add(MaxPooling2D())    
        model.add(Flatten())
        model.add(Dense(120))
        model.add(Dropout(0.5))
        model.add(Dense(84))
        model.add(Dense(1))
    if nVIDIAflag == True:
        model = Sequential()
        model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(160,320,3)))
        model.add(Lambda(lambda x: x/255.0-0.5))
        model.add(Convolution2D(24,5,5, border_mode ='valid', activation='relu'))
        model.add(MaxPooling2D())
        model.add(Convolution2D(36,5,5, border_mode ='valid', activation='relu'))
        model.add(MaxPooling2D())     
        model.add(Convolution2D(48,5,5, border_mode ='valid', activation='relu'))
        model.add(MaxPooling2D())
        model.add(Dropout(0.2))
        model.add(Convolution2D(64,3,3, activation='relu'))
        model.add(MaxPooling2D())          
        model.add(Flatten())
        model.add(Dense(100))
        model.add(Dropout(0.5))
        model.add(Dense(50))
        model.add(Dense(10))
        model.add(Dense(1))        
    
    model.compile(loss = 'mse', optimizer='adam')
    model.fit(X_train,y_train,nb_epoch=1,validation_split=0.2, shuffle = True)
    
    model.save('model.h5')