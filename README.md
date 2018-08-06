# P3--Self-driving-Car-Nanodegree
---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./Images/LeNet.png "LeNet Architecture"
[image2]: ./Images/cnn_nvidia.png "Model Visualization"
[image3]: ./Images/center_2018_08_05_23_15_28_268.jpg "CenterCamera"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"


### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model_as_taught.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* ReadMe.md summarizing the results
* AutonomousDrive_CW.mp4 and AutonomousDrive_CCW.mp4 showing the video while the car runs autonomously


#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model_as_taught.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

I tried with both the LeNet and the nVIDIA bases network architecture. In fact both the architecture are providing almost similar performance. The actual controller has been found to be the dataset. The code snippet below shows the implementation of both the types of networks.The flags were helpful in using a particular type of network for a given dataset.
```
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
 ```
 The architecture can be visually represented as the following:
 ![alt text][image1]
 
However, as the my final network I have frozen a network that is based on the one from Nvidia's CNN for Self-driving car. ([Click for details.](https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf "Click for details."))
The original architecture is presented below for ready reference.
 ```
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
```
![alt text][image2]

The model includes RELU layers to introduce nonlinearity, and the data is normalized in the model using a Keras lambda layer. 

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting. 

The model was trained and validated on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road. The vehicle was facing some serious trouble after crossing the bridge. So, some extra data were generated near this location. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to compare performance between LeNet and nVIDIA architecture.

My first step was to use a convolution neural network model similar to the LeNet. I thought this model might be appropriate because and eventually it appered to be so.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model to include the dropout layers. I also included some extra training data for various sections of the road circuit.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track. To improve the driving behavior in these cases, I manually simulated the crashing situation and made the model train for such situation.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model_as_taught.py) used the whole data stored as a list. However, another code was written (model_as_taught_with_generator.py) to handle large data. When side camera recordings were used, the generator based code was used. However, that seemd to be overfitting the datas. So, finally I decided to go without the side camera data. Therefore, the generator based code is not being used in the final version.

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image3]

To augmented the data sat, I also flipped images and angles thinking that this would remove the bias towards one sided driving. In fact this was the most important augmetation trick that came in useful.

I finally randomly shuffled the data set and put 20% of the data into a validation set. There were 13945 many data in th etraining set and 3487 many in the validation set.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 1 as evidenced by the observation of the loss values. I used an adam optimizer so that manually training the learning rate wasn't necessary.
