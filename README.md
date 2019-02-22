**Behavioral Cloning Project**

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)
![Udacity - Self-Driving Car P3](https://img.shields.io/badge/Status-Submitted-Yellow.svg)

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./report/NVIDIA_3_Epochs_Train_39115_validate_9779.png
[image2]: ./report/NVIDIA_5_Epochs_Train_38572_validate_9644.png
[image3]: ./report/NVIDIA_5_Epochs_Train_39115_validate_9779.png
[image4]: ./report/SmartSelect_20190220-222053_Video_Player.gif
[image5]: ./report/SmartSelect_20190220-222204_Video_Player.gif
[image6]: ./report/SmartSelect_20190220-222446_Video_Player.gif
[image7]: ./report/SmartSelect_20190220-222649_Video_Player.gif
[image8]: ./report/Trained-output-flowchart.png
[image9]: ./report/Training-flowchart.png
[image10]: ./report/nvidia-cnn-architecture.png
[image11]: ./report/center.jpg
[image12]: ./report/left.jpg
[image13]: ./report/right.jpg

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* main.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* README.md summarizing the results
* report folder that consist of multiple videos for the car driven by the designed model.


#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model is based on the End-to-End Deep Learning CNN designed by NVIDIA [[LINK]](https://devblogs.nvidia.com/deep-learning-self-driving-cars/) which consists of a 9 layers, including a normalization layer at the begining, 5 convolutional layers, and 3 fully connected layers. (main.py lines 135-147):

```python
model = Sequential()
model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=(160, 320, 3)))
model.add(Cropping2D(cropping=((75,25), (0,0)), input_shape=(3,160,320)))
model.add(Conv2D(24, (5, 5), strides=(2, 2), activation="relu"))
model.add(Conv2D(36, (5, 5), strides=(2, 2), activation="relu"))
model.add(Conv2D(48, (5, 5), strides=(2, 2), activation="relu"))
model.add(Conv2D(64, (3, 3), activation="relu"))
model.add(Conv2D(64, (2, 2), activation="relu"))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))
```

The model includes RELU layers to introduce non linearity (main.py code lines 138-142), and the data is normalized in the model using a Keras lambda layer (main.py code line 136). 

#### 2. Attempts to reduce overfitting in the model

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 90-102). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 25).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road where adding the input from left and right cameras help in getting the car to the center in case it drifted to one side of the road or another.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to create End-to-End Deep Learning for Self-Driving Car.

My first step was to use a convolution neural network model similar to the Lenet architecture, I thought this model might be appropriate because it had taken consideration of drop-outs and adding a non-linear factor by using RELUs.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that by adding some drop-out layers. but still did not help much. Thats when I decided to base my mode on the NVIDIA CNN model. Following is high level diagrams of the neural network training approach and trained network output:

![alt text][image9]
Training the neural network.

\/

![alt text][image8]
The trained network

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track, and that was specifically happening when the car was trying to do turns. to improve the driving behavior in these cases, I had to get a new data set for the car driving from the side of the road against the turn to make the trained network more robust against turns.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes 
My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 160x20x3 RGB image   							| 
| Lambda (Normalization)     	| ---> Mean = 0 	|
| Cropping Layer    	| cropping=(75 from top,25 from bottom) 	|
| RELU					|												|
| Convolution 3x3     	| 5x5 stride, same padding, outputs 24 levels 	|
| RELU					|												|
| Convolution 3x3     	| 5x5 stride, same padding, outputs 36 levels 	|
| RELU					|												|
| Convolution 3x3     	| 5x5 stride, same padding, outputs 48 levels 	|
| RELU					|												|
| Convolution 3x3     	| 3x3 stride, same padding, outputs 64 levels 	|
| RELU					|												|
| Convolution 3x3     	| 3x3 stride, same padding, outputs 64 levels 	|
| RELU					|												|
| Fully connected		| 						|
| Fully connected		| 						|
| Fully connected		| 						|

The network has more than 25 million connections and 250 thousand parameters.

Here is a visualization of the architecture 

![alt text][image10]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image11]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from left to center :

![alt text][image4]

Here is some exaples for left and right camera's input:

![alt text][image12]
![alt text][image13]

To augment the data sat, I also flipped images and angles thinking that this would make the network more robust and protect it from overfitting, For example, here is an image that has then been flipped:

![alt text][image11]
![alt text][image14]

After the collection process, I had 48894 number of data points. I then preprocessed this data by cropping 75 rows from top and 25 pixels from bottom, where those pixels are irrelevant to the behavior of the car while driving.


I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 3 as evidenced by the graphs below:

![alt text][image1]
![alt text][image2]
![alt text][image3]

I used an adam optimizer so that manually training the learning rate wasn't necessary.

Following is some of the out puts:

![alt text][image5]
![alt text][image6]
![alt text][image7]
