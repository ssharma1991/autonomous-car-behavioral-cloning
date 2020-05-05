# Behavioral Cloning for Self Driving Cars 

[//]: # (Image References)

[image1]: ./Readme_images/autonomous_speeded.gif "Autonomous Driving"


##  Pipeline

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road

The autonomous car takes camera images of the front view and decide the optimum steering angle according to the model. The model is inspired from [NVIDIA's deep learning CNN](https://devblogs.nvidia.com/deep-learning-self-driving-cars/) . [Udacity's self driving car simulator environment](https://github.com/udacity/self-driving-car-sim) is used for this project. The main files included in the project are:
* `model.py` which contains code to import data, augment data, create a model and train it.
* `drive.py` which controls the car autonomously in simulator environment.
* `model.h5` consists of the best trained model with lease mean squared error on validation set.
---

## Step 1- Data Collection from Simulator

Data is collected by running the Simulator in Training mode and manually driving the car. Images from three front facing cameras are taken and the associated steering angle at that instant is collected. 

I have collected data from both available tracks in the Simulator namely: Lake-side track and Jungle track. The motivation is to help generalize out model over varied environments and road conditions. 

Care is taken to include both **Controlled** and **Recovery** runs in the data. The controlled runs include driving the car in the middle of the lane so that the model learns the best practices. However, the recovery runs include small sections where the car recovers from the edge of road to its center. This is extremely important since this information will be required if the car makes a mistake and ends up deviating from the middle of the road. 

Image from each camera is a 160x320 RGB color image while the steering angle ranges from -25 degrees to 25 degrees.

---

## Step 2- Data Augmentation

Since the Simulator environments include closed loop tracks, the data tends to have a left or right steering wheel bias depending on clockwise or counter-clockwise movement of car. To overcome this issue, I use images mirrored across the vertical axis and attach a negative steering angle to them. This balances out the data and aids the training process.

To further enhance my model's ability to recover from a wrong steering decision and get back to center of road, the left and right camera images are used. For left image, steering angle is adjusted by +0.5 correction while for right image, steering angle is adjusted by -0.5 correction. 

The total data is split and shuffled into Training and Validation sets in 8:2 ratio. The validation set is used during training to avoid overfitting.

Also, the top area of the image conststs mostly of sky while the bottom area captures parts of the car's hood. Both of these areas are cropped out since they don't play any part in deciding the steering angle. Thus 65px from top and 30px from the bottom of the camera image are ignored.

Finally, the remaining image pixel values are normalised to stay in the range (-.5,.5).

## Step 3- Model Architecture and Training

The architecture of final convolution neural network used for autonomous driving is described as follows:

| Layer (type)                   |Output Shape       |Params  |
|--------------------------------|-------------------|-------:|
|lambda_1 (Lambda)               |(None, 160, 320, 3)|0       |
|cropping2d_1 (Cropping2D)       |(None, 65, 320, 3) |0       |
|conv2d_1 (Conv2D)               |(None, 61, 316, 24)|1824    |
|max_pooling2d_1 (MaxPooling2D)  |(None, 30, 63, 24) |0       |
|activation_1 (Activation)       |(None, 30, 63, 24) |0       |
|conv2d_2 (Conv2D)               |(None, 26, 59, 36) |21636   |
|max_pooling2d_2 (MaxPooling2D)  |(None, 13, 29, 36) |0       |
|activation_2 (Activation)       |(None, 13, 29, 36) |0       |
|conv2d_3 (Conv2D)               |(None, 11, 27, 48) |15600   |
|max_pooling2d_3 (MaxPooling2D)  |(None, 5, 13, 48)  |0       |
|activation_3 (Activation)       |(None, 5, 13, 48)  |0       |
|conv2d_4 (Conv2D)               |(None, 3, 11, 64)  |27712   |
|max_pooling2d_4 (MaxPooling2D)  |(None, 3, 11, 64)  |0       |
|conv2d_5 (Conv2D)               |(None, 1, 9, 80)   |46160   |
|max_pooling2d_5 (MaxPooling2D)  |(None, 1, 9, 80)   |0       |
|flatten_1 (Flatten)             |(None, 720)        |0       |
|dropout_1 (Dropout)             |(None, 720)        |0       |
|dense_1 (Dense)                 |(None, 100)        |72100   |
|dense_2 (Dense)                 |(None, 50)         |5050    |
|dense_3 (Dense)                 |(None, 10)         |510     |
|dense_4 (Dense)                 |(None, 1)          |11      |
|-                               |**Total params**   |190603  |

For training, I used mean squared error for the loss function since we are dealing with a regression problem. I used Adam optimizer for optimization with learning rate of 1.0e-4 which resulted in better learning. Although, the maximum epochs were set to 10, the training ended before it due to repeated increase in validation error. The best fit model is saved in the file `model.h5`

---

## Step 4- Autonomous mode testing in Simulator

The model is tested on the Lake-side track by running the car in autonomous mode. One of the most problematic areas was sharp left turn after the bridge with a dirt pavement. The car repetedly failed to stay on road and follow the dirt road. Increasing the dropout layer probability seems to have made the algorithm more robost and the car was finally able to complete the track without any problem.

![Car running autonomously on a virtual track][image1]
---
