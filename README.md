#**Behavioral Cloning** 

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/ModelVisualization.png "Model Visualization"
[image2]: ./examples/Center_Lane_Driving.png "Center Lane Driving"
[image3]: ./examples/off_road_start.jpg "Off Road Start"
[image4]: ./examples/off_road_mid.jpg "Off Road Mid"
[image5]: ./examples/off_road_center.jpg "Off Road Center"
[image6]: ./examples/Normal_Image.jpg "Normal Image"
[image7]: ./examples/Flipped_Image.jpg "Flipped Image"
[image8]: ./examples/center.jpg "Center Image"
[image9]: ./examples/left.jpg "Left Image"
[image10]: ./examples/right.jpg "Right Image"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. My project includes the following files to run the simulator in autonomous mode

* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
To record the run, following command can be used.
```sh
python drive.py model.h5
```
Then select autonomous mode.

To record the run, following command can be used.
```sh
python drive.py model.h5 run1
```
where run1 is the name of the folder where all the images of current run will be stored.

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. Model architecture

My model is inspired from NVIDIA's self driving model. The architecutre looks like as follows

Lambda layer: To normalize the pixels to range from -0.5 to 0.5.
Cropping Layer: To remove sky and front hood of the car. This allows only road to be visible.
BatchNormalization: Batch Normalization layer making normalization a part of the model architecture and performing the normalization for each training mini-batch.

Convolution2D: 5x5 kernal, 24 depth with relu activation and stride of 2x2
Convolution2D: 5x5 kernal, 36 depth with relu activation and stride of 2x2
Convolution2D: 5x5 kernal, 48 depth with relu activation and stride of 2x2
Convolution2D: 3x3 kernal, 64 depth with relu activation and stride of 1x1
Convolution2D: 3x3 kernal, 64 depth with relu activation and stride of 1x1
Flatten
Fully Connected Layer: 1164 with relu activation
Fully Connected Layer: 100 with relu activation
Fully Connected Layer: 50 with relu activation
Fully Connected Layer: 10 with relu activation
Fully Connected Layer: 1 with tanh activation

Model's loss is taken as MSE because the label is a real number. 
 
For more details on the architecture, you can look at the NVIDIA's paper on self driving [here] (https://arxiv.org/pdf/1604.07316.pdf)

####2. Attempts to reduce overfitting in the model

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 21-90). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.
For the areas where the model was not doing well, I collected more data on recovery. In general I tried to keep the dataset large to reduce overfitting. 

Also I kept track of training and validation error and kept the epoch low so that overfitting never happens.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 131).

####4. Training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road. 
Also i used left and right images with correction of  0.2 degrees. This helped specially in cased of recovering whenever the car started drifting on sharp turns.

For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to start with a well establish model like NVIDIA and refine it either by changing the model or adding more data.

My first step was to use recreate the NVIDIA pipeline in Keras and then train on the data already provided in the starter kit.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. 
I found that my model had a low mean squared error on both the training and the validation set. This implied that the model was doing okay. 

So i ran the model on simulator and studied the behavior of the car. I noticed the car was able to navigate fine on straight regions but was always going off road on sharp turns.

To improve the driving behavior in these cases, I collected more data on recovering if the road start going to the edge of road. Also I in general collected more data.

To increase the amount of data further increased the number of images, I used left and right images as well with a correction of 0.2 degrees. Also i flipped the images and made the measurements negative.

This gave 6x size to the original data size.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model architecture (model.py lines 114-128) remaind the same because I realized successive improvements in the performance just by adding more data. 

The model looks like ![Model used][image1]


####3. Creation of the Training Set & Training Process

To start with, I used the data provided with the starter code which mostly had center lane driving. Here is an example image of center lane driving:

![Center Lane Driving][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to recover if starts going off road.

These images show what a recovery looks like as follows:

![Off Road Start][image3]
![Off Road Mid Recovery][image4]
![Off Road Recovery][image5]

To augment the data set, I use the left and right images with measurement correction. Theses images look as follows

![Center Image][image8]
![Left Image][image9]
![Right Image][image10]

Further I flipped all images and angles thinking that this would simulate counter clockwise movement. For example, here is an image that has flipped:

![Normal Image][image6]
![Flipped Image][image7]

After the collection process and flipping images, I had 77k number of data points. I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 5 as evidenced by tracking the validation error.

I noticed beyond 5 validation error didnt reduce and some cases increased. So I kept the epochs as 5 for final training.

I used an adam optimizer so that manually training the learning rate wasn't necessary.

I tested the final model for 2 different driving speeds to test the limits. Video run1.mp4 show car driving in autonomous mode at speed 12 and run2.mp4 at speed 9.

###4. Training resources
I tried training on both GPU and CPU. I noticed that it took about 6 hours to train on CPU and about 30 min to 40 min on GPU. 

Also since the data was so huge, loading it all together was difficult. Hence I python generator to yield data in batches and used keras fit_generator function to perform training using generator. 