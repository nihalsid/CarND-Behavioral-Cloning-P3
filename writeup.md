# **Behavioral Cloning** 


The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/center.jpg "Center"
[image2]: ./examples/recovery1.jpg "Recovery 1"
[image3]: ./examples/recovery2.jpg "Recovery 2"
[image4]: ./examples/recovery3.jpg "Recovery 3"
[image5]: ./examples/flip1.jpg "Flip 1"
[image6]: ./examples/flip2.jpg "Flip 2"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed
Model contains a normalizing layer, followed by a cropping layer. This is followed by 5x5 convolutions with 24 filters, maxpool, 5x5 convolutions with 36 filters, maxpool, 5x5 convolutions with 48 filters, maxpool, 3x3 convolutions with 64 filters, 3x3 convolutions with 64 filters, then fully connected layers with 100, 50, 10 and finally 1 output neuron.  

#### 2. Attempts to reduce overfitting in the model

To reduce overfitting early stopping technique was used, where we stopped training when the validation loss was observed to be increasing. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 62).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to reduce underfitting by increasing model depth. 

My first step was to use a convolution neural network model similar to the nvidias self driving network architecture (https://devblogs.nvidia.com/deep-learning-self-driving-cars/). This was suggested in the course and worked pretty well.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model to use less epochs just before it started to overfit, as shown by the loss increase on validation set.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I added more training data to the dataset.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 49-64) consisted of a convolution neural network with the following layers and layer sizes.


#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image1]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image2]

![alt text][image3]

![alt text][image4]

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image5]

![alt text][image6]

After the collection process, I had 26k number of data points. I then preprocessed this data by normalization and mean centering.

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 5 as evidenced by increase in validation loss after epoch 5. I used an adam optimizer so that manually training the learning rate wasn't necessary.
