# **Behavioral Cloning** 

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: model.png "Model Visualization"
[image2]: history.png "Training accuracy"
[video1]: run1.mp4 "Autonomous drive video"

# Files Submitted & Code Quality

## Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* run.py: the main entry point for controlling the training and testing process
* config.py: processing command line arguments
* utils.py: containing utilities functions
* train.py: containing the code for training and testing the model
* model.py: containing the code for creating, running training/testing, and saving the model
* drive.py: for driving the car in autonomous mode
* model.h5: containing a trained convolution neural network 
* writeup_report.md: summarizing the results

All major functions in the code provide docstring and comments.

## Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

# Usage Guide

Training and testing can be launched with the following command:

```
python run.py --dirs folder_list --checkpoint checkpoint_name --batch batch_size --accept threshold --epochs epochs
--all_cameras boolean --flip boolean
```
* --checkpoint: Path and base name of checkpoints to to saved for epochs that meet the acceptance threshold
* --dirs: training sample folders, seperated by :, default is data folder
* --test: testing sample folder, default is None
* --model: the saved checkopint to load to continue/transfer learning, default is None
* --trainings: The number of independent trainings to perform, default is 1
* --epochs: the number of epochs per train. Default is 200
* --batch: the batch size, default is 256
* --lr: the learning rate. Default is None, and will be inferred automatically depending on the batch size
* --drr: the dropout retention ratio, default is 0.5
* --accept: the accepted training validation accuracy. Default is 0.994
* --cr: the left/right cammera correction, default is 0.2
* --all_cameras: True to use three cameras, False to use the center camera. Default is False
* --flip: True to include flipped images, Default is True
* --cont: True to continue from the trained model, False to train from weights, default is False

Furthermore, the code can take the following command from the keyboard during the course of training and testing:

1. *accept*: change the accepted accuracy, e.g. *accept 0.994*
2. *save*: save currently trained model when the current epoch is completed
3. *train*: start training if it has not yet been started
4. *test*: start test against all saved models in the desginated checkpoint folders
5. *stop*: stop training after the current epoch
4. *exit*: exit the program


# Model Architecture

## The model architecture

My model consists of a convolution neural network based on Nvidia Self-Driving Car model as follows:

![image1]

| Layer         		|     Description	        					        | 
|:---------------------:|:-----------------------------------------------------:| 
| Input         		| 3*160*320 RGB image   					            | 
| Cropping2D         	| Crop input image to 3*90*320              	        | 
| Normalization2D       | Custom layer to normalize image                       | 
| Convolution 5x5     	| 2x2 stride, valid padding, filters 24  	            |
| Leaky RELU			|												        |
| Convolution 5x5     	| 2x2 stride, valid padding, filters 36                 |
| Leaky RELU			|												        |
| Convolution 5x5     	| 2x2 stride, valid padding, filters 48     	        |
| Leaky RELU			|												        |
| Convolution 3x3     	| 1x1 stride, valid padding, filters 64      	        |
| Leaky RELU			|												        |
| Convolution 3x3     	| 1x1 stride, valid padding, filters 64      	        |
| Leaky RELU			|												        |
| Fully connected		| Activation elu, units 100             		        |
| Dropout       		| keep probability: 50%    		                        |
| Fully connected		| Activation elu, units 100             		        |
| Dropout       		| keep probability: 50%    		                        |
| Fully connected		| Activation elu, units 100             		        |
| Dropout       		| keep probability: 50%    		                        |
| Fully connected		| Activation elu, units 100             		        |

The normalization is performed by a Keras custom layer as Lambda layer on Windows 10 cannot be saved due
to some unicode problem. All convolution layers used Leaky ReLu for activation, and all the fully connected
network layer uses elu for activation.

## Model Architecture Strategy

The mode architecture is based on Nvidia Self-Driving Car CNN architecture with the following differences:

* The model uses Leaky ReLu in the convolution layers as it has shown to be able to produce better performance
than LeRu in my experiments
* Elu is used in the model for the same reason comparing to ReLu and tanh
* To reduce overfitting, dropout is used in the fully connected network layers


## Training Strategy

Since the trainig process is very time-consuming, I want to be able to save training checkpoint at any time, stop
the training process, test it with the simulator, and resume the training with existing samples for better accuracy
or with new samples for improving handling of driving conditions , I also have im plemented extra functionalities
in the program as described in the **Usage Guide** section.

The training process has the following characteristics:
* The training samples are splitted into 90% training set and 10% validation set
* A seperate test set can be specified to ensure that it can be completely hidden during the entire training process 
* Adam optimizer is used for training for faster training and improved accuracy
* Mean squared error is used as the loss function However, due to time constraints, other loss functions like mean
absolute error has not yet been tried.

In addition, it uses the following callbacks during the training:

* ReduceLROnPlateau: to reduce learning rate when training stopped improving, this has improved the overally training
accuracy
* TernsorBoard: to produce logs for TensorBoard analysis
* End of Epoch: to save trained model to checkpoint at the end of an epoch if the training accuracy is above the threshold

## Data Collection

It was initially challenging to navigate the car using mouse and keyboard for collect training data using. Eventually,
I collected one set of center lane driving for two laps, one set of center lane driving in the opposite direction, and
few set of data for corrected driving to teach the car to avoid going off the road. In addition to that, I also used
the data provided by the course to increase the data set. The actually training included both the original and flipped
data. Only center lane images were used, and using left/right images did not show evidence of improvement.

## Training

The csv of the data sets stored in different folders during the training were processed first to yield two list, one for
image names, and another for the directions. The lists were then shuffled randomly, and then splited into the training
set containing 90% of data, and the validation set containing 10% of the data.

Two training data generators were used during data, one for training data, and another for validation data. The generators
shuffle the data first, then generate batches of training data by reading images specified in the corresponding image name
list into an image list.

The final training consists of 77,101 training samples, and 8,567 validation samples.

The network were able to accomplish over 97% validation accuracy at the initial epoch, and trained models were saved after
reaching 99.4% validation accuracy.

The initial training contains only the center lane driving samples, and the car went off the road quickly. Corrected driving
samples were progressively added in seperated trainings until results were acceptable. The followings were observed:

* If we continue training using new samples, the effects of the new sample may be very small, due to learning rate decay
strategy that will result in very small learning rate.
* The best results were obtained using the trained weights with a learning rate the is close to the initial learning rate.
In order to avoid the new samples to become dominate the resulting network, a good portions of orioginal training samples
were also included.
* The correction training did not use as many epochs as the originl training, but the result of 'correction' was evident.

## Results

A typical training and validation loss obtained in one of the trainings is show in the following figure. The validation set has lower loss than the training indicating there is no overfitting issue:

![image2]

The test result of the training is demonstrated in the following video:

![video1]
