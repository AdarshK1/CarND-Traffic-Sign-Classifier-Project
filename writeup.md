# **Traffic Sign Recognition** 

## Adarsh Kulkarni


**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./images/histogram_labels.png "Visualization"
[image2]: ./images/histogram_labels_augmentation.png "Visualization after Augmentation"
[image3]: ./images/sample_gray.png "Sample Gray Image"
[image4]: ./images/ahead_only_resized.jpg "Traffic Sign 1"
[image5]: ./images/beware_ice_snow_resized.jpg "Traffic Sign 2"
[image6]: ./images/double_curve_resized.jpg "Traffic Sign 3"
[image7]: ./images/mandatory_roundabout_resized.jpg "Traffic Sign 4"
[image8]: ./images/priority_road_resized.jpg "Traffic Sign 5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/AdarshK1/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of original training set is 34799
* The size of original validation set is 4410
* The size of original test set is 12630
* The size of augmented training set is 70509
* The size of augmented validation set is 7835
* The size of augmented test set is 13826
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43 

#### 2. Include an exploratory visualization of the dataset.

Here are two exploratory visualizations of the data set. The first one, shown below, is of the distribution of labels in the starting dataset.

![alt text][image1]

This second visualization shows the distribution after upweighting the classes with lower frequency

![alt text][image2]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.

My preprocessing pipeline had two main steps:
1. I turned all of the images to grayscale, in order to remove distracting information from the classifier
2. I then normalized all the images to have a mean of 0 and a range from -1 to 1

An example of a grayscaled image. It wouldn't make much sense to display a normalized image since the PNG would look all black!

![alt text][image3]

To add more data to the the data set, I used a very simple technique. I decided that it was most important to increase the frequency of under-represented labels in the training set. Fundamentally, the network needs to have a critical mass of data to learn from for each label. Additionally, the data went through a small transformation (randomly chosen as a rotation, shift or flip) before being added back into the training set.


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:|
| Input         		| 32x32x3 RGB image   							|
| Convolution 5x5     	| Stride: (1,1), Padding: Same, Output 32x32x16 |
| Relu					| Activation Function							|
| Bath Normalization    | Normalize Across Batch                        |
| Max pooling 2x2	    | Stride: (2,2), Padding: Same, Output 16x16x16 |
| Convolution 3x3     	| Stride: (1,1), Padding: Same, Output 16x16x32 |
| Relu					| Activation Function							|
| Bath Normalization    | Normalize Across Batch                        |
| Max pooling	      	| Stride: (2,2), Padding: Same, Output 8x8x32   |
| Flatten        		| Flatten, Output: 1x2048        				|
| Dropout               | Regularization, rate: 0.5                     |
| Fully connected		| Output: 512        							| 
| Relu  				| Activation Function         					|
| Dropout               | Regularization, rate: 0.5                     |
| Fully connected		| Output: 128        							|
| Relu  				| Activation Function         					|
| Dropout               | Regularization, rate: 0.5                     |
| Fully connected		| Output: n_classes (43)        				|
| Relu  				| Activation Function         					|
 

#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

I have a personal workstation equipped with an Nvidia Titan Xp that I used to train the model. I used the following hyperparamters:
- Optimizer: Adam
- Learning Rate: 0.001
- Batch Size: 256
- Epochs: 500

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 0.976
* validation set accuracy of 0.964
* test set accuracy of 0.959

I used an Iterative approach to finding a good model, so I will detail the answers to the iterative approach questions below.

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
  * I started with a version of Yann LeCunn's network, which is a convolutional network followed by dense layers.
  * I made some modifications right off the bat though, replacing the subsamples with maxpools.
* What were some problems with the initial architecture?
  * The network was fairly mediocre; after >100 epochs training it had plateaued in the mid 70's
* How was the architecture adjusted and why was it adjusted?
  * I tweaked the convolution sizes quite a bit, before settling first on a (5,5) and then a (3,3)
  * I tweaked the number of filters quite a bit before settling on 16 for the first convolution and 32 for the second one.
  * I added two batch normalization steps after each set of convolution and Relu.
  * I added three sets of dropout at a rate of 0.5 after the flatten layer, first dense layer, and second dense layer, since the model started to overfit on the training data
* Which parameters were tuned? How were they adjusted and why?
 * As mentioned above, I tweaked the filter sizes and number of filters in both convolutional layers. This was mostly a trial and error process of trying to find a better solution, since at first the model was really not doing well.
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?
  * For image classification problems, and in fact any kind of problem that uses images, convolutions have been shown as the most efficient and effective way to learn information and characteristics. This extends from our simple classification case all the way to semantic segmentation and object detection.
  * Also, as mentioned in the architecture answer, dropout layers were added to help the model stop overfitting on the training data, making it less likely to overweight any one given node.

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are the five German traffic signs that I found on the web:

Note that before these images were inferred on, the appropriate resizing and preprocessing steps were done.

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

Some notes on why I chose these particular images:
- Image 1: I chose this image because it is at an angle and has that watermark. I was curious to see if the watermark would through the network off
- Image 2: I was curious to see if the buil-up of actual ice and snow (which one can imagine is a very real problem) would through off the network. As it happens, it did.
- Image 3: This sign looks similar to the slippery sign, so I was curious if it could make the distinction
- Image 4: The shadow in this picture was meant to be a callenge
- Image 5: This is also a darker and angled picture.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Priority Road      	| Priority Road                                 | 
| Ahead Only   			| Ahead Only                                    |
| Beware Ice/Snow		| Roadwork										|
| Double Curve	      	| End of No Passing					 			|
| Mandatory Roundabout	| Mandatory Roundabout                          |


The model was able to correctly guess 3 of the 5 traffic signs, which gives an accuracy of 60%. This is a lot lower than the ~90% on the test set seen above. However, I choose this pictures specifically as edge cases, so 3/5 correct is within the scope of reason.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability.

The code for making predictions on my final model is located in the 15-19th cells of the Ipython notebook.

For the first image, the model is very sure that the sign is a priority road, which is correct.

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 99.9%         		| Priority Road   								| 
| 0.07%     			| Roundabout Mandatory 							|
| 0.01%	 				| End of all speed and passing limits			|
| 0.007%	      		| Keep right					 				|
| 0.0005%				| End of no passing by vehicles over 3.5 metric tons|

For the second image, the model is very confident that the sign is ahead only, which is also correct

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 99.9%         		| Ahead Only   									| 
| 0.002%     			| Keep right 									|
| 0.0008%				| Yield											|
| 0.0006%	      		| Turn left ahead					 			|
| 0.00002%				| Go straight or left      						|

For the third image, the model is again rather sure that the sign is Road work, but this turns out to be wrong. The only reasonable explanation to this is that the occlusion of features by the snow itself led to the mis-classification. That being said, the model is slightly more unsure than it has been in the previous cases, but again not by much.

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 99.7%         		| Road Work   									| 
| 0.2%     				| Bumpy Road 									|
| 0.04%					| Priority Road									|
| 0.03%	      			| Go straight or left			 				|
| 0.01%				    | Roundabout mandatory     						|

For the fourth image, the model is in fact much less sure of its answer, which ended up being incorrect.

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 79.1%         		| End of no passing   							| 
| 11.0%     			| Slippery road 								|
| 9.63%					| Children Crossing								|
| 0.11%	      			| Dangerous Curve				 				|
| 0.005%				| Vehicles over 3.5 metric tons prohibited      |

For the fifth and final image, the model is pretty sure, although not as much as the first two, of it's correct answer.

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Roundabout Mandatory   						| 
| .20     				| End of all speed and passing limits 			|
| .05					| Speed limit (20km/h)							|
| .04	      			| Speed limit (30km/h)					 		|
| .01				    | Speed limit (50km/h)      					|
