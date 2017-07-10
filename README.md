## Project: Build a Traffic Sign Recognition Program
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

Overview
---
In this project I have built a CNN network based on the LeNet architecture.
---
**Project Goals**

* Load the data set (zipped in data folder in repo)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

### Dependencies
I used the starter kit provided by Udacity:

* [CarND Term1 Starter Kit](https://github.com/udacity/CarND-Term1-Starter-Kit)

The lab environment can be created with CarND Term1 Starter Kit. Click [here](https://github.com/udacity/CarND-Term1-Starter-Kit/blob/master/README.md) for the details.

### Example Images

Here are a few examples images of German traffic signs.

[//]: # (Image References)

[image1]: data/a.jpg "Ahead Only"
[image2]: ./data/b.jpg "Yield"
[image3]: ./data/c.jpg "30 km/h"
[image4]: ./data/d.jpg "Stop"
[image5]: ./data/e.jpg "General Caution"

---
### Source Code

Here is a link to my [jupyter project code](https://github.com/matthewsommer/Traffic-Sign-Classifier/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

### Exploratory visualization of the dataset.

Here is a chart of number of images per label in the training data. This illustrates that there are underrepresented labels. It would be best to add more data for these labels.

![Graph of images per label classification in Training data][./images/Classifications_Count.png]

##Design and Test a Model Architecture

####1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

I've normalized all the image data to help the model converge and not overfit when training the data and making a prediction of the image. By normalizing the RGB values the data has close to zero mean and standard deviation.

To help train I also normalized the counts of each sign image. Some of the classifications did not have enough data to train the model. I augmented the underrepresented signs by randomly copying one of it's instances and doing a random transformation to it. I wanted to pick a random instance of a sign so as not to have too much duplicate data which wouldn't help train the model.

To transform the images I'm using just two methods, scaling and rotation. I'm able to get a good accuracy with these transformation so I didn't add any more. If i wanted to improve this I would add a brightness or contrast transformation. I also might use this brightness/contrast transformation on some of the training images as they are very dark.

I did not transform any of the original data (other than normalization) as I thought it best to leave it as it was and only augument the data with transformed versions of the original data.


##Neural Net Model Architecture

My final model is the LeNet model with dropouts added.

| Layer or Operation    |     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution        	| 1x1 stride, VALID padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride, VALID padding, outputs 14x14x6	|
| Convolution   	    | 1x1 stride, VALID padding, outputs 10x10x16   |
| RELU and Dropout		|		        								|
| Max pooling	      	| 2x2 stride, VALID padding, outputs 5x5x16	    |
| Flatten   	      	| Outputs 400	                                |
| Fully connected		| Output 120        							|
| RELU and Dropout		|		        								|
| Fully connected		| Output 84        		    					|
| RELU and Dropout		|		        								|
| Fully connected		| Output 43        		    					|
 


####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model I used a batch size of 128, 20 epochs, and the AdamOptimizer. My learning rate is 0.001.

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of ?
* validation set accuracy of ? 
* test set accuracy of ?

I started with the LeNet architecture and added dropout. If I had more time I would switch to a more modern architecture but the LeNet works well for this project. When I started I realized that I would not get good results with the data that I had because some of the labels did not have enough data compared to others, so I decided to create some more data. I tried adjusting the learning rate up and down but found that 0.0001 worked well to get the accuracy up.
 
###Test a Model on New Images

Here are five German traffic signs that I found on the web:

![alt text][./data/a.jpg] ![alt text][./data/b.jpg] ![alt text][./data/c.jpg] 
![alt text][./data/d.jpg] ![alt text][./data/e.jpg]

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Ahead Only      		| Ahead Only   									| 
| Yield     			| Yield 										|
| 30 km/h				| 30 km/h										|
| Stop  	      		| Stop      					 				|
| General Caution		| General Caution      							|


The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. This is promising results but the data set is so small that isn't that meaningful.

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ... 
