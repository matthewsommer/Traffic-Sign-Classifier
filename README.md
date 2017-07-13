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

![image1](./data/a.jpg "Ahead Only")![image2](./data/b.jpg "Yield")![image3](./data/c.jpg "30 km/h")![image4](./data/d.jpg "Stop")![image5](./data/e.jpg "General Caution")

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

![Graph of images per label classification in Training data](/images/Classifications_Count.png)

### Design and Test a Model Architecture

I've normalized all the image data to help the model converge and not overfit when training the data and making a prediction of the image. By normalizing the RGB values the data has close to zero mean and standard deviation.

To help train I also normalized the counts of each sign image. Some of the classifications did not have enough data to train the model. I augmented the underrepresented signs by randomly copying one of it's instances and doing a random transformation to it. I wanted to pick a random instance of a sign so as not to have too much duplicate data which wouldn't help train the model.

To transform the images I'm using just two methods, scaling and rotation. I'm able to get a good accuracy with these transformation so I didn't add any more. If i wanted to improve this I would add a brightness or contrast transformation. I also might use this brightness/contrast transformation on some of the training images as they are very dark.

I did not transform any of the original data (other than normalization) as I thought it best to leave it as it was and only augument the data with transformed versions of the original data.

### Neural Net Model Architecture

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
 


### Method to Find Solution and Training Parameters

When I began this project I started with the LeNet CNN architecture as it seemed to apply well to this problem since it worked well with the letter classification lab in the lecture and lab. Given that I'm new to Convolutional Neural Networks I didn't want to try an architecture that was too foriegn to me as I needed to learn the overall methods used predict what is shown in an image. Also the LeNet architecture is well documented and used so there are plenty of resources on the internet to help me get going. If I had more time I would switch to a more modern architecture but the LeNet works well for this project. Initially I had an accuracy around 87%. My first attempt at increasing the accuracy I added dropout but that didn't increase the accuracy (actually it made it worse but I realize now that I think I had a bug that caused this). Then I started trying to play with the parameters to increase the accuracy but that didn't help much. I realized that I would not get good results with the data that I had because some of the labels did not have enough data compared to others, so I decided to create some more data. The biggest increase in accuracy came when I added more augmented data. When I just duplicated the data without transforming it, the accuracy didn't go up (which makes sense because more of the same data doesn't really help train the network). I tried adjusting the learning rate up and down but found that 0.0001 worked well to get the accuracy up.

To train the model I used a batch size of 128, 20 epochs, and the AdamOptimizer. My learning rate is 0.001.

### Training Results

My final model results were:
* training set accuracy of 98.4%
* validation set accuracy of 95.2%
* test set accuracy of 93.1%
 
### Testing the Model on New Images

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Ahead Only      		| Ahead Only   									| 
| Yield     			| Yield 										|
| 30 km/h				| 30 km/h										|
| Stop  	      		| Stop      					 				|
| General Caution		| General Caution      							|


The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. This is promising results but the data set is so small that isn't that meaningful. The speed limit sign was probably the hardest to determine given how similar it is to the other speed limit signs (as evidenced by softmax below).

## Softmax Predictions for each image

For the first image, the model is very certain that the sign is Ahead only, which it is. The ahead only sign might be hard to classify because there are other similar signs for left and right turns as well as go straight or left/right. The sign should be easy to classify thoug as the arrows are very distinctly different from each other. The top five soft max probabilities are:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00         			| Ahead Only (35)								| 
| .00     				| Turn Right Ahead (33)							|
| .00					| Go Straight or Right (36)						|
| .00	      			| Turn Left Ahead (34)			 				|
| .00				    | Yield (13)                					|

For the second image the model is predicting Yield which it is. This sign might be hard to classify as it's very similar to General Caution and a couple other signs with a triangle and similar coloring. I think it won't be hard for the classifier though because the center is a solid color with no graphic. The top five soft max probabilities are:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00         			| Yield (13)   									| 
| .00     				| Priority road (12)							|
| .00					| No Vehicles (15)								|
| .00	      			| Stop (14)     				 				|
| .00				    | 20 km/h (0)        							|

For the third image the model is predicting 30 km/h which it is. This sign might be difficult to classify because it's one of the speed limit signs so the classifier will have to identify which speed is being shown. The evidence of this is shown in the softmax probablities because the other potential classifications are the other speed limit signs. The top five soft max probabilities are:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .99         			| 30 km/h (1)   								| 
| .00     				| 50 km/h (2)   								|
| .00					| 20 km/h (0)									|
| .00	      			| 80 km/h (5)					 				|
| .00				    | 70 km/h (4)      		    					|

For the fourth image the model is predicting Stop which it is. The Stop sign should be the easiest to classify. I'm not sure why one of the probablities is Bicycyles crossing but I can understand No Entry as it's a solid red color. Though the probablities are basically zero so the model is confidently predicting the stop sign. The top five soft max probabilities are:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .99         			| Stop (14)    									| 
| .001     				| Bicycles Crossing (29)						|
| .00					| No Entry (17)									|
| .00	      			| Yield (13)	        		 				|
| .00				    | Right-of-way at the next intersection (11) 	|

For the fifth image the model is predicting General Caustion which it is. I thought this sign might be the hardest to classify as there are other triangle shaped signs with similar coloring and a graphic in the center, but the network didn't have much trouble classifying the image as it's very clear. The top five soft max probabilities are:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .99         			| General Caustion (18) 						| 
| .00     				| Pedestrians (27) 								|
| .00					| Traffic Signals (26)							|
| .00	      			| Right-of-way at the next intersection (11) 	|
| .00				    | Road narrows on the right (24)				|