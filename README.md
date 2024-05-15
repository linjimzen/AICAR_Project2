# **Traffic Sign Recognition** 
[訓練資料可以此獲得](https://github.com/v-thiennp12/Traffic-Sign-Recognition-with-Keras-Tensorflow/tree/main/traffic-signs-data)

> 貢獻者: N96121147_林璟任
---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

Here is a link to my [project code](https://github.com/linjimzen/AICAR_Project2.git)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32,32,3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.
下圖為training set中，每個class的數量，可以觀察到某幾類的數量較少，因此在訓練上預期會出現某幾類的準確率可能較低的狀況
![number of classes](https://hackmd.io/_uploads/S1rkVW-XR.png)

例如此圖為label:16 數量為360張
![image](https://hackmd.io/_uploads/SJAmHW-m0.png)



### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)
首先，先將對輸入的彩色圖像轉灰階，目的是降低參數量(3個channel -> 1個)，並且還能保留圖片的特徵
接者，將對灰階圖使用直方圖均衡化，目的是增強圖像對比度，使得圖像更加清晰，細節更加突出


|原圖|轉灰階|直方圖均衡化|
|--|--|--|
| ![image](https://hackmd.io/_uploads/rJnkdWbmA.png)| ![image](https://hackmd.io/_uploads/HJcTvW-mC.png) |![image](https://hackmd.io/_uploads/HJJ1_Z-Q0.png)|


此外我使用了ImageDataGenerator來對input data做augmentation，通過旋轉、縮放、平移、剪切等，可以生成新的圖像數據，增加訓練的多樣性，以及模型的泛化能力
![image](https://hackmd.io/_uploads/Hya72-Z7R.png)



#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.


My final model consisted of the following layers:

![model structure](https://hackmd.io/_uploads/BkBUTWZX0.png)


| Layer            | Description                                    |
|------------------|------------------------------------------------|
| Input            | 32x32x1 image                                 |
| Convolution 5x5  | 1x1 stride, valid padding, outputs 28x28x32    |
| RELU             | Rectified Linear Unit activation function      |
| Average pooling  | 2x2 stride, valid padding, outputs 14x14x32    |
| Convolution 5x5  | 1x1 stride, valid padding, outputs 10x10x64    |
| RELU             | Rectified Linear Unit activation function      |
| Average pooling  | 2x2 stride, valid padding, outputs 5x5x64      |
| Flatten          | Flattens the input, outputs 1600               |
| Fully connected  | Dense layer with 1024 units, ReLU activation  |
| Fully connected  | Dense layer with 128 units, ReLU activation   |
| Fully connected  | Dense layer with 43 units, Softmax activation |

 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

| Parameter        | Description                                       |
|------------------|---------------------------------------------------|
| Optimizer        | Nadam optimizer with learning rate of 0.001       |
| Loss Function    | Categorical Crossentropy                         |
| Epochs           | 30                                                |
| Batch Size       | 128                                               |



#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of LeNet_model is 0.9756
* validation set accuracy of LeNet_model is 0.9805
* test set accuracy of LeNet_model is 0.9463

![image](https://hackmd.io/_uploads/rkWfrzfQR.png)





If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
模型的架構參考主要是來自[github](https://github.com/v-thiennp12/Traffic-Sign-Recognition-with-Keras-Tensorflow/blob/main/traffic_sign_classifier_LeNet.ipynb)，和老師提供的初始模型相同，他是以LeNet的模型去做變化，因此我初步是按照參考架構的配置去建立模型。

* What were some problems with the initial architecture?
在初始的架構上有一些問題，首先是初始架構的activation function是用Tanh而不是用ReLU，在CNN的架構上應該使用ReLU，讓計算簡單、避免梯度消失問題等等。第二是神經元的個數，在Layer3中他使用的神經元個數並不是常見的2的次方倍。第三是他並沒有在最後加上softmax。
    

* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
調整的部分有二：
在activation function的部分，將tanh改為relu，並且在最後一層Layer加上softmax。
神經元的部分將layer3的神經元分別改為1024和128

* Which parameters were tuned? How were they adjusted and why?
原本參考的BATCH SIZE是64，我改為128，讓每次訓練可以用到更多的數據讓模型訓練

* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?
LeNet是經典的CNN模型，其中卷積層的優勢在於它能夠提取局部的特徵，對於模型處理分類問題是很重要的。
另外雖然我使用的模型架構並沒有使用到dropout layer，但是從訓練過程中可以觀察到，有些許overfitting的狀況發生，如果加上dropout layer可以替模型去除一些不重要的訊息，或許能夠幫助提升accuracy



### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:
![Vehicles over 3.5 metric tons prohibited](https://hackmd.io/_uploads/B1JrIyZXC.png)
![Keep right](https://hackmd.io/_uploads/ByTrIJWQA.png)
![No passing](https://hackmd.io/_uploads/ryaS8yZQA.png)
![Speed limit 80km](https://hackmd.io/_uploads/SJ6rU1-X0.png)
![Turn left ahead](https://hackmd.io/_uploads/rk6H8yWmR.png)

第1和第5張圖可能會比較難做classify，因為在training和validation set中這兩者的數量相對較少，另外，第四張圖可能會因為也有其他類似的速限圖讓辨識能力下降

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

模型預測結果為:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Vehicles over 3.5 metric tons prohibited      		| Vehicles over 3.5 metric tons prohibited  									| 
| Keep right    			| Keep right 										|
| No passing					| No passing											|
| Speed limit 80km/hr	      		| Speed limit 80km/hr					 				|
| Turn left ahead			| Turn left ahead      							|

模型可以成功將5張圖片猜出，accuracy為100%，相較於test set為94.6%有更好的結果，此外，模型也有成功的預測第1和第5張圖，因此可以推測此模型有不錯的成效


#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

top5預測的code cell在倒數第二個


第一張圖可以看到，模型非常確定預測的結果是Vehicles over 3.5 metric tons prohibited，top5 accuracy如下

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1       			| Vehicles over 3.5 metric tons prohibited   	| 
| 1e-20     				| No passing 								|
| 1e-21					| Speed limit (120km/h)							|
| 2e-22	      			| No vehicles					 				|
| 1e-23				    | Speed limit (80km/h)    						|


第二張圖可以看到，模型非常確定預測的結果是Keep right prohibited，top5 accuracy如下

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0      			    | Keep right   						            | 
| 4e-29    				| Priority road 						        |
| 2e-30					| Go straight or right							|
| 7e-31	      			| No passing for vehicles over 3.5 metric tons	|
| 1e-31		            | Turn left ahead    						    |


第三張圖可以看到，模型非常確定預測的結果是No passing，top5 accuracy如下

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.99       			| No passing   						            | 
| 3e-06   				| Vehicles over 3.5 metric tons prohibited 	    |
| 1e-07                 | No passing for vehicles over 3.5 metric tons	|
| 1e-09      			| No vehicles					 		        |
| 28e-11			    | End of no passing    						    |

第四張圖可以看到，模型的預測結果不如其他四張圖，但是預測機率也有0.8，我認為這樣的結果算是合理，因為這張圖預測的其他結果都是類似的圖(都是速限圖)，這本來就是比較接近的特徵，因此這樣的結果我認為算是不錯

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.8       			| Speed limit (80km/h)   						| 
| 0.1    				| Speed limit (100km/h) 						|
| 0.04					| Speed limit (60km/h)							|
| 0.02	      			| Speed limit (120km/h)					 		|
| 0.005			        | Speed limit (50km/h)    						|

第五張圖可以看到，模型非常確定預測的結果是Turn left ahead，top5 accuracy如下

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1       			    | Turn left ahead   					        | 
| 1e-11    				| Keep right 				                    |
| 4e-13					| Ahead only					                |
| 3e-13	      			| Roundabout mandatory			                |
| 2e-13		            | Slippery road    					            |



### 下圖為testing data的confusion matrix，幾個明顯可以看到的預測錯誤如下

![image](https://hackmd.io/_uploads/ryKiTMf7C.png)



1. class 18 and 26
![image](https://hackmd.io/_uploads/BkaPAzMmA.png)
![image](https://hackmd.io/_uploads/BJR_CzG7R.png)

2. class 11 and 30
![image](https://hackmd.io/_uploads/rJMaRMzm0.png)
![image](https://hackmd.io/_uploads/S1Q30ffm0.png)

3. class 2 and 5
![image](https://hackmd.io/_uploads/HJQ117zQ0.png)
![image](https://hackmd.io/_uploads/S13JkmG70.png)


