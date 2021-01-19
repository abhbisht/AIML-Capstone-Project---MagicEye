# AIML-Capstone-Project---MagicEye
Multilabel Image Classification model to help to Identify Indoor Objects for Visually Impaired Person
Implementation:

To train the model to identify the objects we have followed below steps:

Dataset Preparation and Image Labeling: Preparing a Dataset is the first step for any Machine learning model. As the project is to identify indoor objects we prepared datasets by capturing images of indoor objects around our surroundings. We have captured multiple images of the same object from different directions, multiple mobile cameras under different lighting conditions. 

To make the model more generalized and identify variations we have also used images from open source online repositories. 

We have prepared a larger dataset of around 2800 images, out of these 1000 images are captured by using mobile phone

We have labeled each image with an object label, in this project we took a total of 8 indoor objects which includes Bed, Door, Refrigerator, Stairs, Chair, Closet, Table, Flowerpot. If an image has multiple labels we label that image with multiple labels.

We have created a csv file having image name, image path and labels.


Data Preprocessing:

Label Encoding: CNN classification model requires output labels in one-hot encoding format. As our output label is object name, and each image can have multiple objects of different class, we used MultiLabelBinarizer to encode our labels to one hot encoding format. Multilabelbinarizer allows us to encode multiple labels per instance. 
 
The output of Multilabelbinaries is in one-hot encoded format, where number of output features is equal to number of total object class. If an object has multiple images, the corresponding output features value is 1 for that image.

Batch Generator:

We have used batch generator function to generate and feed our CNN model with random training    and validation datasets. Batch generator provides a random batch of 64 augmented images dataset. 

We have used batch size of 32 and 64 images also helps to reduce computing resources required to process all images at once. 

Batch generator function pick random images from train and test datasets, converts image in to array, perform image scaling, image normalization and Image augmentation using Keras.ImageDataGenerator 

Image Scaling:

Prior to feeding images to the model we need to define the standard image size we model can accept as input. As input images can be of any size, we are resizing the image to a standard size which model can accept. 

We have tried multiple image sizes to check the accuracy of the model on different image sizes like 224 x 224, 300 x 300, 600x 600. 

Finally we have selected image size as 224 x 224 ( height, width), which is also the standard input size of most of pretrained models




Image Normalization:

As we know Linear Models such as Linear Regression, Logistic Regression & Neural Networks including Convolution Networks where a dependent variable is a function of independent Variables.
y=f(x), where f(x) will generalise well on train and test data when values of all independent variables are on the same scale which neglects the effect of higher coefficients of independent variables with higher magnitude values.


And to achieve the same we have various standardization and normalization techniques but in case of Image Data where each pixel value is a feature and could range between [0-255], it’s ideal for Convolutional neural networks to have normalized values in range [0-1] either by explicitly doing the normalization by dividing each pixel with 255 OR if we are using a pre-trained CNN State of the Art Model , we can use preprocess input function Available in keras Library  to take care of preprocessing before feeding the image to convolution layer. In our case , we are using a preprocess input function which helps in faster computation and Avoids Exploding Gradients Problems.

Image Augmentation:

There is a saying for the Machine Learning Models : No Free Lunch which means no one model works best for Every possible situation.

Deep learning models become quickly overfit with small datasets and require a lot of data for training. To overcome the problem we have used Image augmentation and transfer learning in our model.

Image Augmentation and Transfer Learning are two techniques available to overcome this problem, which is a kind of free lunch available to implement every time for image related solutions, it could be classification , object detection and segmentation problems.

Augmentation is the technique where we play with the pixels of  Original Image and generate new Images out of it such as RandomCrop, Brightness & Contrast , Flip, Cutout & Many More.
Convolution Neural Networks are data hungry just like Deep Neural Networks where more data we feed to the Networks the better optimal solution we get and our model generalizes well on Unseen data.


Image augmentation using batch generator helped to avoid overfitting in our model. 




Transfer Learning:

Transfer Learning is something which makes our life easier where some of the pre-designed and pre-available CNN Architecture in Keras Library achieve state of the art results on Imagenet dataset which consists of 1000 classes. Some of the Pre-trained models available to start with image related problems are - AlexNet, VGG16/19, ResNet50/101/152 & Google Inception V3.

Indoor objects which we are trying to classify are part of Imagenet Dataset. So little fine tuning of those pre-trained weights could help us achieve optimal solution but our dataset is noisy where images are taken from Real Environment using cameras of different resolution. Hence excited to see how our model will perform on the noisy data for the model which are already trained on high resolution images of the same DataSet (ImageNet).


Classes Imbalance:

It is  important to maintain the same proportions of data for all the classes in Validation DataSet so that we are more confident to tell about our model performance on unseen data from different classes. To Avoid  cases where a Convolutional Neural Networks learns more about the Majority classes and less about Minority Classes, it is always a good practice to use a Stratified Train-Test Data Split strategy.

Iterative-stratification is a project that provides scikit-learn compatible cross validators with stratification for multilabel data.
Presently scikit-learn provides several cross validators with stratification. However, these cross validators do not offer the ability to stratify multilabel data. This iterative-stratification project offers implementations of MultilabelStratifiedKFold, MultilabelRepeatedStratifiedKFold, and MultilabelStratifiedShuffleSplit with a base algorithm for stratifying multilabel data described in the following paper:
Sechidis K., Tsoumakas G., Vlahavas I. (2011) On the Stratification of Multi-Label Data. In: Gunopulos D., Hofmann T., Malerba D., Vazirgiannis M. (eds) Machine Learning and Knowledge Discovery in Databases. ECML PKDD 2011. Lecture Notes in Computer Science, vol 6913. Springer, Berlin, Heidelberg.
Model Building: 

As the objective of our project is to identify the object in real time and should be light enough to run on mobile devices. As the objects we are identifying are also matching with Imagenet dataset and our datasets size is small, we prefer to use pre-trained light weight models like Mobilenetv2, Resnet50v2, VGG16, EfficientNetB0, Groundup model.

We have trained and tested different models, changed hyper parameters to fine tune to achieve maximum and consistent validation accuracy. The table below show different experiments and changes we have tried to choose the best model. 

Models	Feature Extractor	Classifier with 8 Outputs (to classify each object in output)
1	VGG16	All layers non-trainable, Removed Dense layer	Sigmoid
2	ResNetV2	All layers non-trainable, Removed Dense layer	Sigmoid
3	ResNetV2	All layers trainable, Removed Dense layer	Sigmoid
4	EfficientNetB0	All layers non-trainable, Removed Dense layer	Sigmoid
5	MobilenetV2	All layers non-trainable, Removed Dense layer	Sigmoid
6	MobilenetV2	All layers trainable, Removed Dense layer	Sigmoid
7	MobilentV2	Last 10 layers trainable, Removed Dense layer	Sigmoid
8	MobilenetV2	First 50 layers frozen, Removed Dense layer	Sigmoid
9	Ground Up Architecture	8 layers, Alexnet Architecture	Sigmoid

As we need to identify multiple objects in an image, we use sigmoid function in output to give 8 unique output each with probability score of presence of the each object.
