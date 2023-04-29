# Computer Vision in Remote Sensing


```
{

**Convolutional neural networks (CNN) for Tree Classification & Mapping**

Convolutional neural networks (CNN) is a network architecture deep learning algorithm, that works by automatically learning features 
from the input images, allowing them to achieve high accuracy in complex task. CNN is made up of several layers that process and 
transform an input to produce an output. CNN is commonly used for image classification, object detection, and segmentation tasks.

There are three key concepts in CNNs : local receptive fields, shared weights and biases, and activation and pooling. Using these 
three concepts, we can configure the layers in a CNN. CNN can have tens or hundreds of hidden layers that each learn to detect different 
features in an image. Thus, a very hidden layer increases the complexity of the learned image features. The first hidden layer learns 
how to detect edges, and the last learns how to detect more complex shapes. The final layer/final output  connects every neuron, from 
the last hidden layer to the output neurons (The MathWorks, Inc. 2017).

There are three ways to use CNN for image analysis. The first method is to train CNN from scratch. This method is the most accurate and 
the most challenging. The second method relies on transfer learning, which is based on the idea that you can use knowledge of one type of 
problem to solve a similar problem. For example, a CNN model that has been trained to recognize animals to initialize and train a new model that differentiates between cars and trucks.  The third method, you can use a pre-trained CNN to extract features for training a machine learning model. For example, this can involve a hidden layer that has learned how to detect edges in an image is broadly relevant to images from many different domains. This method requires the least amount of data and computational resources vs the first two methods (The MathWorks, Inc. 2017).

CNN & Hyperspectral Imagery:

In “A Convolutional Neural Network Classifier Identifies Tree Species in Mixed-Conifer Forest from Hyperspectral Imagery” by Fricker et al 2019, focus on the automation of tree species classification and mapping by using CNN, field-based training data, high spatial resolution airborne hyperspectral imagery.
The methodology workflow includes three inputs dataset, Hyperspectral imagery RGB imager.  Field data is used to produce Imagery label chips for each dataset, each data is set to a Hypermeter tuning training, testing and validation and then the CNN algorithm is used to execute Species Based Prediction of 8 classes for each dataset and then exported as a Geo tiff output which will be ran by a classification Accuracy Report. Overall, the study seeks to evaluate the application of CNNs to identify and classify tree species in hyperspectral imagery in comparison to RGB subset. Results  show an average classification F-scores for all species was 0.87 for the hyperspectral CNN model and 0.64 for the RGB model (Fricker et al 2019). Overall,  indicating that deep learning for image classification and tree species identification using high resolution hyperspectral image data improved classification accuracy in comparison to broad-band RGB imagery (Fricker et al 2019).

CNN & RGB Orthoimages: 

In the research article “Convolutional Neural Networks  Enable Efficient, Accurate and Fne-grained Segmentation of Plant Species and Communities from High-Resolution UAV Imagery” by Kettenborn, et al 2019 focus on plant species identification and mapping using deep learning, convolutional neural networks (CNN).  The applied workflow  involves producing high-resolution orthoimages, and digital elevation models (DEM) that were derived from UAV-based RGB aerial imagery-photogrammetric processing. The orthoimages were used to create reference data for each target class by visual interpretation. The reference data, orthoimage and DEMs was used to train the CNN-based segmentation of each target class - plant species using the U-net architecture. Then the trained CNN models are applied to the photogrammetric data using image tiles extracted by a regular grid. The methodology is based on using a CNN-based segmentation approach (U-net) in combination with training data that was derived from visual interpretation of UAV based high-resolution RGB imagery. The resulting segmentation  output is merged to produce a spatially continuous map of the target class. Result  demonstrate that this approach can accurately segment and maps vegetation species and  communities with at least 84% accuracy (Kettenborn, et al 2019).

CNN & Lidar Point Cloud: 

In  “PointCNN-Based Individual Tree Detection Using LiDAR Point Clouds” by Ying et al. 2021,  present  a PointCNN-based method of 3D tree detection using LiDAR point clouds, the research aims to address and improve the detection accuracy of tree in a complex scene/high density region containing a diverse of tree species. The applied workflow first builds a canopy height model CHM using raw LiDAR point clouds. Then obtains the maximum value of the local height on the CHM as the seed point -the detection samples. The 3D-CNN classifier is based on PointCNN which is used to classify detection samples, and the classification results are used for filtering rough seed points and performs the tree stagger filter rough seed points as the first screening. In the final stage a stagger analysis of seed points was performed to determine whether two close seed points belong to the same tree as the second screening. Concluding that the seed points remaining after two filters were taken as the locations of the detected trees. Results of the study show the highest matching score and average score reached 91.0 and 88.3. Thus, indicating that  methodology can effectively extract tree information in complex scenes ( Ying et al. 2021).

References: 

Fricker, Geoffrey A., Jonathan D. Ventura, Jeffrey A. Wolf, Malcolm P. North, Frank W. Davis, and Janet Franklin. 2019. "A Convolutional Neural Network Classifier Identifies Tree Species in Mixed-Conifer Forest from Hyperspectral Imagery" Remote Sensing 11, no. 19: 2326. https://doi.org/10.3390/rs11192326

Kattenborn, Teja, Jana Eichel, and Fabian Ewald Fassnacht. "Convolutional Neural Networks enable efficient, accurate and fine-grained segmentation of plant species and communities from high-resolution UAV imagery." Scientific reports 9, no. 1 (2019): 1-9.

The MathWorks, Inc. Introduction to Deep Learning: What Are Convolutional Neural Networks? Recorded: 24 Mar 2017 https://www.mathworks.com/videos/introduction-to-deep-learning-what-are-convolutional-neural-networks--1489512765771.html/

Ying, W., Dong, T., Ding, Z., Zhang, X. (2021). PointCNN-Based Individual Tree Detection Using LiDAR Point Clouds. In: , et al. Advances in Computer Graphics. CGI 2021. Lecture Notes in Computer Science(), vol 13002. Springer, Cham. https://doi.org/10.1007/978-3-030-89029-2_7


}
```
