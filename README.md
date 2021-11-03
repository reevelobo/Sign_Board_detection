# Sign_Board_detection
**Pre-Processing And Modelling** <br />
The first thing we had to do is to resize all the images to 32x32x3 and read them into a numpy array as training features. At the same time, we created another numpy array with labels of each image. We needed to do the same for testing images. However the labels for testing images are stored as ClassId in test.csv with paths of that image. So I use pandas to read the csv file, load the image from the path and assign the corresponding ClassId. Here is a comparison between an RGB and grayscale image. The grayscale image still retains its features and can be recognized but with much smaller size.  From the training set, I randomly spitted 20% as a validation set for use during the process of model training. The model accuracy of training and validation will give us information about underfitting or overfitting. 

![detection_p2](https://user-images.githubusercontent.com/41796498/140182357-0054868c-71f5-4d86-9ecf-8aee2544a1ad.png)


The model used for traffic sign classification is a deep neural network model, more exactly a convolutional neural network (CNN), the reason we preferred a CNN is that they have very good results in image recognition, and also, we don't have to worry about filters and weights initialization as they are randomly initialized using gaussian distribution and then fine-tuned using gradient descend. The main advantage of a CNN is the convolution operation. The model scans the input image many times to find certain features. This scanning which is the convolution operation can be set with 2 main parameters: stride and padding type. After the first convolution step we end up with a set of new frames, which can be found in the second layer. Each frame contains information about one feature and its presence in the input image. The resulting frame will have larger values in places where a feature is strongly visible and lower values where the features are low. The model architecture is inspired from Yann Le Cun paper on classification of traffic signs. With the help of Keras we managed to build our model sequentially.

The model we used is based on LeNet architecture, the input shape is 32x32x1. First convolution layer will have a depth of 60, a filter size of (5, 5), and a stride of (1, 1) and it is composed of two identical convolution operators. The main advantage when using a second convolution layer is that it has more flexibility in expressing non-linear transformations without losing information. MaxPool removes information from the signal, dropout forces distributed representation, thus both effectively make it harder to propagate information. A convolution sweeps the window through images then calculates its input and filter dot product pixel values. This process emphasizes the relevant features of the image. With this process, we detect a particular feature from the input image and produce feature maps, resulting from the convolution process which emphasizes the important features. These resulting features will always change depending on the filter values affected by the gradient descent to minimize prediction loss. For traffic sign recognition, highly non-linear transformation has to be applied on raw data, so stacking multiple convolutions (with ReLU) will make it easier to learn. we observe that if we remove the nonlinear operation, the model performance decreases. Rectified linear unit also known as ReLU and it receives a real number as a parameter and returns the received number if it is greater than 0 otherwise it returns 0. The next layer is called max pooling layer and effectively downsizes the data by only selecting the max value pixel for adjacent pixels. LeNet uses a (2, 2) filter size.

Next LeNet has a second convolution layer with depth of 30, filter size of (3, 3) and ReLU as an activation function, together with a max-pool layer. After the second stack of convolutions we use dropout, which is a form of regularization where weights are kept with a probability p; the unkept weights are thus “dropped”. This prevents the model from overfitting. The data resulting from the convolution layers is flattened before the fully connected layers. The flattening process takes a two-dimensional matrix of features and transforms it into a single vector that can be fed to the fully connected neural network classifier. The last part of the model consists of two fully connected layers. The first one uses ReLU as an activation function then a dropout layer is added to reduce overfitting. The second layer is represented by a fully connected layer with size of 17 representing the number of classes. The output of convolution and pooling layers is flattened into a single vector of values, each value representing a probability that a certain feature belongs to a traffic sign class. 

Input values go into the first layer of neurons where they are multiplied by weights and pass through a ReLU activation function. In the fully connected layer an input image goes through the network and it returns as a stack of matrices of features which are compressed into a single vector. When the input image is a traffic sign, certain values in the vector tend to be higher, so the network knows to categories the sign to the corresponding class. We used SoftMax function as the final activation function in the network as this function normalizes an input vector into a range that often leads to a probabilistic interpretation. The fully connected part of the neural network goes through its own backpropagation process in order to determine the most accurate weights. Each neuron receives weights that prioritize the most appropriate traffic sign class so that neurons decide which class has the highest probability and that is the network answer for the input image. 

| Layer (Type)                   |   Output Shape            |     Param   |
|--------------------------------|---------------------------|-------------|
| Conv2d_12 (Conv2D)             |   (None, 28, 28, 6)       |     156     |
| max_pooling2d_12 (MaxPooling)  |   (None, 14, 14, 6)       |     0       |
| conv2d_13 (Conv2D)             |   (None, 10, 10, 16)      |     2416    |
| max_pooling2d_13 (MaxPooling)  |   (None, 5, 5, 16)        |     0       |
| flatten_5 (Flatten)            |   (None , 400)            |     0       |
| dense_17 (Dense)               |   (None , 120)            |    48120    |
| dense_18 (Dense)               |   (None, 84)              |    10164    |
| dropout_5 (Dropout)            |   (None, 84)              |     0       |
| dense_19 (dense)               |   (None, 43)              |     3655    |

Total params : 64,511 <br />
Trainable params : 64,511 <br />
Non-trainable params: 0 <br />

**Training And Validation** <br />
In the training step an input is fed to the network and all the layers elements effectively constitute a transformation of this input to a predicted output. In the process of training all filters and weights are randomly assigned using gaussian distribution. The variation between this predicted output and the actual output is defined as loss. The loss value is then passed backwards through filters and used to adjust the values of the filters and neuron weights. This process is called backpropagation and what it does is to minimize the difference between predicted and actual output. This way the value of the filters are constantly adjusted during training and the system is considered trained when the loss is minimized. 
 
An epoch is when the whole dataset is passed forward and backward through the network but only once. 

batch_size_val = 50  <br />
steps_per_epoch_val= 250 <br />
epochs_vl =30 <br />

