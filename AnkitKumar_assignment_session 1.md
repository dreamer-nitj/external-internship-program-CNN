### Convolution
The convolution operation in Convolutional neural network applies a filter to the input image to take the weighted sum over them, pass the output through activation function to pass it on to the next layer.

> Convolution operation produces a feature map using filter/kernel to be fed to the next layer. The output from the process of applying a feature/filter/kernel to the input produces a feature map. 

The CNNs consist of input, multiple hidden layers ( fully connected ), activation function and weights and biases to apply to the inputs to produce the final output. The filters are generally the weight matrix that has the same number of channels as input channels. the number of output will be same as the number of inputs. CNNs may also use the the concept of pooling which basically tries to reduce the number of input neurons to be fed to the next layer by combining the outputs of these neurons to produce one output. This could be just an average of all the neurons or could be the max value among them as done in max pooling.

> Max pooling takes input as the outputs of multiple neurons and combine them to produce only one output which is the max value among them.

The epoch/iteration is a process of running through the training set once. It takes the CNN more than one epoch to fully converge over the training data.

![Image](https://raw.githubusercontent.com/iamaaditya/iamaaditya.github.io/master/images/conv_arithmetic/full_padding_no_strides_transposed.gif)

The convolution operation over an image represented by 32x32x3 (widthxheightxchannels) with a filter 5x5x3 produces 28x28x1. The filter is applied to the whole image by sliding it over to this image and taking the dot product between the filter and the chunk of input image undergoing convolution.

![Image](https://cdn-images-1.medium.com/max/800/1*mcBbGiV8ne9NhF3SlpjAsA.png)

### 3x3 Convolution
A 3x3 convolution is the process of using a filter with weight matrix size of 3x3 and applying it over the whole image. This process generates a feature map which extracts the most important portions of the image. its also called feature extraction. A 3x3 convolution can be performed with a padding of zero, one or more depending on how big the input image is. in CNNs, we generally use a padding size of 0.

> Below is an image of the filter with horizontal and vertical padding of 1

![Image](https://k-d-w.org/uploads/images/autoencoder/padding_strides.gif)