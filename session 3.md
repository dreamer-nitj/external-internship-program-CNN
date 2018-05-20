### Dilated Convolution
Dilated convolution is just the normal convolution operation with modified kernel, to be exact, wider kernel.

> Dilation is largely the same as run-of-the-mill convolution (frankly so is deconvolution), except that it introduces gaps into it's kernels, i.e. whereas a standard kernel would typically slide over contiguous sections of the input, it's dilated counterpart may, for instance, "encircle" a larger section of the image --while still only have as many weights/inputs as the standard form. 

The familiar discrete convolution is simply the 1-dilated convolution. If we increase this value from 1 to 2, then the receptive field of the kernel grows without the loss of resolution or coverage. this achieves integrating more knowledge of the wider context with less cost. This has usage in detecting object boundriese and image segmentation.

![Image](https://i.stack.imgur.com/qA0Kx.gif)

### Activation Function
The activation function in a neural network acts on the output of intermediate layers to transform the processed output from that layer. The activation function transforms the value into the interval of [-1, 1] or [0, 1]. This basically tells if a neuron should be activated or not.

The activation function are of 2 types, 1) Linear transformation 2) Non Linear transformation type. Activation functions make the back-propagation possible using gradients and helps in processing complex tasks like language translation and image classifications.

Some examples of activation functions are as follows:
#### a) Sigmoid function:
The sigmoid function output values falls within [0, 1]

![Image](https://cdn-images-1.medium.com/max/800/1*f9erByySVjTjohfFdNkJYQ.jpeg)

#### b) ReLu function (Rectified Linear Unit):
This function has the output value 0 for negative values and acts as an identity function for x > 0. so 

f(x) = 0 for x < 0
f(x) = x for x >= 0

![Image](https://qph.ec.quoracdn.net/main-qimg-4229dd280e03b7b3a5dc26c808c4b15b)

### Initialization
In deep neural networks, the weights for the network layers need to be initialized correctly otherwise the network may never converge even after thousands of iterations.

![Image](https://intoli.com/blog/neural-network-initialization/img/training-losses.png)

There are various kinds of techniques to initialize the networks weights like:
a) Random
b) Xavier
c) Glorot

#### a) Random weight initializaation
In this strategy, the weights are sampled from a standard distribution ( often a normal distribution ) with low deviation. The low deviation allows us to bias the network towards a simple 0 solution, without the bad repercussions of actually initializing the weights to 0.

#### b) Xavier weight initialization
In this strategy, the weights could be sampled from a gaussian distribution which has the zero mean and some finite variance. our aim is to basically keep the variance same after each layer pass. This helps us keep the signal from exploding to a high value or vanishing to zero.

The formula for weight variance in this is:
```
var(wi) = 1/N(avg)
where N(avg) = N(in) + N(out) / 2
```
