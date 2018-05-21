## Forward propagation and Backward propagation

#### input X
|  | X | | | 
| --- | --- | --- | ---
| 1 | 0 | 1 | 0 |
| 1 | 0 | 1 | 1 |
| 0 | 1 | 0 | 1 |

#### output Y
| Y |
|--- |
| 1 |
| 1 |
| 0 |

### Python code for forward & back propagation
```
import numpy as np


# ReLU function
def relu(p):
    # ReLU activation function
    return np.maximum(p, 0)


# derivative of ReLU function
def derivatives_relu(p):
    # Derivative of ReLU
    return 1 * (p > 0)


# input array X
x = np.array([[1, 0, 1, 0], [1, 0, 1, 1], [0, 1, 0, 1]])

# output array y
y = np.array([[1], [1], [0]])


# variable initialization
epochs = 100  # number of iterations
input_layer_neurons = x.shape[1]  # i.e number of channels, in this case 4
hidden_layer_neurons = 3  # number of channels in hidden layer
output_neurons = 1  # number of channels in the output layer

# weights and bias initialization
wh = np.random.uniform(size=(input_layer_neurons, hidden_layer_neurons))
bh = np.random.uniform(size=(1, hidden_layer_neurons))
weight_out = np.random.uniform(size=(hidden_layer_neurons, output_neurons))
bias_out = np.random.uniform(size=(1, output_neurons))

print('input', x)
print('weight_hidden', wh)
print('bias_hidden', bh)
print('weight_output', weight_out)
print('bias_output', bias_out)

# forward propagation
hidden_layer_input1 = np.dot(x, wh)
hidden_layer_input = hidden_layer_input1 + bh
hidden_layer_activations = relu(hidden_layer_input)

output_layer_input1 = np.dot(hidden_layer_activations, weight_out)
output_layer_input = output_layer_input1 + bias_out
output = relu(output_layer_input)

# back propagation
error_output = y - output
slope_output_layer = derivatives_relu(output)
slope_hidden_layer = derivatives_relu(hidden_layer_activations)

delta_output = error_output * slope_output_layer
error_at_hidden_layer = delta_output.dot(weight_out.T)
delta_hidden_layer = error_at_hidden_layer * slope_hidden_layer

# do adjustments
weight_out -= hidden_layer_activations.T.dot(delta_output)
bias_out -= np.sum(delta_output, axis=0, keepdims=True)
wh -= x.T.dot(delta_hidden_layer)
bh -= np.sum(delta_hidden_layer, axis=0, keepdims=True)

print("hidden_layer_activations", hidden_layer_activations)
print("output", output)
print("error_output", error_output)
print("slope_output_layer", slope_output_layer)
print("slope_hidden_layer", slope_hidden_layer)
print("delta_output_layer", delta_output)
print("error_hidden_layer", error_at_hidden_layer)
print("delta_hidden_layer", delta_hidden_layer)

```

### output from the above code
```
('input', array([[1, 0, 1, 0],
       [1, 0, 1, 1],
       [0, 1, 0, 1]]))
('weight_hidden', array([[ 0.43430322,  0.75521676,  0.50732082],
       [ 0.58932832,  0.17122361,  0.32693257],
       [ 0.78043841,  0.05454598,  0.25157794],
       [ 0.81713755,  0.58508727,  0.73588931]]))
('bias_hidden', array([[ 0.23438114,  0.1211831 ,  0.55089755]]))
('weight_output', array([[ 0.88972827],
       [ 0.58849862],
       [ 0.49670947]]))
('bias_output', array([[ 0.88129083]]))
('hidden_layer_activations', array([[ 1.44912278,  0.93094584,  1.30979631],
       [ 2.26626033,  1.51603311,  2.04568562],
       [ 1.64084701,  0.87749398,  1.61371944]]))
('output', array([[ 3.36906491],
       [ 4.80594154],
       [ 3.65915253]]))
('error_output', array([[-2.36906491],
       [-3.80594154],
       [-3.65915253]]))
('slope_output_layer', array([[1],
       [1],
       [1]]))
('slope_hidden_layer', array([[1, 1, 1],
       [1, 1, 1],
       [1, 1, 1]]))
('delta_output_layer', array([[-2.36906491],
       [-3.80594154],
       [-3.65915253]]))
('error_hidden_layer', array([[-2.10782404, -1.39419144, -1.17673698],
       [-3.3862538 , -2.23979136, -1.8904472 ],
       [-3.25565147, -2.15340623, -1.81753571]]))
('delta_hidden_layer', array([[-2.10782404, -1.39419144, -1.17673698],
       [-3.3862538 , -2.23979136, -1.8904472 ],
       [-3.25565147, -2.15340623, -1.81753571]]))
```

