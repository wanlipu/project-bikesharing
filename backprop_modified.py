import numpy as np


def sigmoid(x):
    """
    Calculate sigmoid
    """
    return 1 / (1 + np.exp(-x))


x = np.array([[0.5, 0.1, -0.2]])  # 1 by 3
target = 0.6
learnrate = 0.5

weights_input_hidden = np.array([[0.5, -0.6], # 3 by 2
                                 [0.1, -0.2],
                                 [0.1, 0.7]])

weights_hidden_output = np.array([[0.1],        # 2 by 1
                                  [-0.3]])

## Forward pass
hidden_layer_input = np.dot(x, weights_input_hidden) # 1 by 3 dot 3 by 2 -> 1 by 2
hidden_layer_output = sigmoid(hidden_layer_input) # 1 by 2

output_layer_in = np.dot(hidden_layer_output, weights_hidden_output) # 1 by 2 dot 2 by 1 -> 1 by 1
output = sigmoid(output_layer_in) # 1 by 1

## Backwards pass
## TODO: Calculate output error
error = target - output

# TODO: Calculate error term for output layer
output_error_term = error * output * (1 - output) # 1 by 1

# TODO: Calculate error term for hidden layer
hidden_error_term = np.dot(output_error_term, weights_hidden_output.T) * hidden_layer_output * (1 - hidden_layer_output)   # 1 by 1 dot 1 by 2 -> 1 by 2
# hidden_error_term = weights_hidden_output * output_error_term * hidden_layer_output * (1 - hidden_layer_output)


# TODO: Calculate change in weights for hidden layer to output layer
delta_w_h_o = learnrate * output_error_term * hidden_layer_output

# TODO: Calculate change in weights for input layer to hidden layer
delta_w_i_h = (learnrate * np.dot(hidden_error_term.T, x)).T # 2 by 1 dot 1 by 3


print('Change in weights for hidden layer to output layer:')
print(delta_w_h_o)
print('Change in weights for input layer to hidden layer:')
print(delta_w_i_h)
