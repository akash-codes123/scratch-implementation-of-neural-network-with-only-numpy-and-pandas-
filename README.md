# scratch-implementation-of-neural-network-with-only-numpy-and-pandas-
in this project the main aim is to understand the in depth formation of a simple deeplearning neural network which consists of : 1 input lyer , 1 hidden layer and 1 output layer.

dataset for this project - https://www.kaggle.com/competitions/digit-recognizer

STEPS to follow through the project or create your own neural network-

Training a neural network for multiclass classification involves several mathematical concepts and operations. Let's break down the steps involved, including the mathematical logic behind each operation:

1. Initialization:
   - Initialize the weights (W) and biases (b) of the neural network. These parameters will be learned during training.

2. Forward Propagation:
   - Compute the weighted sum (Z) of the inputs (X) and the weights (W) of the hidden layer: Z1 = W1 * X + b1.
   - Apply the ReLU activation function to the hidden layer's weighted sum: A1 = ReLU(Z1).
   - Compute the weighted sum of the hidden layer's activations (A1) and the weights (W) of the output layer: Z2 = W2 * A1 + b2.
   - Apply the softmax activation function to the output layer's weighted sum to obtain class probabilities: A2 = softmax(Z2).

3. Cost Function:
   - Calculate the cost (loss) of the model's predictions by comparing them to the true labels. Commonly used cost functions for multiclass classification include the categorical cross-entropy or the softmax loss.

4. Backward Propagation:
   - Compute the derivative of the cost function with respect to the output layer's activations: dZ2 = A2 - Y, where Y is the true label in one-hot encoded form.
   - Compute the gradients of the weights (W2) and biases (b2) of the output layer: dW2 = (1/m) * dZ2 * A1.T and db2 = (1/m) * np.sum(dZ2, axis=1, keepdims=True), where m is the number of samples.
   - Compute the derivative of the hidden layer's activations using the chain rule: dZ1 = W2.T * dZ2 * ReLU_deriv(Z1), where ReLU_deriv(Z1) represents the derivative of the ReLU activation function.
   - Compute the gradients of the weights (W1) and biases (b1) of the hidden layer: dW1 = (1/m) * dZ1 * X.T and db1 = (1/m) * np.sum(dZ1, axis=1, keepdims=True).

5. Gradient Descent:
   - Update the weights and biases using the gradients computed during backward propagation: W1 = W1 - learning_rate * dW1, b1 = b1 - learning_rate * db1, W2 = W2 - learning_rate * dW2, b2 = b2 - learning_rate * db2. Here, learning_rate represents the step size for updating the parameters and is a hyperparameter that needs to be tuned.

6. Repeat Steps 2-5:
   - Repeat the forward propagation, cost calculation, backward propagation, and gradient descent steps for a specified number of epochs or until the model converges.
By iteratively adjusting the weights and biases based on the gradients computed during backpropagation and using gradient descent optimization, the neural network learns to make better predictions over time, ultimately improving its performance in multiclass classification tasks.
