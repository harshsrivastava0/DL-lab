#Experiment: Handwritten Digit Classification Using MNIST and NumPy

##Aim
To implement a handwritten digit classification system using the MNIST dataset in .ubyte format, leveraging only NumPy for data processing, model creation, and evaluation.

##Theory
The MNIST dataset contains images of handwritten digits (0-9) represented as 28x28 grayscale images, along with their corresponding labels. The dataset is widely used for image classification tasks.

Each image in the dataset is represented as a 1D array of 784 pixels (28x28 flattened). Using a simple neural network with one hidden layer, we will classify the digits based on their pixel intensity patterns.

##Key concepts:

###Neural Network:
A computational model inspired by the structure of the human brain.

###Activation Function:
Applies non-linearity to the output of neurons (e.g., ReLU, sigmoid, softmax).

###Loss Function:
Measures the difference between predicted and actual labels (e.g., cross-entropy loss).

###Gradient Descent:
Optimizes the weights of the network to minimize the loss function.


##PROCEDURE:
Load and Preprocess Data:

Read the MNIST dataset from .ubyte files.
Normalize pixel values to the range [0, 1].
Split the data into training and testing sets.
Define the Neural Network:

Input layer: 784 neurons (for 28x28 images).
Hidden layer: 128 neurons with ReLU activation.
Output layer: 10 neurons with softmax activation.
Initialize Parameters:

Randomly initialize weights and biases for all layers.
Forward Propagation:

Compute activations for the hidden and output layers using matrix multiplication.
Loss Calculation:

Use cross-entropy loss to quantify the prediction error.
Backward Propagation:

Compute gradients of weights and biases using chain rule.
Update Parameters:

Apply gradient descent to adjust weights and biases.
Evaluate the Model:

Measure accuracy on the test set.
Plot and Analyze Results:

Visualize training loss and test accuracy over epochs.

