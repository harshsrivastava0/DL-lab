#EXPERIMENT2:
1. TRAIN A NEURAL NETWORK TO CLASSIFY LINEARY SEPARABLE DATASET(WITHOUT ANY HIDDEN LAYER
2. OBSERVE ITS FAILURE TO CLASSIFY NON LINEAR DATASET
3. ADD A HIDDEN LAYER AND OBSERVE THE DIFFERENCE


##Aim
To compare the performance of a neural network in classifying linearly separable and non-linearly separable datasets:

Phase 1: Train a neural network with no hidden layers to classify a linearly separable dataset (such as AND/OR).
Phase 2: Add a hidden layer and train again, now on both linear and non-linear datasets (such as XOR).



##Theory
Neural Network Basics: A neural network is a computational model inspired by biological neural networks. A simple neural network consists of layers of nodes (neurons), where each node is connected to nodes in adjacent layers by weighted connections.

No hidden layer: A neural network with only an input layer and an output layer can only classify linearly separable datasets. This means it can only draw straight decision boundaries to separate the classes.
Hidden layer: Adding a hidden layer allows the network to learn non-linear decision boundaries. This enables the neural network to classify non-linearly separable datasets.
Linear vs Non-Linear Separability:

Linearly separable: Data points of different classes can be separated by a straight line (or hyperplane in higher dimensions). Example: AND, OR, or simple binary classification datasets.
Non-linearly separable: No straight line can separate the classes. Example: XOR problem.
Activation Function:

The Sigmoid function will be used to introduce non-linearity into the network. Its output ranges from 0 to 1, which works well for classification tasks.
Sigmoid activation helps in transforming the linear combinations of inputs into non-linear activations, especially when hidden layers are added.
Backpropagation and Gradient Descent:

Backpropagation is the method used to train a neural network by updating the weights based on the error (difference between predicted and actual output). It propagates the error backward through the network and adjusts the weights to minimize the loss function.
Procedure
Dataset Generation:

Linearly separable dataset: Create a dataset with simple binary classification tasks such as the AND/OR gate.
Non-linearly separable dataset: Create a dataset like XOR, where classes cannot be separated by a straight line.
Training without Hidden Layer (Phase 1):

##Procedure
Initialize a neural network with 2 input nodes (for 2 features) and 1 output node (binary classification).
Use random weights and biases for the network.
Train the network using backpropagation and gradient descent for the linearly separable dataset.
Observe the network's ability to converge and classify correctly.
Training with Hidden Layer (Phase 2):

Now, modify the network by adding a hidden layer (e.g., 2 neurons in the hidden layer).
Use random weights and biases for the hidden and output layers.
Train the network on both the linear dataset (like AND/OR) and non-linear dataset (like XOR).
Observe the network’s ability to classify both types of datasets.
Evaluation:

Evaluate the performance by comparing the predicted outputs with the actual class labels.
Track accuracy over the number of epochs.
Plot the decision boundaries of the classifier and observe how the hidden layer affects non-linearly separable data.


##Conclusion:
Without Hidden Layer: The neural network can classify linearly separable data correctly but fails to handle non-linearly separable data (like XOR).
With Hidden Layer: Adding the hidden layer allows the network to classify both linearly and non-linearly separable data.
