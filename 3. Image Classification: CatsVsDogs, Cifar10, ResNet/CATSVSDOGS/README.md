EXP3.1:
Convolutional Layers:

Conv1: From 3 input channels to 32 feature maps.
Conv2: From 32 to 64 feature maps.
Conv3: From 64 to 128 feature maps.
Pooling: After each convolution, max pooling reduces spatial dimensions (height and width) but keeps the number of feature maps unchanged.

Flattening: After the third convolutional block, the 128 feature maps are flattened into a 1D vector, which is passed to fully connected layers.

Fully Connected Layers:

First fully connected layer: 512 neurons.
Output layer: 2 neurons (for binary classification).
So, the number of features starts from 3 (input channels) and increases to 128 (after the final convolution), then the features are flattened into a vector passed through 512 neurons before the final 2-class output.

dataset: cats vs dogs
optimiser: adam,lr=0.001

weight initialisation:default

activation function :relu

learnings: I am satisfied with this architecture i have applied batch normalisation a well as dropout layer to reduce the chance overfitting will experiment with more different types of optimisers 
