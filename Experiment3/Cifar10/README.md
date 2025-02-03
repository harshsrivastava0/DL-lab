Convolutional Layers (Feature Extraction)


Conv1: Takes 3 input channels (RGB) and increases them to 64 feature maps.


Conv2: Expands from 64 to 128 feature maps, learning more complex patterns.


Conv3: Further increases from 128 to 256 feature maps, capturing high-level details.


Pooling: After each convolution, max pooling reduces the height and width of the feature maps but keeps the number of feature maps the same.


Flattening (Transition to Fully Connected Layers)


After the third convolutional block, the 256 feature maps are flattened into a 1D vector of 4096 values, making it ready for classification.
Fully Connected Layers (Classification)


First fully connected layer: Reduces the 4096 features to 512 neurons, allowing the model to learn deep relationships.


Output layer: Reduces from 512 to 10 neurons, where each neuron represents one of the 10 CIFAR-10 classes.
