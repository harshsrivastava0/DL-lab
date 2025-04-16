# EXPERIMENT 4:
Text Generation using RNN and LSTM

##Objective
The aim of this experiment is to explore text generation using Recurrent Neural Networks (RNNs) and understand the impact of different word representations:
1. One-Hot Encoding
2. Trainable Word Embeddings
Train an RNN model on a dataset of 100 poems and compare the performance of both encoding techniques.

##Dataset
Use the provided dataset of 100 poems for training your text generation model. The dataset consists of multiple lines of poetry, which will be used to generate text sequences.

##Part 1: One-Hot Encoding Approach
###Model Architecture
Use an RNN and LSTM model.

###Preprocessing
1. Tokenize the text into words.
2. Convert each word into a one-hot vector.

###Implementation Steps
1. Define an RNN model using PyTorch.
2. Train the model using the dataset.
3. Generate text using the trained model.


##Part 2: Trainable Word Embeddings Approach
###Model Architecture
Use an embedding layer in the RNN model.

###Preprocessing
1. Tokenize the text into words.
2. Convert each word into an index.


###Implementation Steps

1. Define an RNN model with an embedding layer using PyTorch.
2. Train the model and compare performance with the one-hot encoding method.
3. Generate text using the trained model.


##Comparison and Analysis
1. Compare the training time and loss for both methods.
2. Evaluate the quality of generated text.
3. Discuss the advantages and disadvantages of each approach.
