Text Generation using RNN and LSTM
Objective
The aim of this experiment is to explore text generation using Recurrent Neural Networks (RNNs) and understand the impact of different word representations:

One-Hot Encoding

Trainable Word Embeddings

Train an RNN model on a dataset of 100 poems and compare the performance of both encoding techniques.

Dataset
Use the provided dataset of 100 poems for training your text generation model. The dataset consists of multiple lines of poetry, which will be used to generate text sequences.

Part 1: One-Hot Encoding Approach
Preprocessing
Tokenize the text into words.

Convert each word into a one-hot vector.

Model Architecture
Use an RNN and LSTM model.

The input should be one-hot encoded word sequences.

Train the model to predict the next word in a sequence.

Implementation Steps
Tokenize the dataset and create a vocabulary.

Convert words into one-hot encoded vectors.

Define an RNN model using PyTorch.

Train the model using the dataset.

Generate text using the trained model.

Part 2: Trainable Word Embeddings Approach
Preprocessing
Tokenize the text into words.

Convert each word into an index.

Model Architecture
Use an embedding layer in the RNN model.

Train the embedding layer along with the model.

Predict the next word in a sequence.

Implementation Steps
Tokenize the dataset and create a vocabulary.

Convert words into indexed sequences.

Define an RNN model with an embedding layer using PyTorch.

Train the model and compare performance with the one-hot encoding method.

Generate text using the trained model.

Comparison and Analysis
Compare the training time and loss for both methods.

Evaluate the quality of generated text.

Discuss the advantages and disadvantages of each approach.
