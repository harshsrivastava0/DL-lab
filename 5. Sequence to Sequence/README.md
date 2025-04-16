# LSTM Encoder-Decoder (Without Attention)

## Overview

This project implements a basic sequence-to-sequence (seq2seq) model using LSTM layers for English-to-Spanish translation without any attention mechanism.

## Dataset

- Format: English-Spanish sentence pairs separated by a tab.
- Example:
  Hello.    Hola.
- Split:
  - 80% training
  - 10% validation
  - 10% testing

## Preprocessing

- Tokenization
- Lowercasing
- Padding sequences

## Model

- Encoder: LSTM
- Decoder: LSTM with teacher forcing
- Embeddings: Trainable or pre-trained (e.g., GloVe)

## Training

- Loss: Cross-entropy
- Optimizer: Adam (or similar)
- Strategy: Teacher forcing

## Evaluation

- BLEU score on validation and test sets

## Reference

- Sutskever et al., "Sequence to Sequence Learning with Neural Networks" (2014)

