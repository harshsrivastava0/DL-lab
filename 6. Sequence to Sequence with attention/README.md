# LSTM Encoder-Decoder (With Attention)

## Overview

This project enhances the basic seq2seq model by adding attention mechanisms for better English-to-Spanish translation performance.

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
- Decoder: LSTM with attention
- Embeddings: Trainable or pre-trained
- Attention Types:
  - Bahdanau (Additive)
  - Luong (Multiplicative)

## Training

- Loss: Cross-entropy
- Optimizer: Adam (or similar)
- Strategy: Teacher forcing

## Evaluation

- BLEU score
- Compare with baseline model
- Visualize attention weights for sample outputs


## Reference

- Bahdanau et al., "Neural Machine Translation by Jointly Learning to Align and Translate" (2015)
- Luong et al., "Effective Approaches to Attention-based Neural Machine Translation" (2015)


