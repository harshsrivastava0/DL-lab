from google.colab import drive
drive.mount('/content/drive')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import string
import random
import os
import time
import pickle
from tqdm.notebook import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

from sklearn.model_selection import train_test_split
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu

# Set random seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Configuration parameters
class Config:
    # Data parameters
    max_sentence_length = 50
    max_vocab_size = 15000
    sample_size = 10000  # Number of sentence pairs to use

    # Model parameters
    embedding_dim = 256
    hidden_dim = 512
    num_layers = 2
    dropout = 0.2

    # Training parameters
    batch_size = 64
    learning_rate = 0.001
    num_epochs = 10
    teacher_forcing_ratio = 0.5
    grad_clip = 1.0

    # Attention parameters
    attention_type = 'bahdanau'  # Options: 'none', 'bahdanau', 'luong'

config = Config()

# Define special tokens
PAD_token = 0
SOS_token = 1
EOS_token = 2
UNK_token = 3

# Define Language class for vocabulary processing
class Language:
    def __init__(self, name, max_vocab_size=15000):
        self.name = name
        self.max_vocab_size = max_vocab_size
        self.word2index = {"<pad>": PAD_token, "<sos>": SOS_token, "<eos>": EOS_token, "<unk>": UNK_token}
        self.word2count = {}
        self.index2word = {PAD_token: "<pad>", SOS_token: "<sos>", EOS_token: "<eos>", UNK_token: "<unk>"}
        self.n_words = 4  # Count SOS, EOS, PAD, UNK

    def add_sentence(self, sentence):
        for word in sentence.split():
            self.add_word(word)

    def add_word(self, word):
        if word not in self.word2index:
            self.word2count[word] = 1
            self.word2index[word] = self.n_words
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

    def trim(self):
        keep_words = []
        for k, v in self.word2count.items():
            keep_words.append((k, v))

        keep_words = sorted(keep_words, key=lambda x: -x[1])[:self.max_vocab_size-4]  # -4 for special tokens

        # Reset dictionaries
        self.word2index = {"<pad>": PAD_token, "<sos>": SOS_token, "<eos>": EOS_token, "<unk>": UNK_token}
        self.word2count = {}
        self.index2word = {PAD_token: "<pad>", SOS_token: "<sos>", EOS_token: "<eos>", UNK_token: "<unk>"}
        self.n_words = 4

        # Rebuild with most common words
        for word, count in keep_words:
            self.word2index[word] = self.n_words
            self.index2word[self.n_words] = word
            self.word2count[word] = count
            self.n_words += 1

# Function to preprocess text
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"([.!?])", r" \1", text)  # Add space before punctuation
    text = re.sub(r"[^a-zA-Z.!?áéíóúüñ¿¡]+", r" ", text)  # Keep only letters and Spanish characters
    text = text.strip()
    return text

# Function to read and prepare dataset
def prepare_data(file_path, sample_size=None):
    print("Reading data...")
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Sample a subset if specified
    if sample_size and sample_size < len(lines):
        lines = random.sample(lines, sample_size)

    # Create language instances
    source_lang = Language('english', max_vocab_size=config.max_vocab_size)
    target_lang = Language('spanish', max_vocab_size=config.max_vocab_size)

    pairs = []
    for line in tqdm(lines):
        parts = line.strip().split('\t')
        if len(parts) == 2:
            source_text = preprocess_text(parts[0])
            target_text = preprocess_text(parts[1])

            # Skip pairs that are too long
            if (len(source_text.split()) <= config.max_sentence_length and
                len(target_text.split()) <= config.max_sentence_length):
                source_lang.add_sentence(source_text)
                target_lang.add_sentence(target_text)
                pairs.append([source_text, target_text])

    print(f"Read {len(pairs)} sentence pairs")

    # Trim vocabularies
    source_lang.trim()
    target_lang.trim()
    print(f"Trimmed vocabulary sizes: English = {source_lang.n_words}, Spanish = {target_lang.n_words}")

    return source_lang, target_lang, pairs

# Function to convert sentences to indexes
def indexes_from_sentence(lang, sentence):
    indexes = []
    for word in sentence.split():
        if word in lang.word2index:
            indexes.append(lang.word2index[word])
        else:
            indexes.append(UNK_token)

    indexes.append(EOS_token)
    return indexes

# Dataset class
class TranslationDataset(Dataset):
    def __init__(self, pairs, source_lang, target_lang):
        self.pairs = pairs
        self.source_lang = source_lang
        self.target_lang = target_lang

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        source_sentence = self.pairs[idx][0]
        target_sentence = self.pairs[idx][1]

        source_indexes = indexes_from_sentence(self.source_lang, source_sentence)
        target_indexes = indexes_from_sentence(self.target_lang, target_sentence)

        return torch.tensor(source_indexes), torch.tensor(target_indexes)

# Collate function for DataLoader
def collate_fn(batch):
    source_batch, target_batch = [], []
    for source, target in batch:
        source_batch.append(source)
        target_batch.append(target)

    source_batch = pad_sequence(source_batch, batch_first=True, padding_value=PAD_token)
    target_batch = pad_sequence(target_batch, batch_first=True, padding_value=PAD_token)

    return source_batch, target_batch

# Encoder model
class Encoder(nn.Module):
    def __init__(self, input_dim, embed_dim, hidden_dim, num_layers, dropout):
        super(Encoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.embedding = nn.Embedding(input_dim, embed_dim)
        self.rnn = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        # src: [batch_size, src_seq_len]
        embedded = self.dropout(self.embedding(src))
        # embedded: [batch_size, src_seq_len, embed_dim]

        outputs, (hidden, cell) = self.rnn(embedded)
        # outputs: [batch_size, src_seq_len, hidden_dim]
        # hidden: [num_layers, batch_size, hidden_dim]
        # cell: [num_layers, batch_size, hidden_dim]

        return outputs, hidden, cell

# Attention modules
class BahdanauAttention(nn.Module):
    def __init__(self, hidden_dim):
        super(BahdanauAttention, self).__init__()
        self.W_a = nn.Linear(hidden_dim * 2, hidden_dim)
        self.v_a = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, query, keys):
        # query: [batch_size, hidden_dim]
        # keys: [batch_size, src_seq_len, hidden_dim]

        batch_size = query.shape[0]
        src_len = keys.shape[1]

        # Repeat query for each source position
        query = query.unsqueeze(1).repeat(1, src_len, 1)
        # query: [batch_size, src_seq_len, hidden_dim]

        # Concatenate query and keys
        energy = torch.tanh(self.W_a(torch.cat((query, keys), dim=2)))
        # energy: [batch_size, src_seq_len, hidden_dim]

        attention = self.v_a(energy).squeeze(2)
        # attention: [batch_size, src_seq_len]

        # Apply softmax to get attention weights
        attention_weights = F.softmax(attention, dim=1)
        # attention_weights: [batch_size, src_seq_len]

        return attention_weights

class LuongAttention(nn.Module):
    def __init__(self, hidden_dim):
        super(LuongAttention, self).__init__()
        self.W_a = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, query, keys):
        # query: [batch_size, hidden_dim]
        # keys: [batch_size, src_seq_len, hidden_dim]

        # Project query
        query = self.W_a(query)
        # query: [batch_size, hidden_dim]

        # Compute dot product attention
        attention = torch.bmm(keys, query.unsqueeze(2)).squeeze(2)
        # attention: [batch_size, src_seq_len]

        # Apply softmax to get attention weights
        attention_weights = F.softmax(attention, dim=1)
        # attention_weights: [batch_size, src_seq_len]

        return attention_weights

# Decoder model
class Decoder(nn.Module):
    def __init__(self, output_dim, embed_dim, hidden_dim, num_layers, dropout, attention_type='none'):
        super(Decoder, self).__init__()
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.attention_type = attention_type

        self.embedding = nn.Embedding(output_dim, embed_dim)

        # Add attention if specified
        if attention_type == 'bahdanau':
            self.attention = BahdanauAttention(hidden_dim)
            self.rnn = nn.LSTM(embed_dim + hidden_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)
        elif attention_type == 'luong':
            self.attention = LuongAttention(hidden_dim)
            self.rnn = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)
        else:  # No attention
            self.rnn = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)

        self.fc_out = nn.Linear(hidden_dim * 2 if attention_type != 'none' else hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, cell, encoder_outputs=None):
        # input: [batch_size, 1]
        # hidden: [num_layers, batch_size, hidden_dim]
        # cell: [num_layers, batch_size, hidden_dim]
        # encoder_outputs: [batch_size, src_seq_len, hidden_dim]

        input = input.unsqueeze(1)  # Add sequence dimension
        embedded = self.dropout(self.embedding(input))
        # embedded: [batch_size, 1, embed_dim]

        if self.attention_type == 'bahdanau':
            # Apply Bahdanau attention - additive before LSTM
            query = hidden[-1]  # Use top layer hidden state
            attention_weights = self.attention(query, encoder_outputs)
            # attention_weights: [batch_size, src_seq_len]

            # Apply attention weights to encoder outputs
            context = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs)
            # context: [batch_size, 1, hidden_dim]

            # Concatenate with input embedding
            rnn_input = torch.cat((embedded, context), dim=2)
            # rnn_input: [batch_size, 1, embed_dim + hidden_dim]

            output, (hidden, cell) = self.rnn(rnn_input, (hidden, cell))
            # output: [batch_size, 1, hidden_dim]

            # Concatenate output with context for prediction
            output = torch.cat((output, context), dim=2)
            # output: [batch_size, 1, hidden_dim * 2]

        elif self.attention_type == 'luong':
            # Apply Luong attention - multiplicative after LSTM
            output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
            # output: [batch_size, 1, hidden_dim]

            query = output.squeeze(1)  # Current decoder output
            attention_weights = self.attention(query, encoder_outputs)
            # attention_weights: [batch_size, src_seq_len]

            # Apply attention weights to encoder outputs
            context = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs)
            # context: [batch_size, 1, hidden_dim]

            # Concatenate output with context for prediction
            output = torch.cat((output, context), dim=2)
            # output: [batch_size, 1, hidden_dim * 2]

        else:  # No attention
            output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
            # output: [batch_size, 1, hidden_dim]

        prediction = self.fc_out(output.squeeze(1))
        # prediction: [batch_size, output_dim]

        return prediction, hidden, cell, attention_weights if self.attention_type != 'none' else None

# Seq2Seq model
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        # src: [batch_size, src_seq_len]
        # trg: [batch_size, trg_seq_len]

        batch_size = src.shape[0]
        trg_len = trg.shape[1]
        trg_vocab_size = self.decoder.output_dim

        # Tensor to store decoder outputs
        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)

        # Store attention weights if using attention
        attentions = torch.zeros(batch_size, trg_len, src.shape[1]).to(self.device) if self.decoder.attention_type != 'none' else None

        # Encode the source sequence
        encoder_outputs, hidden, cell = self.encoder(src)

        # First input to the decoder is the <SOS> token
        input = trg[:, 0]

        for t in range(1, trg_len):
            # Pass through decoder
            output, hidden, cell, attention = self.decoder(input, hidden, cell, encoder_outputs)

            # Store prediction and attention
            outputs[:, t, :] = output
            if attention is not None:
                attentions[:, t, :] = attention

            # Teacher forcing decision
            teacher_force = random.random() < teacher_forcing_ratio

            # Get the highest predicted token
            top1 = output.argmax(1)

            # Next input is either the target word or the predicted word
            input = trg[:, t] if teacher_force else top1

        return outputs, attentions

def train(model, iterator, optimizer, criterion, clip, device, teacher_forcing_ratio=0.5):
    model.train()
    epoch_loss = 0

    for i, (src, trg) in enumerate(iterator):
        src, trg = src.to(device), trg.to(device)

        optimizer.zero_grad()

        output, _ = model(src, trg, teacher_forcing_ratio)

        # Exclude the first token (<SOS>)
        output_dim = output.shape[-1]
        output = output[:, 1:].reshape(-1, output_dim)
        trg = trg[:, 1:].reshape(-1)

        # Calculate loss
        loss = criterion(output, trg)

        # Backpropagation
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        # Update parameters
        optimizer.step()

        epoch_loss += loss.item()

        if i % 100 == 0:
            print(f"Batch {i}, Loss: {loss.item():.4f}")

    return epoch_loss / len(iterator)

# Evaluation function
def evaluate(model, iterator, criterion, device):
    model.eval()
    epoch_loss = 0

    with torch.no_grad():
        for i, (src, trg) in enumerate(iterator):
            src, trg = src.to(device), trg.to(device)

            # Forward pass with no teacher forcing
            output, _ = model(src, trg, 0)

            # Exclude the first token (<SOS>)
            output_dim = output.shape[-1]
            output = output[:, 1:].reshape(-1, output_dim)
            trg = trg[:, 1:].reshape(-1)

            # Calculate loss
            loss = criterion(output, trg)

            epoch_loss += loss.item()

    return epoch_loss / len(iterator)

# Function to calculate BLEU score
def calculate_bleu(model, data_loader, target_lang, device):
    model.eval()
    references = []
    hypotheses = []

    with torch.no_grad():
        for src, trg in data_loader:
            src = src.to(device)

            # Forward pass
            batch_size = src.shape[0]
            src_len = src.shape[1]

            # Encode source
            encoder_outputs, hidden, cell = model.encoder(src)

            # Start with <SOS> token
            input = torch.LongTensor([SOS_token] * batch_size).to(device)

            # Store generated translations
            translations = torch.zeros(batch_size, config.max_sentence_length).long().fill_(PAD_token).to(device)
            translations[:, 0] = SOS_token

            # Store attention if using attention
            attentions = torch.zeros(batch_size, config.max_sentence_length, src_len).to(device) if model.decoder.attention_type != 'none' else None

            for t in range(1, config.max_sentence_length):
                # Forward through decoder
                output, hidden, cell, attention = model.decoder(input, hidden, cell, encoder_outputs)

                # Get prediction
                pred = output.argmax(1)

                # Store prediction
                translations[:, t] = pred

                # Store attention
                if attention is not None:
                    attentions[:, t, :] = attention

                # Stop if all EOS
                if all(pred == EOS_token):
                    break

                # Next input is predicted token
                input = pred

            # Convert translations to words
            for i in range(batch_size):
                # Get reference (ground truth) sentence
                ref_sent = []
                for idx in trg[i].cpu().numpy():
                    if idx == EOS_token:
                        break
                    if idx != PAD_token and idx != SOS_token:
                        ref_sent.append(target_lang.index2word[idx.item()])
                references.append([ref_sent])

                # Get hypothesis (prediction) sentence
                hyp_sent = []
                for idx in translations[i].cpu().numpy():
                    if idx == EOS_token:
                        break
                    if idx != PAD_token and idx != SOS_token:
                        hyp_sent.append(target_lang.index2word[idx.item()])
                hypotheses.append(hyp_sent)

    # Calculate BLEU score
    bleu_score = corpus_bleu(references, hypotheses) * 100
    return bleu_score, references, hypotheses, attentions if model.decoder.attention_type != 'none' else None

# Translation function
def translate_sentence(sentence, source_lang, target_lang, model, device, max_len=50):
    model.eval()

    # Process input sentence
    tokens = sentence.lower().split()
    tokens = [token for token in tokens if token.strip()]

    # Convert to indexes
    indexes = [source_lang.word2index.get(token, UNK_token) for token in tokens]
    indexes.append(EOS_token)

    # Convert to tensor
    src_tensor = torch.LongTensor(indexes).unsqueeze(0).to(device)

    # Get encoder outputs
    encoder_outputs, hidden, cell = model.encoder(src_tensor)

    # Start with <SOS> token
    input = torch.LongTensor([SOS_token]).to(device)

    # Store translation and attention
    translation = []
    attention_matrix = torch.zeros(max_len, len(indexes))

    for t in range(max_len):
        # Forward through decoder
        output, hidden, cell, attention = model.decoder(input, hidden, cell, encoder_outputs)

        # Store attention
        if attention is not None:
            attention_matrix[t] = attention.squeeze()

        # Get predicted token
        pred_token = output.argmax(1).item()

        # Stop if EOS
        if pred_token == EOS_token:
            break

        # Add token to translation if not special token
        if pred_token != PAD_token and pred_token != SOS_token:
            translation.append(target_lang.index2word[pred_token])

        # Next input is predicted token
        input = torch.LongTensor([pred_token]).to(device)

    return ' '.join(translation), attention_matrix[:len(translation)+1, :]

# Visualize attention
def display_attention(sentence, translation, attention, source_lang, target_lang):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)

    # Convert sentence from indexes to words
    source_words = [source_lang.index2word[idx] for idx in sentence if idx != PAD_token]
    target_words = ['<sos>'] + translation.split() + ['<eos>']

    attention = attention[:len(target_words), :len(source_words)]

    cax = ax.matshow(attention.detach().cpu().numpy(), cmap='viridis')

    ax.set_xticklabels([''] + source_words, rotation=90)
    ax.set_yticklabels([''] + target_words)

    ax.xaxis.set_major_locator(plt.MultipleLocator(1))
    ax.yaxis.set_major_locator(plt.MultipleLocator(1))

    plt.colorbar(cax)
    plt.show()

# Main execution flow
def main():
    # Step 1: Load and preprocess data
    # For Kaggle, let's assume the file is uploaded and saved as en-es.txt
    file_path = '/content/drive/My Drive/spa.txt'  # Update this path

    source_lang, target_lang, pairs = prepare_data(file_path, sample_size=config.sample_size)

    # Split data into train, validation, and test sets
    train_pairs, test_pairs = train_test_split(pairs, test_size=0.2, random_state=SEED)
    val_pairs, test_pairs = train_test_split(test_pairs, test_size=0.5, random_state=SEED)

    print(f"Number of training examples: {len(train_pairs)}")
    print(f"Number of validation examples: {len(val_pairs)}")
    print(f"Number of testing examples: {len(test_pairs)}")

    # Create datasets and data loaders
    train_dataset = TranslationDataset(train_pairs, source_lang, target_lang)
    val_dataset = TranslationDataset(val_pairs, source_lang, target_lang)
    test_dataset = TranslationDataset(test_pairs, source_lang, target_lang)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, collate_fn=collate_fn)

    # Step 2: Train model with Bahdanau attention
    print("\nTraining model with Bahdanau attention...")

    # Initialize encoder and decoder with Bahdanau attention
    encoder_attn = Encoder(
        input_dim=source_lang.n_words,
        embed_dim=config.embedding_dim,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        dropout=config.dropout
    )

    decoder_attn = Decoder(
        output_dim=target_lang.n_words,
        embed_dim=config.embedding_dim,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        dropout=config.dropout,
        attention_type='bahdanau'
    )

    # Initialize the model
    model_attn = Seq2Seq(encoder_attn, decoder_attn, device).to(device)

    # Define loss function and optimizer
    criterion_attn = nn.CrossEntropyLoss(ignore_index=PAD_token)
    optimizer_attn = optim.Adam(model_attn.parameters(), lr=config.learning_rate)

    # Training loop for the model with Bahdanau attention
    best_valid_loss_attn = float('inf')

    for epoch in range(config.num_epochs):
        start_time = time.time()

        train_loss = train(model_attn, train_loader, optimizer_attn, criterion_attn, config.grad_clip, device, config.teacher_forcing_ratio)
        valid_loss = evaluate(model_attn, val_loader, criterion_attn, device)

        end_time = time.time()
        epoch_mins, epoch_secs = divmod(end_time - start_time, 60)

        # Save model if it has the best validation loss
        if valid_loss < best_valid_loss_attn:
            best_valid_loss_attn = valid_loss
            torch.save(model_attn.state_dict(), 'best-model-bahdanau-attention.pt')

        print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs:.2f}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {np.exp(train_loss):.3f}')
        print(f'\tValid Loss: {valid_loss:.3f} | Valid PPL: {np.exp(valid_loss):.3f}')

    # Calculate BLEU score for the model with Bahdanau attention
    model_attn.load_state_dict(torch.load('best-model-bahdanau-attention.pt'))
    bleu_score_attn, references_attn, hypotheses_attn, attentions_attn = calculate_bleu(model_attn, test_loader, target_lang, device)
    print(f"BLEU score (with Bahdanau attention): {bleu_score_attn:.2f}")

    # Step 3: Train model with Luong attention
    print("\nTraining model with Luong attention...")

    # Initialize encoder and decoder with Luong attention
    encoder_luong = Encoder(
        input_dim=source_lang.n_words,
        embed_dim=config.embedding_dim,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        dropout=config.dropout
    )

    decoder_luong = Decoder(
        output_dim=target_lang.n_words,
        embed_dim=config.embedding_dim,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        dropout=config.dropout,
        attention_type='luong'
    )

    # Initialize the model
    model_luong = Seq2Seq(encoder_luong, decoder_luong, device).to(device)

    # Define loss function and optimizer
    criterion_luong = nn.CrossEntropyLoss(ignore_index=PAD_token)
    optimizer_luong = optim.Adam(model_luong.parameters(), lr=config.learning_rate)

    # Training loop for the model with Luong attention
    best_valid_loss_luong = float('inf')

    for epoch in range(config.num_epochs):
        start_time = time.time()

        train_loss = train(model_luong, train_loader, optimizer_luong, criterion_luong, config.grad_clip, device, config.teacher_forcing_ratio)
        valid_loss = evaluate(model_luong, val_loader, criterion_luong, device)

        end_time = time.time()
        epoch_mins, epoch_secs = divmod(end_time - start_time, 60)

        # Save model if it has the best validation loss
        if valid_loss < best_valid_loss_luong:
            best_valid_loss_luong = valid_loss
            torch.save(model_luong.state_dict(), 'best-model-luong-attention.pt')

        print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs:.2f}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {np.exp(train_loss):.3f}')
        print(f'\tValid Loss: {valid_loss:.3f} | Valid PPL: {np.exp(valid_loss):.3f}')

    # Calculate BLEU score for the model with Luong attention
    model_luong.load_state_dict(torch.load('best-model-luong-attention.pt'))
    bleu_score_luong, references_luong, hypotheses_luong, attentions_luong = calculate_bleu(model_luong, test_loader, target_lang, device)
    print(f"BLEU score (with Luong attention): {bleu_score_luong:.2f}")

    # Step 4: Compare model performance
    print("\nModel Performance Comparison:")
    print(f"Model without attention - BLEU score: {bleu_score:.2f}")
    print(f"Model with Bahdanau attention - BLEU score: {bleu_score_attn:.2f}")
    print(f"Model with Luong attention - BLEU score: {bleu_score_luong:.2f}")

    # Step 5: Visualize attention for a few examples
    def visualize_attention_examples(model, source_lang, target_lang, examples):
        for sentence in examples:
            src_tokens = preprocess_text(sentence).split()
            src_indexes = [source_lang.word2index.get(token, UNK_token) for token in src_tokens]
            src_indexes.append(EOS_token)

            translation, attention_matrix = translate_sentence(
                sentence, source_lang, target_lang, model, device
            )

            print(f"Source: {sentence}")
            print(f"Translation: {translation}")

            if attention_matrix is not None:
                display_attention(src_indexes, translation, attention_matrix, source_lang, target_lang)
            print("\n")

    # Sample sentences for visualization
    example_sentences = [
        "hello how are you",
        "i love learning languages",
        "where is the nearest restaurant",
        "can you help me with directions",
        "what time is the next train"
    ]

    print("\nVisualizing Bahdanau Attention:")
    visualize_attention_examples(model_attn, source_lang, target_lang, example_sentences)

    print("\nVisualizing Luong Attention:")
    visualize_attention_examples(model_luong, source_lang, target_lang, example_sentences)


# Call the main function
if __name__ == "__main__":
    main()
