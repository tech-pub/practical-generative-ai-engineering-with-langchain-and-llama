import torch
import torch.nn as nn
import torch.optim as optim
import math

# --- 1. Illustrating Core Concepts of Generative AI and LLMs ---
# This simplistic example demonstrates key ideas like:
# - Embeddings (representing words numerically)
# - Positional Encoding (adding order info)
# - Attention (focusing on relevant parts)
# - A very basic 'Transformer-like' block (linear layers instead of full self-attention)
# - Pre-training (learning to predict the next word)

# --- Hyperparameters ---
VOCAB_SIZE = 10  # Number of unique tokens (e.g., words)
EMBED_DIM = 8    # Dimension of the word embeddings
SEQ_LEN = 5      # Maximum sequence length
NUM_HEADS = 2    # For multi-head attention (simplified here)

# --- 2. Embeddings ---
# Maps discrete tokens (integers) to continuous vectors.
token_embedding_layer = nn.Embedding(VOCAB_SIZE, EMBED_DIM)

# --- 3. Positional Encoding ---
# Adds information about the position of tokens in the sequence.
# Essential because transformers process tokens in parallel without inherent order.
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0)) # Add batch dimension

    def forward(self, x):
        # x is (batch_size, seq_len, embed_dim)
        return x + self.pe[:, :x.size(1)]

positional_encoding_layer = PositionalEncoding(EMBED_DIM, SEQ_LEN)

# --- 4. Simplified Attention / Transformer Block Concept ---
# In a real transformer, this would be a full MultiHeadAttention layer.
# Here, we use linear layers to simulate some form of "feature extraction"
# that's position-aware, aiming to learn relationships, albeit very simplistically.
class SimpleTransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(SimpleTransformerBlock, self).__init__()
        # Simulate 'query', 'key', 'value' projections and output, but collapsed
        # into simple feed-forward layers for this basic example.
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.ReLU(),
            nn.Linear(embed_dim * 2, embed_dim)
        )
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # x is (batch_size, seq_len, embed_dim)
        residual = x
        x = self.ffn(x)
        x = self.layer_norm(x + residual) # Add & Norm
        return x

transformer_block = SimpleTransformerBlock(EMBED_DIM, NUM_HEADS)

# --- 5. Language Model Head (for next token prediction) ---
lm_head = nn.Linear(EMBED_DIM, VOCAB_SIZE) # Projects embedding to vocab probabilities

# --- 6. Pre-training Simulation (Next Token Prediction) ---
# Goal: Given a sequence, predict the next token.

# Example input sequence (batch_size=1, seq_len=4)
# Tokens: 1, 2, 3, 4
# Target (next token to predict): 5
input_sequence = torch.tensor([[1, 2, 3, 4]]) # Batch size 1, Sequence length 4
target_next_token = torch.tensor([5]) # Batch size 1, Target token 5

# --- Forward Pass ---
# 1. Embed tokens
embeddings = token_embedding_layer(input_sequence) # (1, 4, EMBED_DIM)

# 2. Add positional encoding
sequence_representation = positional_encoding_layer(embeddings) # (1, 4, EMBED_DIM)

# 3. Pass through simplified transformer block
processed_sequence = transformer_block(sequence_representation) # (1, 4, EMBED_DIM)

# 4. Take the last token's representation to predict the next
last_token_representation = processed_sequence[:, -1, :] # (1, EMBED_DIM)

# 5. Predict the next token's probabilities
prediction_logits = lm_head(last_token_representation) # (1, VOCAB_SIZE)

# --- Loss Calculation (simulating a single pre-training step) ---
loss_fn = nn.CrossEntropyLoss()
loss = loss_fn(prediction_logits, target_next_token)

print(f"Prediction logits (for VOCAB_SIZE={VOCAB_SIZE}): {prediction_logits}")
print(f"Target next token: {target_next_token.item()}")
print(f"Simulated Pre-training Loss: {loss.item():.4f}")

# This code snippet demonstrates the data flow and core components
# (embeddings, positional encoding, a simplified transformer block, and a prediction head)
# that are fundamental to modern LLMs and their pre-training objective.
# It omits the full complexity of self-attention, multi-layer transformers,
# and actual training loops, focusing on conceptual illustration.
