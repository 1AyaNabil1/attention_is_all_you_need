---
layout: default
title: Architecture
---

<div class="content">

# Architecture Documentation

This document provides a detailed explanation of the Transformer architecture implementation.

---

## Table of Contents

1. [Overview](#overview)
2. [Input Embedding](#input-embedding)
3. [Positional Encoding](#positional-encoding)
4. [Multi-Head Attention](#multi-head-attention)
5. [Feed-Forward Networks](#feed-forward-networks)
6. [Layer Normalization](#layer-normalization)
7. [Encoder](#encoder)
8. [Decoder](#decoder)
9. [Output Projection](#output-projection)

---

## Overview

The Transformer architecture is based on the encoder-decoder structure with attention mechanisms as the core building block. Unlike RNNs, it processes entire sequences in parallel, making it highly efficient for training.

```
Input Sequence → Embedding → Positional Encoding → Encoder → Decoder → Output Projection → Output Sequence
                                                       ↑          ↑
                                                       └──────────┘
                                                    Cross-Attention
```

**Key Specifications:**
- Model dimension (d_model): 512
- Number of layers (N): 6 (both encoder and decoder)
- Number of attention heads (h): 8
- Feed-forward dimension (d_ff): 2048
- Dropout rate: 0.1

---

## Input Embedding

**Class**: `InputEmbedding`

Converts discrete tokens into continuous vector representations.

### Implementation Details:

```python
class InputEmbedding(torch.nn.Module):
    def __init__(self, d_model, vocab_size: int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)
```

### Key Features:

- **Scaling**: Embeddings are multiplied by √d_model to maintain variance
- **Learned Embeddings**: Parameters are learned during training
- **Shared Weights**: Source and target embeddings can optionally share weights

**Input**: Token indices (batch_size, seq_len)  
**Output**: Embedded vectors (batch_size, seq_len, d_model)

---

## Positional Encoding

**Class**: `PositionalEncoding`

Since the Transformer has no inherent notion of sequence order, positional encodings add position information to the embeddings.

### Implementation Details:

Uses sinusoidal functions as described in the paper:

```
PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

Where:
- `pos` is the position in the sequence
- `i` is the dimension index

### Key Features:

- **Fixed Encodings**: Not learned, computed using sine and cosine functions
- **Relative Positions**: The model can learn to attend by relative positions
- **Dropout**: Applied after adding positional encodings to embeddings

**Input**: Embedded vectors (batch_size, seq_len, d_model)  
**Output**: Position-encoded vectors (batch_size, seq_len, d_model)

---

## Multi-Head Attention

**Class**: `MultiHeadAttention`

The core mechanism that allows the model to focus on different parts of the input sequence.

### Attention Formula:

```
Attention(Q, K, V) = softmax(QK^T / √d_k) V
```

### Multi-Head Mechanism:

Instead of a single attention function, the model uses multiple attention "heads" in parallel:

1. **Linear Projections**: Q, K, V are projected into h different representation subspaces
2. **Parallel Attention**: Each head performs attention independently
3. **Concatenation**: Outputs are concatenated and projected

```
MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W^O
where head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
```

### Implementation Details:

- **d_k = d_v = d_model / h = 64**: Dimension per head
- **Number of heads (h)**: 8
- **Masking**: Optional mask to prevent attention to certain positions
- **Dropout**: Applied to attention weights

**Input**: Q, K, V tensors and optional mask  
**Output**: Attention output (batch_size, seq_len, d_model)

### Types of Attention:

1. **Self-Attention (Encoder)**: Q = K = V = input
2. **Masked Self-Attention (Decoder)**: Prevents attending to future positions
3. **Cross-Attention (Decoder)**: Q from decoder, K and V from encoder

---

## Feed-Forward Networks

**Class**: `FeedForwardBlock`

Position-wise fully connected feed-forward network applied to each position independently.

### Formula:

```
FFN(x) = max(0, xW_1 + b_1)W_2 + b_2
```

### Implementation Details:

```python
class FeedForwardBlock(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)  # 512 → 2048
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)  # 2048 → 512

    def forward(self, x):
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))
```

### Key Features:

- **Two Linear Transformations**: With ReLU activation in between
- **Dimension Expansion**: Expands to 2048, then projects back to 512
- **Dropout**: Applied after ReLU activation

**Input**: (batch_size, seq_len, d_model)  
**Output**: (batch_size, seq_len, d_model)

---

## Layer Normalization

**Class**: `LayerNormalization`

Normalizes inputs across the feature dimension to stabilize training.

### Formula:

```
LayerNorm(x) = α * (x - μ) / (σ + ε) + β
```

Where:
- μ: mean across features
- σ: standard deviation across features
- α, β: learnable parameters
- ε: small constant for numerical stability (10^-6)

### Key Features:

- **Per-Sample Normalization**: Applied independently to each sample
- **Learnable Parameters**: Scale (α) and shift (β) parameters
- **Stabilizes Training**: Reduces internal covariate shift

---

## Encoder

**Class**: `Encoder`, `EncoderLayer`

The encoder consists of N=6 identical layers, each with two sub-layers.

### Encoder Layer Structure:

```
Input → Layer Norm → Multi-Head Self-Attention → Residual Connection
                                                          ↓
      ← Residual Connection ← Feed-Forward Network ← Layer Norm
```

### Components:

1. **Multi-Head Self-Attention**: Allows each position to attend to all positions
2. **Feed-Forward Network**: Position-wise fully connected network
3. **Residual Connections**: Around each sub-layer
4. **Layer Normalization**: Applied before each sub-layer

### Implementation:

```python
class EncoderLayer(nn.Module):
    def forward(self, x, src_mask):
        # Self-attention with residual connection
        x = self.residual_connections[0](
            x, lambda x: self.self_attention_block(x, x, x, src_mask)
        )
        # Feed-forward with residual connection
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x
```

**Input**: Source sequence (batch_size, seq_len, d_model)  
**Output**: Encoded representation (batch_size, seq_len, d_model)

---

## Decoder

**Class**: `Decoder`, `DecoderBlock`

The decoder consists of N=6 identical layers, each with three sub-layers.

### Decoder Layer Structure:

```
Input → Layer Norm → Masked Multi-Head Self-Attention → Residual
                                                             ↓
         ← Residual ← Feed-Forward ← Layer Norm ← Residual ←
                                                      ↑
                        Encoder Output → Cross-Attention
```

### Components:

1. **Masked Multi-Head Self-Attention**: Prevents attending to future positions
2. **Multi-Head Cross-Attention**: Attends to encoder output
3. **Feed-Forward Network**: Position-wise fully connected network
4. **Residual Connections**: Around each sub-layer
5. **Layer Normalization**: Applied before each sub-layer

### Implementation:

```python
class DecoderBlock(nn.Module):
    def forward(self, x, encoder_output, src_mask, tgt_mask):
        # Masked self-attention
        x = self.residual_connections[0](
            x, lambda x: self.self_attention_block(x, x, x, tgt_mask)
        )
        # Cross-attention with encoder output
        x = self.residual_connections[1](
            x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask)
        )
        # Feed-forward network
        x = self.residual_connections[2](x, self.feed_forward_block)
        return x
```

### Masking:

- **Source Mask**: Prevents attending to padding tokens in source sequence
- **Target Mask**: Causal mask ensures autoregressive property (no peeking ahead)

**Input**: Target sequence and encoder output  
**Output**: Decoded representation (batch_size, seq_len, d_model)

---

## Output Projection

**Class**: `ProjectionLayer`

Projects decoder output to vocabulary space for token prediction.

### Implementation:

```python
class ProjectionLayer(nn.Module):
    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        return self.proj(x)
```

### Key Features:

- **Linear Transformation**: Maps d_model to vocab_size
- **Softmax**: Applied during inference/training to get probabilities
- **Output**: Logits for each token in vocabulary

**Input**: (batch_size, seq_len, d_model)  
**Output**: (batch_size, seq_len, vocab_size)

---

## Complete Model

**Class**: `Transformer`

Combines all components into the complete Transformer model.

### Building the Model:

```python
def build_transformer(
    src_vocab_size: int,
    tgt_vocab_size: int,
    src_seq_len: int,
    tgt_seq_len: int,
    d_model: int = 512,
    N: int = 6,
    h: int = 8,
    dropout: float = 0.1,
    d_ff: int = 2048
) -> Transformer:
    # Create embeddings
    src_embed = InputEmbedding(d_model, src_vocab_size)
    tgt_embed = InputEmbedding(d_model, tgt_vocab_size)
    
    # Create positional encodings
    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
    tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)
    
    # Create encoder and decoder blocks...
    # Initialize parameters with Xavier uniform
    
    return transformer
```

### Parameter Initialization:

All parameters with dimension > 1 are initialized using Xavier uniform initialization for stable training.

---

## Model Flow

### Training:

```
1. Source tokens → Embedding → Pos. Encoding → Encoder
2. Target tokens → Embedding → Pos. Encoding → Decoder (with encoder output)
3. Decoder output → Projection → Loss computation with labels
```

### Inference (Greedy Decoding):

```
1. Encode source sequence once
2. Initialize decoder input with [SOS] token
3. Loop:
   - Decode with current tokens
   - Project to vocabulary
   - Select highest probability token
   - Append to decoder input
   - Stop if [EOS] token or max length reached
```

---

## Computational Complexity

Per layer complexity for sequence length n and representation dimension d:

| Operation | Self-Attention | Recurrent | Convolutional |
|-----------|---------------|-----------|---------------|
| Complexity per Layer | O(n²·d) | O(n·d²) | O(k·n·d²) |
| Sequential Operations | O(1) | O(n) | O(1) |
| Maximum Path Length | O(1) | O(n) | O(log_k(n)) |

The Transformer's self-attention has constant path length but quadratic complexity in sequence length, making it efficient for shorter sequences.

---

## References

- [Original Paper: Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/)

</div>
- [The Annotated Transformer](http://nlp.seas.harvard.edu/annotated-transformer/)

---

[← Back to Home](index.md) | [Next: Training Guide →](training.md)