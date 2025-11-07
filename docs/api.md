---
layout: default
title: API Reference
---

<div class="content">

# API Reference

Complete API documentation for all classes and functions in the Transformer implementation.

---

## Table of Contents

1. [Model Components](#model-components)
2. [Dataset](#dataset)
3. [Training Utilities](#training-utilities)
4. [Configuration](#configuration)

---

## Model Components

### `build_transformer()`

Factory function to build a complete Transformer model.

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
) -> Transformer
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `src_vocab_size` | int | - | Source vocabulary size |
| `tgt_vocab_size` | int | - | Target vocabulary size |
| `src_seq_len` | int | - | Maximum source sequence length |
| `tgt_seq_len` | int | - | Maximum target sequence length |
| `d_model` | int | 512 | Model dimension |
| `N` | int | 6 | Number of encoder/decoder layers |
| `h` | int | 8 | Number of attention heads |
| `dropout` | float | 0.1 | Dropout probability |
| `d_ff` | int | 2048 | Feed-forward dimension |

**Returns:** `Transformer` - Complete transformer model

**Example:**

```python
from src.model import build_transformer

model = build_transformer(
    src_vocab_size=10000,
    tgt_vocab_size=8000,
    src_seq_len=512,
    tgt_seq_len=512
)
```

---

### `InputEmbedding`

Converts token indices to dense vector embeddings.

```python
class InputEmbedding(torch.nn.Module):
    def __init__(self, d_model: int, vocab_size: int)
```

**Parameters:**
- `d_model` (int): Embedding dimension
- `vocab_size` (int): Size of vocabulary

**Methods:**

#### `forward(x)`

```python
def forward(self, x: torch.Tensor) -> torch.Tensor
```

**Parameters:**
- `x` (torch.Tensor): Input token indices, shape `(batch_size, seq_len)`

**Returns:** 
- torch.Tensor: Embedded vectors, shape `(batch_size, seq_len, d_model)`

**Example:**

```python
embedding = InputEmbedding(d_model=512, vocab_size=10000)
tokens = torch.tensor([[1, 2, 3, 4]])
embedded = embedding(tokens)  # Shape: (1, 4, 512)
```

---

### `PositionalEncoding`

Adds positional information to embeddings using sinusoidal functions.

```python
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, seq_len: int, dropout: float)
```

**Parameters:**
- `d_model` (int): Model dimension
- `seq_len` (int): Maximum sequence length
- `dropout` (float): Dropout probability

**Methods:**

#### `forward(x)`

```python
def forward(self, x: torch.Tensor) -> torch.Tensor
```

**Parameters:**
- `x` (torch.Tensor): Input embeddings, shape `(batch_size, seq_len, d_model)`

**Returns:**
- torch.Tensor: Position-encoded embeddings, same shape as input

**Example:**

```python
pos_encoding = PositionalEncoding(d_model=512, seq_len=100, dropout=0.1)
embedded = torch.randn(8, 50, 512)
encoded = pos_encoding(embedded)  # Shape: (8, 50, 512)
```

---

### `LayerNormalization`

Normalizes inputs across the feature dimension.

```python
class LayerNormalization(nn.Module):
    def __init__(self, eps: float = 10**-6)
```

**Parameters:**
- `eps` (float): Small constant for numerical stability (default: 1e-6)

**Attributes:**
- `alpha` (nn.Parameter): Learnable scale parameter
- `bias` (nn.Parameter): Learnable shift parameter

**Methods:**

#### `forward(x)`

```python
def forward(self, x: torch.Tensor) -> torch.Tensor
```

**Parameters:**
- `x` (torch.Tensor): Input tensor

**Returns:**
- torch.Tensor: Normalized tensor, same shape as input

---

### `FeedForwardBlock`

Position-wise feed-forward network.

```python
class FeedForwardBlock(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float)
```

**Parameters:**
- `d_model` (int): Input/output dimension
- `d_ff` (int): Hidden layer dimension
- `dropout` (float): Dropout probability

**Methods:**

#### `forward(x)`

```python
def forward(self, x: torch.Tensor) -> torch.Tensor
```

**Parameters:**
- `x` (torch.Tensor): Input tensor, shape `(batch_size, seq_len, d_model)`

**Returns:**
- torch.Tensor: Output tensor, same shape as input

**Example:**

```python
ff_block = FeedForwardBlock(d_model=512, d_ff=2048, dropout=0.1)
x = torch.randn(8, 50, 512)
output = ff_block(x)  # Shape: (8, 50, 512)
```

---

### `MultiHeadAttention`

Multi-head attention mechanism.

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, h: int, dropout: float)
```

**Parameters:**
- `d_model` (int): Model dimension
- `h` (int): Number of attention heads
- `dropout` (float): Dropout probability

**Attributes:**
- `d_k` (int): Dimension per head (d_model / h)
- `w_q` (nn.Linear): Query projection
- `w_k` (nn.Linear): Key projection
- `w_v` (nn.Linear): Value projection
- `w_o` (nn.Linear): Output projection

**Methods:**

#### `forward(query, key, value, mask)`

```python
def forward(
    self,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: Optional[torch.Tensor] = None
) -> torch.Tensor
```

**Parameters:**
- `query` (torch.Tensor): Query tensor, shape `(batch_size, seq_len, d_model)`
- `key` (torch.Tensor): Key tensor, shape `(batch_size, seq_len, d_model)`
- `value` (torch.Tensor): Value tensor, shape `(batch_size, seq_len, d_model)`
- `mask` (torch.Tensor, optional): Attention mask

**Returns:**
- torch.Tensor: Attention output, shape `(batch_size, seq_len, d_model)`

#### `attention(query, key, value, mask, dropout)` (static)

```python
@staticmethod
def attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: Optional[torch.Tensor],
    dropout: nn.Dropout
) -> Tuple[torch.Tensor, torch.Tensor]
```

**Parameters:**
- `query`, `key`, `value` (torch.Tensor): Attention inputs
- `mask` (torch.Tensor, optional): Attention mask
- `dropout` (nn.Dropout): Dropout layer

**Returns:**
- Tuple[torch.Tensor, torch.Tensor]: (attention_output, attention_scores)

**Example:**

```python
mha = MultiHeadAttention(d_model=512, h=8, dropout=0.1)
x = torch.randn(8, 50, 512)
output = mha(query=x, key=x, value=x)  # Self-attention
```

---

### `ResidualConnection`

Residual connection with layer normalization.

```python
class ResidualConnection(nn.Module):
    def __init__(self, features: int, dropout: float)
```

**Parameters:**
- `features` (int): Number of features
- `dropout` (float): Dropout probability

**Methods:**

#### `forward(x, sublayer)`

```python
def forward(
    self,
    x: torch.Tensor,
    sublayer: Callable[[torch.Tensor], torch.Tensor]
) -> torch.Tensor
```

**Parameters:**
- `x` (torch.Tensor): Input tensor
- `sublayer` (Callable): Sublayer function to apply

**Returns:**
- torch.Tensor: Output with residual connection

---

### `EncoderLayer`

Single encoder layer with self-attention and feed-forward network.

```python
class EncoderLayer(nn.Module):
    def __init__(
        self,
        features: int,
        self_attention_block: MultiHeadAttention,
        feed_forward_block: FeedForwardBlock,
        dropout: float
    )
```

**Parameters:**
- `features` (int): Model dimension
- `self_attention_block` (MultiHeadAttention): Self-attention mechanism
- `feed_forward_block` (FeedForwardBlock): Feed-forward network
- `dropout` (float): Dropout probability

**Methods:**

#### `forward(x, src_mask)`

```python
def forward(
    self,
    x: torch.Tensor,
    src_mask: torch.Tensor
) -> torch.Tensor
```

**Parameters:**
- `x` (torch.Tensor): Input tensor
- `src_mask` (torch.Tensor): Source mask

**Returns:**
- torch.Tensor: Encoder layer output

---

### `Encoder`

Complete encoder with N layers.

```python
class Encoder(nn.Module):
    def __init__(self, features: int, layers: nn.ModuleList)
```

**Parameters:**
- `features` (int): Model dimension
- `layers` (nn.ModuleList): List of encoder layers

**Methods:**

#### `forward(x, mask)`

```python
def forward(
    self,
    x: torch.Tensor,
    mask: torch.Tensor
) -> torch.Tensor
```

**Parameters:**
- `x` (torch.Tensor): Input tensor
- `mask` (torch.Tensor): Source mask

**Returns:**
- torch.Tensor: Encoder output

---

### `DecoderBlock`

Single decoder layer with self-attention, cross-attention, and feed-forward network.

```python
class DecoderBlock(nn.Module):
    def __init__(
        self,
        features: int,
        self_attention_block: MultiHeadAttention,
        cross_attention_block: MultiHeadAttention,
        feed_forward_block: FeedForwardBlock,
        dropout: float
    )
```

**Parameters:**
- `features` (int): Model dimension
- `self_attention_block` (MultiHeadAttention): Self-attention mechanism
- `cross_attention_block` (MultiHeadAttention): Cross-attention mechanism
- `feed_forward_block` (FeedForwardBlock): Feed-forward network
- `dropout` (float): Dropout probability

**Methods:**

#### `forward(x, encoder_output, src_mask, tgt_mask)`

```python
def forward(
    self,
    x: torch.Tensor,
    encoder_output: torch.Tensor,
    src_mask: torch.Tensor,
    tgt_mask: torch.Tensor
) -> torch.Tensor
```

**Parameters:**
- `x` (torch.Tensor): Decoder input
- `encoder_output` (torch.Tensor): Encoder output
- `src_mask` (torch.Tensor): Source mask
- `tgt_mask` (torch.Tensor): Target mask (causal)

**Returns:**
- torch.Tensor: Decoder layer output

---

### `Decoder`

Complete decoder with N layers.

```python
class Decoder(nn.Module):
    def __init__(self, features: int, layers: nn.ModuleList)
```

**Parameters:**
- `features` (int): Model dimension
- `layers` (nn.ModuleList): List of decoder layers

**Methods:**

#### `forward(x, encoder_output, src_mask, tgt_mask)`

```python
def forward(
    self,
    x: torch.Tensor,
    encoder_output: torch.Tensor,
    src_mask: torch.Tensor,
    tgt_mask: torch.Tensor
) -> torch.Tensor
```

**Parameters:**
- `x` (torch.Tensor): Decoder input
- `encoder_output` (torch.Tensor): Encoder output
- `src_mask` (torch.Tensor): Source mask
- `tgt_mask` (torch.Tensor): Target mask

**Returns:**
- torch.Tensor: Decoder output

---

### `ProjectionLayer`

Projects decoder output to vocabulary space.

```python
class ProjectionLayer(nn.Module):
    def __init__(self, d_model: int, vocab_size: int)
```

**Parameters:**
- `d_model` (int): Input dimension
- `vocab_size` (int): Vocabulary size

**Methods:**

#### `forward(x)`

```python
def forward(self, x: torch.Tensor) -> torch.Tensor
```

**Parameters:**
- `x` (torch.Tensor): Input tensor, shape `(batch_size, seq_len, d_model)`

**Returns:**
- torch.Tensor: Logits, shape `(batch_size, seq_len, vocab_size)`

---

### `Transformer`

Complete Transformer model.

```python
class Transformer(nn.Module):
    def __init__(
        self,
        encoder: Encoder,
        decoder: Decoder,
        src_embed: InputEmbedding,
        tgt_embed: InputEmbedding,
        src_pos: PositionalEncoding,
        tgt_pos: PositionalEncoding,
        projection_layer: ProjectionLayer
    )
```

**Methods:**

#### `encode(src, src_mask)`

```python
def encode(
    self,
    src: torch.Tensor,
    src_mask: torch.Tensor
) -> torch.Tensor
```

**Parameters:**
- `src` (torch.Tensor): Source token indices
- `src_mask` (torch.Tensor): Source mask

**Returns:**
- torch.Tensor: Encoder output

#### `decode(encoder_output, src_mask, tgt, tgt_mask)`

```python
def decode(
    self,
    encoder_output: torch.Tensor,
    src_mask: torch.Tensor,
    tgt: torch.Tensor,
    tgt_mask: torch.Tensor
) -> torch.Tensor
```

**Parameters:**
- `encoder_output` (torch.Tensor): Encoder output
- `src_mask` (torch.Tensor): Source mask
- `tgt` (torch.Tensor): Target token indices
- `tgt_mask` (torch.Tensor): Target mask

**Returns:**
- torch.Tensor: Decoder output

#### `project(x)`

```python
def project(self, x: torch.Tensor) -> torch.Tensor
```

**Parameters:**
- `x` (torch.Tensor): Decoder output

**Returns:**
- torch.Tensor: Vocabulary logits

---

## Dataset

### `BilingualDataset`

Dataset class for bilingual translation pairs.

```python
class BilingualDataset(Dataset):
    def __init__(
        self,
        ds,
        tokenizer_src,
        tokenizer_tgt,
        src_lang: str,
        tgt_lang: str,
        seq_len: int
    )
```

**Parameters:**
- `ds`: Hugging Face dataset
- `tokenizer_src`: Source language tokenizer
- `tokenizer_tgt`: Target language tokenizer
- `src_lang` (str): Source language code (e.g., 'en')
- `tgt_lang` (str): Target language code (e.g., 'fr')
- `seq_len` (int): Maximum sequence length

**Methods:**

#### `__len__()`

Returns the number of samples in the dataset.

#### `__getitem__(idx)`

```python
def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]
```

**Parameters:**
- `idx` (int): Sample index

**Returns:**
- Dict containing:
  - `encoder_input` (torch.Tensor): Encoder input tokens
  - `decoder_input` (torch.Tensor): Decoder input tokens
  - `encoder_mask` (torch.Tensor): Encoder attention mask
  - `decoder_mask` (torch.Tensor): Decoder attention mask (causal)
  - `label` (torch.Tensor): Target labels
  - `src_text` (str): Source text
  - `tgt_text` (str): Target text

**Example:**

```python
from src.dataset import BilingualDataset

dataset = BilingualDataset(
    ds=raw_dataset,
    tokenizer_src=src_tokenizer,
    tokenizer_tgt=tgt_tokenizer,
    src_lang='en',
    tgt_lang='fr',
    seq_len=350
)

sample = dataset[0]
print(sample['encoder_input'].shape)  # torch.Size([350])
```

---

### `causal_mask()`

Creates a causal mask for autoregressive decoding.

```python
def causal_mask(size: int) -> torch.Tensor
```

**Parameters:**
- `size` (int): Sequence length

**Returns:**
- torch.Tensor: Boolean mask, shape `(1, size, size)`

**Example:**

```python
from src.dataset import causal_mask

mask = causal_mask(5)
# Output: Upper triangular matrix with zeros above diagonal
# [[1, 0, 0, 0, 0],
#  [1, 1, 0, 0, 0],
#  [1, 1, 1, 0, 0],
#  [1, 1, 1, 1, 0],
#  [1, 1, 1, 1, 1]]
```

---

## Training Utilities

### `get_ds()`

Prepares training and validation datasets.

```python
def get_ds(config: Dict) -> Tuple[DataLoader, DataLoader, Tokenizer, Tokenizer]
```

**Parameters:**
- `config` (Dict): Configuration dictionary

**Returns:**
- Tuple containing:
  - `train_dataloader` (DataLoader): Training data loader
  - `val_dataloader` (DataLoader): Validation data loader
  - `tokenizer_src` (Tokenizer): Source tokenizer
  - `tokenizer_tgt` (Tokenizer): Target tokenizer

---

### `get_model()`

Creates and returns the Transformer model.

```python
def get_model(
    config: Dict,
    vocab_src_len: int,
    vocab_tgt_len: int
) -> Transformer
```

**Parameters:**
- `config` (Dict): Configuration dictionary
- `vocab_src_len` (int): Source vocabulary size
- `vocab_tgt_len` (int): Target vocabulary size

**Returns:**
- Transformer: Model instance

---

### `train_model()`

Main training function.

```python
def train_model(config: Dict) -> None
```

**Parameters:**
- `config` (Dict): Configuration dictionary

**Functionality:**
- Sets up device (CUDA/MPS/CPU)
- Loads datasets and tokenizers
- Initializes model and optimizer
- Runs training loop with validation
- Saves checkpoints

---

### `run_validation()`

Runs validation and computes metrics.

```python
def run_validation(
    model: Transformer,
    validation_ds: DataLoader,
    tokenizer_src: Tokenizer,
    tokenizer_tgt: Tokenizer,
    max_len: int,
    device: str,
    print_msg: Callable,
    global_step: int,
    writer: SummaryWriter,
    num_examples: int = 2
) -> None
```

**Parameters:**
- `model` (Transformer): Model to validate
- `validation_ds` (DataLoader): Validation data loader
- `tokenizer_src`, `tokenizer_tgt` (Tokenizer): Tokenizers
- `max_len` (int): Maximum sequence length
- `device` (str): Device to run on
- `print_msg` (Callable): Print function
- `global_step` (int): Current training step
- `writer` (SummaryWriter): TensorBoard writer
- `num_examples` (int): Number of examples to display

---

### `greedy_decode()`

Greedy decoding for sequence generation.

```python
def greedy_decode(
    model: Transformer,
    source: torch.Tensor,
    source_mask: torch.Tensor,
    tokenizer_src: Tokenizer,
    tokenizer_tgt: Tokenizer,
    max_len: int,
    device: str
) -> torch.Tensor
```

**Parameters:**
- `model` (Transformer): Trained model
- `source` (torch.Tensor): Source sequence
- `source_mask` (torch.Tensor): Source mask
- `tokenizer_src`, `tokenizer_tgt` (Tokenizer): Tokenizers
- `max_len` (int): Maximum generation length
- `device` (str): Device to run on

**Returns:**
- torch.Tensor: Generated sequence

---

### `get_or_build_tokenizer()`

Gets existing tokenizer or builds new one.

```python
def get_or_build_tokenizer(
    config: Dict,
    ds,
    lang: str
) -> Tokenizer
```

**Parameters:**
- `config` (Dict): Configuration dictionary
- `ds`: Dataset to build tokenizer from
- `lang` (str): Language code

**Returns:**
- Tokenizer: WordLevel tokenizer

---

## Configuration

### `get_config()`

Returns default configuration dictionary.

```python
def get_config() -> Dict
```

**Returns:**
- Dict: Configuration with default values

**Configuration Keys:**

```python
{
    'batch_size': 8,
    'num_epochs': 20,
    'lr': 10**-4,
    'seq_len': 350,
    'd_model': 512,
    'datasource': 'opus_books',
    'lang_src': 'en',
    'lang_tgt': 'fr',
    'model_folder': 'weights',
    'model_basename': 'tmodel_',
    'preload': 'latest',
    'tokenizer_file': 'tokenizer_{0}.json',
    'experiment_name': 'runs/tmodel'
}
```

---

### `get_weights_file_path()`

Gets path for model weights file.

```python
def get_weights_file_path(config: Dict, epoch: str) -> str
```

**Parameters:**
- `config` (Dict): Configuration dictionary
- `epoch` (str): Epoch number

**Returns:**
- str: Path to weights file

---

### `latest_weights_file_path()`

Finds the latest weights file.

```python
def latest_weights_file_path(config: Dict) -> Optional[str]
```

**Parameters:**
- `config` (Dict): Configuration dictionary

**Returns:**
- Optional[str]: Path to latest weights file, or None if not found

---

## Usage Examples

### Complete Training Pipeline

```python
from src.train import train_model
from src.config import get_config

# Get configuration
config = get_config()

# Customize if needed
config['batch_size'] = 16
config['num_epochs'] = 30

# Start training
train_model(config)
```

### Inference

```python
import torch
from src.model import build_transformer
from src.train import greedy_decode
from tokenizers import Tokenizer

# Load model
model = build_transformer(vocab_src_len, vocab_tgt_len, 512, 512)
model.load_state_dict(torch.load('weights/model.pt')['model_state_dict'])
model.eval()

# Load tokenizers
tokenizer_src = Tokenizer.from_file('tokenizer_en.json')
tokenizer_tgt = Tokenizer.from_file('tokenizer_fr.json')

# Translate
source_text = "Hello, world!"
source_tokens = tokenizer_src.encode(source_text).ids
source_tensor = torch.tensor([source_tokens])
source_mask = (source_tensor != pad_token).unsqueeze(0).unsqueeze(0)

# Generate translation
output = greedy_decode(model, source_tensor, source_mask, 
                       tokenizer_src, tokenizer_tgt, 350, 'cpu')
translation = tokenizer_tgt.decode(output.tolist())
print(translation)
```

---

[← Back to Home](index.md) | [Next: Examples →](examples.md)</div>
