---
layout: default
title: Examples and Tutorials
---

<div class="content">

# Examples and Tutorials

Practical examples and tutorials for using the Transformer implementation.

---

## Table of Contents

1. [Basic Usage](#basic-usage)
2. [Training from Scratch](#training-from-scratch)
3. [Fine-tuning](#fine-tuning)
4. [Inference and Translation](#inference-and-translation)
5. [Custom Dataset](#custom-dataset)
6. [Attention Visualization](#attention-visualization)
7. [Advanced Configurations](#advanced-configurations)

---

## Basic Usage

### Building a Transformer Model

The simplest way to create a Transformer model:

```python
from src.model import build_transformer

# Build with default parameters (paper configuration)
model = build_transformer(
    src_vocab_size=10000,  # Source vocabulary size
    tgt_vocab_size=8000,   # Target vocabulary size
    src_seq_len=512,       # Max source sequence length
    tgt_seq_len=512        # Max target sequence length
)

print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
# Output: Model parameters: ~65M
```

### Custom Hyperparameters

Customize the model architecture:

```python
# Smaller model for faster training/inference
small_model = build_transformer(
    src_vocab_size=10000,
    tgt_vocab_size=8000,
    src_seq_len=256,
    tgt_seq_len=256,
    d_model=256,      # Reduced dimension
    N=4,              # Fewer layers
    h=4,              # Fewer heads
    dropout=0.1,
    d_ff=1024         # Smaller feed-forward
)

# Larger model for better performance
large_model = build_transformer(
    src_vocab_size=32000,
    tgt_vocab_size=32000,
    src_seq_len=1024,
    tgt_seq_len=1024,
    d_model=768,      # Increased dimension
    N=12,             # More layers
    h=12,             # More heads
    dropout=0.1,
    d_ff=3072         # Larger feed-forward
)
```

---

## Training from Scratch

### Quick Start Training

Train with default configuration:

```python
from src.train import train_model
from src.config import get_config

# Get default configuration
config = get_config()

# Start training
train_model(config)
```

### Custom Training Configuration

Modify configuration for your needs:

```python
from src.config import get_config
from src.train import train_model

# Get base config
config = get_config()

# Customize training parameters
config['batch_size'] = 16          # Larger batch size
config['num_epochs'] = 30          # More epochs
config['lr'] = 5e-5                # Different learning rate
config['seq_len'] = 512            # Longer sequences
config['d_model'] = 768            # Larger model

# Customize dataset
config['lang_src'] = 'de'          # German to English
config['lang_tgt'] = 'en'
config['datasource'] = 'wmt14'     # Different dataset

# Customize paths
config['model_folder'] = 'my_weights'
config['model_basename'] = 'my_model_'
config['experiment_name'] = 'runs/my_experiment'

# Start training
train_model(config)
```

### Resume Training

Resume from a checkpoint:

```python
config = get_config()

# Resume from latest checkpoint
config['preload'] = 'latest'

# Or resume from specific epoch
config['preload'] = '10'  # Resume from epoch 10

train_model(config)
```

### Training with Validation

Monitor validation metrics during training:

```python
import torch
from src.train import train_model, run_validation
from torch.utils.tensorboard import SummaryWriter

config = get_config()

# Enable detailed validation logging
writer = SummaryWriter(config['experiment_name'])

train_model(config)

# View results in TensorBoard
# tensorboard --logdir runs/tmodel
```

---

## Fine-tuning

### Fine-tune on Custom Data

Fine-tune a pre-trained model on your dataset:

```python
import torch
from src.model import build_transformer
from src.config import get_config, get_weights_file_path

# Load pre-trained model
config = get_config()
model = build_transformer(
    src_vocab_size=vocab_src_size,
    tgt_vocab_size=vocab_tgt_size,
    src_seq_len=config['seq_len'],
    tgt_seq_len=config['seq_len']
)

# Load weights
weights_path = get_weights_file_path(config, '19')  # Epoch 19
state = torch.load(weights_path)
model.load_state_dict(state['model_state_dict'])

# Fine-tune with lower learning rate
config['lr'] = 1e-5
config['num_epochs'] = 5
config['preload'] = '19'  # Start from epoch 19

train_model(config)
```

### Transfer Learning

Use embeddings from a pre-trained model:

```python
# Load pre-trained model
pretrained_model = torch.load('pretrained_weights.pt')

# Create new model
new_model = build_transformer(
    src_vocab_size=new_vocab_size,
    tgt_vocab_size=new_vocab_size,
    src_seq_len=512,
    tgt_seq_len=512
)

# Transfer encoder weights
new_model.encoder.load_state_dict(
    pretrained_model['model_state_dict']['encoder']
)

# Freeze encoder for fine-tuning
for param in new_model.encoder.parameters():
    param.requires_grad = False
```

---

## Inference and Translation

### Simple Translation

Translate a single sentence:

```python
import torch
from tokenizers import Tokenizer
from src.model import build_transformer
from src.train import greedy_decode

# Load model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = build_transformer(vocab_src_size, vocab_tgt_size, 512, 512)
model.load_state_dict(torch.load('model.pt')['model_state_dict'])
model.to(device)
model.eval()

# Load tokenizers
tokenizer_src = Tokenizer.from_file('tokenizer_en.json')
tokenizer_tgt = Tokenizer.from_file('tokenizer_fr.json')

# Prepare input
source_text = "Hello, how are you?"
source_tokens = tokenizer_src.encode(source_text).ids

# Add SOS and EOS tokens
sos_token = tokenizer_src.token_to_id('[SOS]')
eos_token = tokenizer_src.token_to_id('[EOS]')
pad_token = tokenizer_src.token_to_id('[PAD]')

# Create input tensor
source_tokens = [sos_token] + source_tokens + [eos_token]
source_tensor = torch.tensor([source_tokens], dtype=torch.int64).to(device)

# Create mask
source_mask = (source_tensor != pad_token).unsqueeze(0).unsqueeze(0).int().to(device)

# Translate
output = greedy_decode(
    model, source_tensor, source_mask,
    tokenizer_src, tokenizer_tgt, 512, device
)

# Decode translation
translation = tokenizer_tgt.decode(output.squeeze(0).tolist())
print(f"Translation: {translation}")
```

### Batch Translation

Translate multiple sentences:

```python
def translate_batch(model, sentences, tokenizer_src, tokenizer_tgt, device, max_len=512):
    """Translate a batch of sentences."""
    model.eval()
    translations = []
    
    with torch.no_grad():
        for sentence in sentences:
            # Tokenize
            tokens = tokenizer_src.encode(sentence).ids
            tokens = [sos_token] + tokens + [eos_token]
            
            # Create tensor and mask
            source = torch.tensor([tokens]).to(device)
            source_mask = (source != pad_token).unsqueeze(0).unsqueeze(0).int().to(device)
            
            # Translate
            output = greedy_decode(
                model, source, source_mask,
                tokenizer_src, tokenizer_tgt, max_len, device
            )
            
            # Decode
            translation = tokenizer_tgt.decode(output.squeeze(0).tolist())
            translations.append(translation)
    
    return translations

# Use it
sentences = [
    "Hello, how are you?",
    "The weather is nice today.",
    "I love programming."
]

translations = translate_batch(model, sentences, tokenizer_src, tokenizer_tgt, device)
for src, tgt in zip(sentences, translations):
    print(f"{src} -> {tgt}")
```

### Translation Pipeline

Create a reusable translation pipeline:

```python
class TranslationPipeline:
    def __init__(self, model_path, tokenizer_src_path, tokenizer_tgt_path, device='cpu'):
        self.device = device
        
        # Load tokenizers
        self.tokenizer_src = Tokenizer.from_file(tokenizer_src_path)
        self.tokenizer_tgt = Tokenizer.from_file(tokenizer_tgt_path)
        
        # Load model
        vocab_src_size = self.tokenizer_src.get_vocab_size()
        vocab_tgt_size = self.tokenizer_tgt.get_vocab_size()
        
        self.model = build_transformer(vocab_src_size, vocab_tgt_size, 512, 512)
        state = torch.load(model_path, map_location=device)
        self.model.load_state_dict(state['model_state_dict'])
        self.model.to(device)
        self.model.eval()
        
        # Special tokens
        self.sos_token = self.tokenizer_src.token_to_id('[SOS]')
        self.eos_token = self.tokenizer_src.token_to_id('[EOS]')
        self.pad_token = self.tokenizer_src.token_to_id('[PAD]')
    
    def translate(self, text, max_len=512):
        """Translate a single text."""
        # Tokenize
        tokens = self.tokenizer_src.encode(text).ids
        tokens = [self.sos_token] + tokens + [self.eos_token]
        
        # Create tensor
        source = torch.tensor([tokens]).to(self.device)
        source_mask = (source != self.pad_token).unsqueeze(0).unsqueeze(0).int()
        
        # Translate
        with torch.no_grad():
            output = greedy_decode(
                self.model, source, source_mask,
                self.tokenizer_src, self.tokenizer_tgt,
                max_len, self.device
            )
        
        # Decode
        return self.tokenizer_tgt.decode(output.squeeze(0).tolist())

# Use the pipeline
pipeline = TranslationPipeline(
    model_path='opus_books_weights/tmodel_19.pt',
    tokenizer_src_path='tokenizer_en.json',
    tokenizer_tgt_path='tokenizer_fr.json',
    device='cuda'
)

translation = pipeline.translate("Hello, world!")
print(translation)
```

---

## Custom Dataset

### Loading Custom Data

Use your own parallel corpus:

```python
from torch.utils.data import Dataset
import pandas as pd

class CustomBilingualDataset(Dataset):
    def __init__(self, csv_path, tokenizer_src, tokenizer_tgt, seq_len):
        self.data = pd.read_csv(csv_path)
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.seq_len = seq_len
        
        self.sos_token = tokenizer_tgt.token_to_id("[SOS]")
        self.eos_token = tokenizer_tgt.token_to_id("[EOS]")
        self.pad_token = tokenizer_tgt.token_to_id("[PAD]")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        src_text = self.data.iloc[idx]['source']
        tgt_text = self.data.iloc[idx]['target']
        
        # Tokenize
        enc_tokens = self.tokenizer_src.encode(src_text).ids
        dec_tokens = self.tokenizer_tgt.encode(tgt_text).ids
        
        # Add padding
        enc_padding = self.seq_len - len(enc_tokens) - 2
        dec_padding = self.seq_len - len(dec_tokens) - 1
        
        # Build tensors
        encoder_input = torch.cat([
            torch.tensor([self.sos_token]),
            torch.tensor(enc_tokens),
            torch.tensor([self.eos_token]),
            torch.tensor([self.pad_token] * enc_padding)
        ])
        
        decoder_input = torch.cat([
            torch.tensor([self.sos_token]),
            torch.tensor(dec_tokens),
            torch.tensor([self.pad_token] * dec_padding)
        ])
        
        label = torch.cat([
            torch.tensor(dec_tokens),
            torch.tensor([self.eos_token]),
            torch.tensor([self.pad_token] * dec_padding)
        ])
        
        return {
            'encoder_input': encoder_input,
            'decoder_input': decoder_input,
            'encoder_mask': (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(),
            'decoder_mask': (decoder_input != self.pad_token).unsqueeze(0).int() & causal_mask(self.seq_len),
            'label': label,
            'src_text': src_text,
            'tgt_text': tgt_text
        }

# Use custom dataset
dataset = CustomBilingualDataset(
    csv_path='my_data.csv',
    tokenizer_src=tokenizer_src,
    tokenizer_tgt=tokenizer_tgt,
    seq_len=350
)

dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
```

### Building Custom Tokenizer

Train a tokenizer on your data:

```python
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

def train_custom_tokenizer(texts, vocab_size=10000):
    """Train a tokenizer on custom text data."""
    tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    
    trainer = WordLevelTrainer(
        special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"],
        vocab_size=vocab_size,
        min_frequency=2
    )
    
    # Train on texts
    tokenizer.train_from_iterator(texts, trainer=trainer)
    
    return tokenizer

# Example usage
texts = [
    "Hello world",
    "How are you",
    # ... more texts
]

tokenizer = train_custom_tokenizer(texts)
tokenizer.save("my_tokenizer.json")
```

---

## Attention Visualization

### Extract Attention Weights

Visualize what the model is attending to:

```python
import matplotlib.pyplot as plt
import numpy as np

def visualize_attention(model, source, target, tokenizer_src, tokenizer_tgt, layer=0, head=0):
    """Visualize attention weights for a specific layer and head."""
    model.eval()
    
    with torch.no_grad():
        # Encode
        encoder_output = model.encode(source, source_mask)
        
        # Decode
        decoder_output = model.decode(encoder_output, source_mask, target, target_mask)
        
        # Get attention weights from specific layer and head
        attention_weights = model.decoder.layers[layer].cross_attention_block.attention_scores
        attention_weights = attention_weights[0, head].cpu().numpy()
    
    # Decode tokens for labels
    src_tokens = [tokenizer_src.decode([i]) for i in source[0].cpu().numpy()]
    tgt_tokens = [tokenizer_tgt.decode([i]) for i in target[0].cpu().numpy()]
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(attention_weights, cmap='viridis')
    
    ax.set_xticks(range(len(src_tokens)))
    ax.set_yticks(range(len(tgt_tokens)))
    ax.set_xticklabels(src_tokens, rotation=90)
    ax.set_yticklabels(tgt_tokens)
    
    ax.set_xlabel('Source')
    ax.set_ylabel('Target')
    ax.set_title(f'Attention Weights - Layer {layer}, Head {head}')
    
    plt.colorbar(im)
    plt.tight_layout()
    plt.savefig(f'attention_layer{layer}_head{head}.png')
    plt.show()

# Use it
visualize_attention(model, source_tensor, target_tensor, 
                   tokenizer_src, tokenizer_tgt, layer=2, head=3)
```

### Attention Heatmap

Create a comprehensive attention heatmap:

```python
def plot_all_heads(model, source, target, layer=0):
    """Plot attention for all heads in a layer."""
    model.eval()
    
    with torch.no_grad():
        encoder_output = model.encode(source, source_mask)
        decoder_output = model.decode(encoder_output, source_mask, target, target_mask)
        
        # Get all attention heads
        attention = model.decoder.layers[layer].cross_attention_block.attention_scores
        attention = attention[0].cpu().numpy()  # Shape: (h, seq_len, seq_len)
    
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    
    for idx, ax in enumerate(axes.flat):
        if idx < attention.shape[0]:
            im = ax.imshow(attention[idx], cmap='viridis')
            ax.set_title(f'Head {idx}')
            plt.colorbar(im, ax=ax)
    
    plt.suptitle(f'All Attention Heads - Layer {layer}')
    plt.tight_layout()
    plt.savefig(f'all_heads_layer{layer}.png')
    plt.show()
```

---

## Advanced Configurations

### Mixed Precision Training

Use automatic mixed precision for faster training:

```python
from torch.cuda.amp import autocast, GradScaler

def train_with_mixed_precision(config):
    # Setup
    device = 'cuda'
    model = get_model(config, vocab_src_len, vocab_tgt_len).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    scaler = GradScaler()
    
    for epoch in range(config['num_epochs']):
        for batch in train_dataloader:
            optimizer.zero_grad()
            
            # Mixed precision forward pass
            with autocast():
                encoder_output = model.encode(encoder_input, encoder_mask)
                decoder_output = model.decode(encoder_output, encoder_mask, 
                                             decoder_input, decoder_mask)
                proj_output = model.project(decoder_output)
                loss = loss_fn(proj_output.view(-1, vocab_size), label.view(-1))
            
            # Scaled backward pass
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
```

### Gradient Accumulation

Train with effectively larger batch sizes:

```python
def train_with_gradient_accumulation(config, accumulation_steps=4):
    model = get_model(config, vocab_src_len, vocab_tgt_len)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    
    for epoch in range(config['num_epochs']):
        for i, batch in enumerate(train_dataloader):
            # Forward pass
            encoder_output = model.encode(encoder_input, encoder_mask)
            decoder_output = model.decode(encoder_output, encoder_mask, 
                                         decoder_input, decoder_mask)
            proj_output = model.project(decoder_output)
            
            # Scale loss by accumulation steps
            loss = loss_fn(proj_output.view(-1, vocab_size), label.view(-1))
            loss = loss / accumulation_steps
            
            # Backward pass
            loss.backward()
            
            # Update weights every accumulation_steps
            if (i + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
```

### Learning Rate Scheduling

Implement warmup and decay:

```python
class TransformerLRScheduler:
    def __init__(self, optimizer, d_model, warmup_steps=4000):
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.current_step = 0
    
    def step(self):
        self.current_step += 1
        lr = self.d_model ** (-0.5) * min(
            self.current_step ** (-0.5),
            self.current_step * self.warmup_steps ** (-1.5)
        )
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        return lr

# Use scheduler
optimizer = torch.optim.Adam(model.parameters(), lr=1)
scheduler = TransformerLRScheduler(optimizer, d_model=512, warmup_steps=4000)

for epoch in range(num_epochs):
    for batch in train_dataloader:
        # Training step
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        # Update learning rate
        current_lr = scheduler.step()
```

---

## Performance Optimization

### Inference Optimization

Optimize model for faster inference:

```python
# Convert to evaluation mode
model.eval()

# Disable gradient computation
torch.set_grad_enabled(False)

# Use torch.jit for faster inference
scripted_model = torch.jit.script(model)
scripted_model.save('model_scripted.pt')

# Load and use
loaded_model = torch.jit.load('model_scripted.pt')
output = loaded_model.encode(source, source_mask)
```

### Beam Search

Implement beam search for better translations:

```python
def beam_search_decode(model, source, source_mask, tokenizer_tgt, 
                       beam_size=5, max_len=512, device='cpu'):
    """Beam search decoding for better translation quality."""
    sos_idx = tokenizer_tgt.token_to_id('[SOS]')
    eos_idx = tokenizer_tgt.token_to_id('[EOS]')
    
    # Encode source once
    encoder_output = model.encode(source, source_mask)
    
    # Initialize beam
    beams = [(torch.tensor([[sos_idx]]).to(device), 0.0)]  # (sequence, score)
    
    for _ in range(max_len):
        new_beams = []
        
        for seq, score in beams:
            if seq[0, -1].item() == eos_idx:
                new_beams.append((seq, score))
                continue
            
            # Decode current sequence
            decoder_mask = causal_mask(seq.size(1)).to(device)
            out = model.decode(encoder_output, source_mask, seq, decoder_mask)
            prob = torch.log_softmax(model.project(out[:, -1]), dim=-1)
            
            # Get top k tokens
            topk_probs, topk_indices = torch.topk(prob, beam_size)
            
            for k in range(beam_size):
                new_seq = torch.cat([seq, topk_indices[0, k].unsqueeze(0).unsqueeze(0)], dim=1)
                new_score = score + topk_probs[0, k].item()
                new_beams.append((new_seq, new_score))
        
        # Keep top beam_size beams
        beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_size]
        
        # Check if all beams ended
        if all(seq[0, -1].item() == eos_idx for seq, _ in beams):
            break
    
    # Return best beam
    return beams[0][0]
```

---

## Troubleshooting Examples

### Debug Mode

Enable detailed debugging:

```python
import logging

logging.basicConfig(level=logging.DEBUG)

# Check model architecture
print(model)

# Check parameter counts
for name, param in model.named_parameters():
    print(f"{name}: {param.shape}, requires_grad={param.requires_grad}")

# Check gradients during training
for name, param in model.named_parameters():
    if param.grad is not None:
        print(f"{name} gradient norm: {param.grad.norm()}")
```

### Memory Profiling

Profile memory usage:

```python
import torch.cuda as cuda

if cuda.is_available():
    print(f"Allocated: {cuda.memory_allocated() / 1e9:.2f} GB")
    print(f"Reserved: {cuda.memory_reserved() / 1e9:.2f} GB")
    
    # Reset peak stats
    cuda.reset_peak_memory_stats()
    
    # Run training/inference
    # ...
    
    print(f"Peak allocated: {cuda.max_memory_allocated() / 1e9:.2f} GB")
```

---

## Additional Resources

- [Original Paper](https://arxiv.org/abs/1706.03762)
- [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [Hugging Face Datasets](https://huggingface.co/docs/datasets/)

---

[← Back to Home](index.md) | [Training Guide](training.md) | [API Reference](api.md)</div>
