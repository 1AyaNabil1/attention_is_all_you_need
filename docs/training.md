---
layout: default
title: Training Guide
---

<div class="content">

# Training Guide

Comprehensive guide for training the Transformer model on translation tasks.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Dataset Setup](#dataset-setup)
3. [Configuration](#configuration)
4. [Training Process](#training-process)
5. [Monitoring](#monitoring)
6. [Evaluation](#evaluation)
7. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### System Requirements

- Python 3.8 or higher
- CUDA-capable GPU (recommended) or Apple Silicon (MPS support)
- Minimum 8GB RAM (16GB recommended)
- ~10GB disk space for dataset and models

### Installation

```bash
# Clone the repository
git clone https://github.com/1AyaNabil1/attention_is_all_you_need.git
cd attention_is_all_you_need

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

Main packages:
- `torch>=2.0.0` - PyTorch framework
- `datasets` - Hugging Face datasets
- `tokenizers` - Fast tokenization
- `tqdm` - Progress bars
- `torchmetrics` - Evaluation metrics
- `tensorboard` - Training visualization

---

## Dataset Setup

The implementation uses the **OPUS Books** dataset for English-French translation.

### Automatic Download

The dataset is automatically downloaded when you run the training script:

```python
from datasets import load_dataset

ds_raw = load_dataset('opus_books', 'en-fr', split='train')
```

### Dataset Statistics

- **Language Pair**: English → French
- **Total Samples**: ~127,000+ parallel sentences
- **Domain**: Book translations
- **Split**: 90% training, 10% validation

### Data Preprocessing

The `BilingualDataset` class handles:

1. **Tokenization**: WordLevel tokenizer with special tokens
2. **Special Tokens**: [UNK], [PAD], [SOS], [EOS]
3. **Padding**: Sequences padded to fixed length (seq_len=350)
4. **Masking**: Padding and causal masks for attention

---

## Configuration

### Default Configuration

The default configuration is defined in `src/config.py`:

```python
def get_config():
    return {
        "batch_size": 8,           # Training batch size
        "num_epochs": 20,          # Number of epochs
        "lr": 10**-4,              # Learning rate (0.0001)
        "seq_len": 350,            # Maximum sequence length
        "d_model": 512,            # Model dimension
        "datasource": 'opus_books', # Dataset name
        "lang_src": "en",          # Source language
        "lang_tgt": "fr",          # Target language
        "model_folder": "weights", # Weights directory
        "model_basename": "tmodel_", # Model filename prefix
        "preload": "latest",       # Load latest checkpoint
        "tokenizer_file": "tokenizer_{0}.json",
        "experiment_name": "runs/tmodel"  # TensorBoard logs
    }
```

### Customizing Configuration

You can modify the configuration by editing `src/config.py` or creating a YAML config file:

```yaml
# configs/config.yaml
batch_size: 16
num_epochs: 30
lr: 0.0001
seq_len: 512
d_model: 512
lang_src: "en"
lang_tgt: "fr"
```

### Model Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `d_model` | 512 | Model dimension |
| `N` | 6 | Number of encoder/decoder layers |
| `h` | 8 | Number of attention heads |
| `d_ff` | 2048 | Feed-forward dimension |
| `dropout` | 0.1 | Dropout rate |
| `seq_len` | 350 | Max sequence length |

---

## Training Process

### Starting Training

```bash
# Navigate to source directory
cd src

# Start training with default config
python train.py
```

### Training Pipeline

The training script (`train.py`) performs the following steps:

#### 1. Device Selection

```python
device = "cuda" if torch.cuda.is_available() else \
         "mps" if torch.backends.mps.is_available() else "cpu"
```

Automatically selects:
- CUDA for NVIDIA GPUs
- MPS for Apple Silicon
- CPU as fallback

#### 2. Dataset Loading

```python
train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config)
```

- Loads OPUS Books dataset
- Builds or loads tokenizers
- Creates training and validation dataloaders

#### 3. Model Initialization

```python
model = get_model(config, 
                  tokenizer_src.get_vocab_size(), 
                  tokenizer_tgt.get_vocab_size()).to(device)
```

- Creates Transformer model
- Moves to appropriate device
- Initializes parameters with Xavier uniform

#### 4. Optimizer Setup

```python
optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], eps=1e-9)
```

Uses Adam optimizer with:
- Learning rate: 0.0001
- Epsilon: 1e-9 for numerical stability

#### 5. Loss Function

```python
loss_fn = nn.CrossEntropyLoss(
    ignore_index=tokenizer_src.token_to_id('[PAD]'),
    label_smoothing=0.1
).to(device)
```

Features:
- **Ignores padding**: Padding tokens don't contribute to loss
- **Label smoothing**: 0.1 for better generalization

#### 6. Training Loop

```python
for epoch in range(initial_epoch, config['num_epochs']):
    model.train()
    for batch in train_dataloader:
        # Forward pass
        encoder_output = model.encode(encoder_input, encoder_mask)
        decoder_output = model.decode(encoder_output, encoder_mask, 
                                     decoder_input, decoder_mask)
        proj_output = model.project(decoder_output)
        
        # Compute loss
        loss = loss_fn(proj_output.view(-1, vocab_size), label.view(-1))
        
        # Backward pass
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
```

### Checkpointing

Models are automatically saved after each epoch:

```python
torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'global_step': global_step
}, model_filename)
```

Checkpoint location: `opus_books_weights/tmodel_{epoch}.pt`

### Resume Training

To resume from the latest checkpoint:

```python
config['preload'] = 'latest'  # Default behavior
```

To resume from a specific epoch:

```python
config['preload'] = '05'  # Resume from epoch 5
```

---

## Monitoring

### TensorBoard Integration

The training process logs metrics to TensorBoard:

```bash
# Start TensorBoard (in a separate terminal)
tensorboard --logdir runs/tmodel

# Open browser to: http://localhost:6006
```

### Logged Metrics

#### Training Metrics

- **train loss**: Cross-entropy loss per batch
- **learning rate**: Current learning rate
- **global step**: Total training steps

#### Validation Metrics

- **validation CER**: Character Error Rate
- **validation WER**: Word Error Rate
- **validation BLEU**: BLEU score for translation quality

### Console Output

During training, you'll see progress bars showing:

```
Processing Epoch 05: 100%|████████| 1234/1234 [12:34<00:00, loss: 2.345]
--------------------------------------------------------------------
SOURCE: The cat sat on the mat.
TARGET: Le chat s'est assis sur le tapis.
PREDICTED: Le chat s'assit sur le tapis.
--------------------------------------------------------------------
```

---

## Evaluation

### Validation During Training

Validation runs automatically at the end of each epoch:

```python
run_validation(model, val_dataloader, tokenizer_src, tokenizer_tgt,
               config['seq_len'], device, print_msg, global_step, writer)
```

### Evaluation Metrics

#### 1. BLEU Score

Measures n-gram overlap between predicted and reference translations:

```python
metric = torchmetrics.BLEUScore()
bleu = metric(predicted, expected)
```

- **Range**: 0-1 (higher is better)
- **Industry Standard**: ~0.3-0.4 is good for this task

#### 2. Character Error Rate (CER)

Measures edit distance at character level:

```python
metric = torchmetrics.CharErrorRate()
cer = metric(predicted, expected)
```

- **Range**: 0-∞ (lower is better)
- **Good Performance**: < 0.2

#### 3. Word Error Rate (WER)

Measures edit distance at word level:

```python
metric = torchmetrics.WordErrorRate()
wer = metric(predicted, expected)
```

- **Range**: 0-∞ (lower is better)
- **Good Performance**: < 0.3

### Greedy Decoding

The validation uses greedy decoding for translation:

```python
def greedy_decode(model, source, source_mask, tokenizer_src, 
                  tokenizer_tgt, max_len, device):
    encoder_output = model.encode(source, source_mask)
    decoder_input = torch.empty(1, 1).fill_(sos_idx)
    
    while decoder_input.size(1) < max_len:
        decoder_mask = causal_mask(decoder_input.size(1))
        out = model.decode(encoder_output, source_mask, 
                          decoder_input, decoder_mask)
        prob = model.project(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        
        if next_word == eos_idx:
            break
            
        decoder_input = torch.cat([decoder_input, next_word.unsqueeze(0)], dim=1)
    
    return decoder_input
```

---

## Troubleshooting

### Common Issues

#### 1. Out of Memory (OOM)

**Symptom**: `RuntimeError: CUDA out of memory`

**Solutions**:
- Reduce `batch_size` in config (try 4 or 2)
- Reduce `seq_len` (try 256 or 128)
- Use gradient accumulation
- Enable mixed precision training

```python
# Reduce batch size
config['batch_size'] = 4
```

#### 2. Slow Training

**Symptom**: Training takes too long

**Solutions**:
- Ensure GPU is being used (check device)
- Reduce `seq_len` for faster batches
- Use fewer validation examples
- Consider using smaller model (N=4, h=4)

```python
# Check device
print(f"Using device: {device}")
```

#### 3. Poor Translation Quality

**Symptom**: High loss, poor BLEU score

**Solutions**:
- Train for more epochs (20-30)
- Adjust learning rate (try 5e-5 or 2e-4)
- Check tokenizer vocabulary size
- Ensure dataset is loading correctly
- Increase model size if underfitting

#### 4. Tokenizer Issues

**Symptom**: `KeyError` or tokenization errors

**Solutions**:
- Delete existing tokenizer files and retrain
- Check special tokens are defined correctly
- Verify dataset language codes

```bash
# Remove old tokenizers
rm tokenizer_*.json
```

#### 5. Validation Fails

**Symptom**: Errors during validation

**Solutions**:
- Ensure validation batch size is 1
- Check masking implementation
- Verify greedy decode logic

### Performance Tips

1. **Data Loading**: Use `num_workers` in DataLoader for faster data loading
2. **Mixed Precision**: Consider using AMP for faster training
3. **Gradient Checkpointing**: Save memory for larger models
4. **Distributed Training**: Use multiple GPUs with DDP

### Debugging

Enable verbose logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

Check model parameters:

```python
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")
```

---

## Advanced Training

### Learning Rate Scheduling

Implement warmup schedule as in the paper:

```python
def get_lr(step, d_model, warmup_steps=4000):
    return d_model**(-0.5) * min(step**(-0.5), step * warmup_steps**(-1.5))
```

### Beam Search Decoding

For better translation quality during inference:

```python
def beam_search_decode(model, source, beam_size=5):
    # Implement beam search for better translations
    pass
```

### Data Augmentation

- Back-translation
- Synthetic data generation
- Noise injection

---

## Expected Results

### Training Timeline

| Epoch | Train Loss | Val BLEU | Val CER | Val WER |
|-------|-----------|----------|---------|---------|
| 1 | 5.2 | 0.05 | 0.85 | 0.92 |
| 5 | 3.1 | 0.15 | 0.45 | 0.62 |
| 10 | 2.3 | 0.25 | 0.32 | 0.48 |
| 15 | 1.9 | 0.32 | 0.25 | 0.38 |
| 20 | 1.6 | 0.38 | 0.20 | 0.32 |

*Note: Results may vary based on hardware and configuration*

### Training Time

- **GPU (NVIDIA RTX 3080)**: ~2-3 hours for 20 epochs
- **Apple Silicon (M1/M2)**: ~4-6 hours for 20 epochs
- **CPU**: Not recommended (very slow)

---

## Next Steps

After successful training:

1. **Evaluate on test set**: Test model on unseen data
2. **Try different language pairs**: Experiment with other languages
3. **Implement beam search**: Improve translation quality
4. **Fine-tune hyperparameters**: Optimize for your use case
5. **Deploy model**: Create inference pipeline

---

## References

- [Training Tips from Original Paper](https://arxiv.org/abs/1706.03762)
- [PyTorch Training Best Practices](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html)
- [Effective Transformer Training](https://arxiv.org/abs/2002.04745)

---

[← Back to Home](index.md) | [Next: API Reference →](api.md)</div>
