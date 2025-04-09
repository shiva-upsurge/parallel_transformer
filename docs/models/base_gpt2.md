# Base GPT-2 Model

## Overview

The base GPT-2 model serves as the foundation for all variants in this project. It implements the standard GPT-2 architecture from Hugging Face's transformers library.

## Architecture Components

- **Transformer Decoder**: Standard transformer decoder architecture
- **Attention Mechanism**: Multi-head self-attention with causal masking
- **Position Embeddings**: Learned positional embeddings
- **Token Embeddings**: Standard token embeddings
- **Layer Normalization**: Applied before attention and feed-forward layers
- **Residual Connections**: Used throughout the network

## Key Classes

1. **GPT2Model**:
   - Base transformer model
   - Handles token and position embeddings
   - Manages attention and feed-forward layers

2. **GPT2Block**:
   - Individual transformer block
   - Contains self-attention and feed-forward layers
   - Implements residual connections and layer normalization

3. **GPT2Attention**:
   - Multi-head self-attention implementation
   - Supports causal masking
   - Handles past key/value states for generation

## Configuration

```python
from transformers import GPT2Config

config = GPT2Config(
    vocab_size=50257,
    n_positions=1024,
    n_embd=768,      # Hidden size
    n_layer=12,      # Number of layers
    n_head=12,       # Number of attention heads
    activation_function="gelu"
)
```

## Usage Example

```python
from transformers import GPT2Model
import torch

# Initialize model
model = GPT2Model(config)

# Forward pass
input_ids = torch.randint(0, 50257, (1, 128))
outputs = model(input_ids)

# Access hidden states
last_hidden_state = outputs.last_hidden_state  # Shape: [batch_size, seq_length, hidden_size]
```

## Model Capabilities

1. **Text Generation**:
   - Autoregressive generation
   - Support for different decoding strategies (greedy, beam search, sampling)

2. **Feature Extraction**:
   - Hidden state extraction
   - Attention pattern analysis

3. **Training**:
   - Supports both full fine-tuning and parameter-efficient methods
   - Gradient checkpointing for memory efficiency

## Performance Characteristics

- **Memory Usage**: O(n²) attention complexity
- **Sequential Processing**: Processes tokens sequentially during generation
- **GPU Support**: Can run on single or multiple GPUs
- **Training Efficiency**: Standard transformer training characteristics
