# Experimental Setup

This document outlines the common experimental settings used across all model variants in this repository.

## Base Architecture

All experiments are built upon the GPT-2 Medium architecture with the following specifications:
- Base Model: `gpt2-medium`
- Tokenizer: `gpt2-medium`
- Model Parameters: Same as GPT-2 Medium

## Dataset Details

### Source
- **Dataset**: WikiText-103-raw-v1
- **Type**: Large-scale language modeling corpus
- **Total Training Tokens**: 116,739,072

### Pre-processed Versions
Two versions of the tokenized and grouped dataset are available on Hugging Face:

1. **1024 Sequence Length** (Used by Drift-Diffusion, Parallel, and Rotating Head models)
   - Dataset: [BluebrainAI/wikitext-103-raw-v1-seq1024-tokenized-grouped](https://huggingface.co/datasets/BluebrainAI/wikitext-103-raw-v1-seq1024-tokenized-grouped)
   - Maximum sequence length: 1024 tokens

2. **512 Sequence Length** (Used by Duo-Predict model)
   - Dataset: [BluebrainAI/wikitext-103-raw-v1-seq512-tokenized-grouped](https://huggingface.co/datasets/BluebrainAI/wikitext-103-raw-v1-seq512-tokenized-grouped)
   - Maximum sequence length: 512 tokens

## Training Configuration

### Hardware Setup
- **GPU**: 1x NVIDIA H100
- **Precision**: BF16 (Brain Floating Point)

### Training Parameters
```json
{
    "per_device_train_batch_size": 64,
    "num_train_epochs": 5,
    "learning_rate": 1e-4,
    "lr_scheduler_type": "linear",
    "warmup_ratio": 0.1,
    "bf16": true
}
```

### Common Settings
- All models use the same learning rate schedule
- Training monitored using Weights & Biases
- Evaluation performed every 500 steps
- Models pushed to Hugging Face Hub after training

## Model Variants

1. **Drift-Diffusion GPT-2**
   - Sequence Length: 1024
   - [Documentation](models/drift_diffusion_gpt2.md)

2. **Parallel GPT-2**
   - Sequence Length: 1024
   - [Documentation](models/parallel_gpt2.md)

3. **Duo-Predict GPT-2**
   - Sequence Length: 512
   - [Documentation](models/duo_predict_gpt2.md)

4. **Rotating Head GPT-2**
   - Sequence Length: 1024
   - [Documentation](models/rotating_head_gpt2.md)

Each model variant introduces unique architectural modifications while maintaining these base experimental settings for fair comparison.
