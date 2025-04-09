# [Note] Ignore Repository Name

This repository contains implementations of four novel GPT-2 variants, each introducing unique architectural modifications to enhance performance and efficiency:

1. **[Drift-Diffusion GPT-2](docs/models/drift_diffusion_gpt2.md)**: Incorporates drift and diffusion mechanisms
2. **[Parallel GPT-2](docs/models/parallel_gpt2.md)**: Implements parallel layer processing with bottleneck methods
3. **[Duo-Predict GPT-2](docs/models/duo_predict_gpt2.md)**: Features alternating attention patterns for enhanced prediction
4. **[Rotating Head GPT-2](docs/models/rotating_head_gpt2.md)**: Uses head-specific rotary positional embeddings

## Experimental Setup

All models share common [experimental settings](docs/experimental_setup.md):

- Base Architecture: GPT-2 Medium
- Dataset: WikiText-103-raw-v1 (116M tokens)
- Training: 5 epochs on H100 GPU with BF16 precision
- Available on HuggingFace Hub under BluebrainAI organization

## Installation

1. Clone the repository:

   ```bash
   git clone <repository-url>
   cd parallel_transformer
   ```
2. Create a virtual environment (recommended):

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```
3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Model Architectures

### 1. Drift-Diffusion GPT-2

- Combines drift and diffusion mechanisms for enhanced token representation
- Applies drift for temporal evolution and diffusion for context mixing
- Each attention head can be configured with independent drift-diffusion parameters

### 2. Parallel GPT-2

- Splits model layers into parallel paths
- Implements mean and concat bottleneck methods
- Enables efficient multi-GPU distribution
- Requires even number of layers for balanced paths

### 3. Duo-Predict GPT-2

- Alternating attention patterns for odd and even positions
- Custom attention mask matrix for enhanced prediction
- Modified loss computation for dual predictions
- Sequence length limited to 512 tokens

### 4. Rotating Head GPT-2

- Head-specific rotary positional embeddings
- Two variants: Learnable Rotations (LR) and Geometric Progression (GP)
- Optional layer normalization for stability
- Efficient implementation with cached patterns

## Pre-trained Models

All models are available on HuggingFace Hub under the BluebrainAI organization:

1. Drift-Diffusion Variants:

   - `BluebrainAI/dd-gpt2-medium-wikitext`
2. Parallel Variants:

   - `BluebrainAI/parallel-gpt2-medium-wikitext`
   - `BluebrainAI/parallel-mean-bottleneck-gpt2-medium-wikitext`
3. Duo-Predict Variants:

   - `BluebrainAI/duo-predict-gpt2-medium-wikitext`
   - `BluebrainAI/duo-predict-loss-fixed-gpt2-medium-wikitext`
4. Rotating Head Variants:

   - `BluebrainAI/rotating-head-lr-gpt2-medium-wikitext`
   - `BluebrainAI/rotating-head-gp-gpt2-medium-wikitext`
   - `BluebrainAI/rotating-head-lr-norm-gpt2-medium-wikitext`
   - `BluebrainAI/rotating-head-gp-norm-gpt2-medium-wikitext`

## Usage

### Installation

1. Clone the repository:

   ```bash
   git clone <repository-url>
   cd parallel_transformer
   ```
2. Create a virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

### Training

1. Choose a configuration file from `src/configs/`
2. Run training:
   ```bash
   python train.py --config src/configs/MODEL_CONFIG.json
   ```

### Evaluation

1. For perplexity and accuracy metrics:

   ```bash
   python eval_model.py --config src/configs/MODEL_CONFIG.json
   ```

## Documentation

- [Experimental Setup](docs/experimental_setup.md)
- Model Documentation:
  - [Drift-Diffusion GPT-2](docs/models/drift_diffusion_gpt2.md)
  - [Parallel GPT-2](docs/models/parallel_gpt2.md)
  - [Duo-Predict GPT-2](docs/models/duo_predict_gpt2.md)
  - [Rotating Head GPT-2](docs/models/rotating_head_gpt2.md)

## Citation

If you use this code in your research, please cite our work:

```bibtex
@misc{parallel_transformer,
  author = {BluebrainAI},
  title = {Parallel Transformer Models},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/BluebrainAI/parallel_transformer}
}
```
