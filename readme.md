# IGNORE THE REPONSE NAME:



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

## Model Architecture

### Attention Mechanisms

#### 1. Scaled Dot-Product Attention

The model implements scaled dot-product attention mechanism (`scaled_dot_product_attention`):

#### 2. Custom Attention Mask Matrix

The `create_attention_mask_matrix` function implements a specialized attention pattern:

- Creates a matrix for parallel prediction
- For odd-indexed rows: allows attention to even positions up to current position
- For even-indexed rows: allows attention to odd positions up to (i-2) plus current and next position
- Enables bidirectional context flow while maintaining parallel processing

Attention mask matrix pattern (1 indicates attention is allowed):
```
odd rows represents: self-attention
even rows represents: custom-attention to attend current token, previous mask token and all the previous tokens

Position:  1 2 3 4 5 6 7 8 
 Row 1:    1 1 0 0 0 0 0 0   
 Row 2:    1 1 1 0 0 0 0 0   
 Row 3:    1 1 1 1 0 0 0 0   
 Row 4:    1 0 1 1 1 0 0 0
 Row 5:    1 1 1 1 1 1 0 0   
 Row 6:    1 0 1 0 1 1 1 0   
 Row 7:    1 1 1 1 1 1 1 1   
 Row 8:    1 0 1 0 1 0 1 1   
```

This pattern ensures:
- Masked tokens (even positions) can see previous context
- Original tokens (odd positions) can attend to relevant masked tokens
- Maintains parallel processing capability while preserving contextual information flow

### Dataset : Masking Strategy

The model uses a special masking strategy implemented in `mask_sequences` function for pretraining:

- Takes input sequences and doubles their length to accommodate mask tokens
- Places original tokens at odd indices and mask tokens at even indices
- Only adds mask tokens before non-padding tokens
- Handles attention masks appropriately to maintain model's attention mechanism
- Uses special `<|MASK|>` token for masking

This masking approach enables the model to learn bidirectional context by predicting masked tokens during pretraining.


note: `config.max_position_embeddings` is without interleaved mask token

## Usage

### Running Pre-trained Model

To run the pre-trained model using a configuration file:

```bash
python pretrained.py --config src/configs/duo-predict-loss-fixed-gpt2-medium.json
```

The config file should contain the necessary parameters for model initialization and training.

### Evaluating the Model

To evaluate the model's performance:

```bash
python eval_model.py --config src/configs/duo-predict-loss-fixed-gpt2-medium.json
```

```yaml

```
