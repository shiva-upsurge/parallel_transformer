# Model Documentation

This directory contains detailed documentation for each model architecture in the project.

## Available Models

1. [Base GPT-2](base_gpt2.md)
   - Standard GPT-2 implementation
   - Foundation for other variants

2. [Parallel GPT-2](parallel_gpt2.md)
   - Parallel processing capabilities
   - Multi-GPU support
   - Bottleneck methods for combining outputs

3. [Duo-Predict GPT-2](duo_predict_gpt2.md)
   - Custom attention mechanism
   - Alternating prediction pattern
   - Modified loss function

## Model Comparison

| Feature          | Base GPT-2   | Parallel GPT-2             | Duo-Predict GPT-2                |
| ---------------- | ------------ | -------------------------- | -------------------------------- |
| Architecture     | Sequential   | Parallel                   | Sequential with custom attention |
| GPU Support      | Single/Multi | Multi with parallelization | Single/Multi                     |
| Attention        | Standard     | Standard                   | Custom pattern                   |
| Special Features | -            | Bottleneck methods         | Alternating attention            |
| Loss Function    | Standard     | Standard                   | Modified for prediction          |

## Configuration Files

Each model has its own configuration class:

- `GPT2Config`: Base configuration
- `ParallelGPT2Config`: For parallel processing
- `DuoPredictGPT2Config`: For duo-predict model

All configurations extend the base GPT-2 configuration with model-specific parameters.
