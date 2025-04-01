import json
import logging
import math
import nltk
import numpy as np
import os
import sys
import torch
from dataclasses import dataclass, field
from typing import Optional
from datasets import load_dataset
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    HfArgumentParser,
    default_data_collator,
    set_seed,
)
from src.data.tokenization.tokenized_hub import preprocess_and_tokenize_data
from src.models.modeling_parallel_gpt2 import ParallelGPT2LMHeadModel, ParallelGPT2Config
from dotenv import load_dotenv
load_dotenv()
# Set up logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

@dataclass
class ModelArguments:
    """Arguments pertaining to which model/config we are going to evaluate."""
    cache_dir: Optional[str] = field(
        default="./output/models/parallel-gpt2-medium-wikitext",
        metadata={"help": "The path to the local model directory."}
    )
    tokenizer_file: Optional[str] = field(
        default="gpt2-medium",
        metadata={"help": "The tokenizer to use for evaluation."}
    )
    use_auth_token: bool = field(
        default=False,
        metadata={"help": "Whether to use auth token for accessing private models."}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where to store downloaded models and datasets."}
    )

@dataclass
class DataEvalArguments:
    """Arguments pertaining to what data we are going to evaluate on."""
    dataset_path: Optional[str] = field(
        default="wikitext",
        metadata={"help": "The path of the dataset to use."}
    )
    dataset_name: Optional[str] = field(
        default="wikitext-103-raw-v1",
        metadata={"help": "The name of the dataset to use."}
    )
    max_seq_length: Optional[int] = field(
        default=1024,
        metadata={"help": "The maximum sequence length for the model."}
    )
    keep_in_memory: bool = field(
        default=False,
        metadata={"help": "Whether to keep the dataset in memory."}
    )
    streaming: bool = field(
        default=False,
        metadata={"help": "Whether to stream the dataset."}
    )

@dataclass
class EvalOutputArguments:
    """Arguments pertaining to evaluation output."""
    output_file: Optional[str] = field(
        default="evaluation_results.txt",
        metadata={"help": "Path to save the formatted evaluation results."}
    )

def create_compute_metrics(tokenizer):
    """Create a metrics computation function that uses the given tokenizer.
    
    Args:
        tokenizer: The tokenizer to use for converting token IDs to tokens.
        
    Returns:
        A function that computes metrics given an EvalPrediction object.
    """
    # Ensure NLTK punkt is available
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    
    def compute_metrics(eval_preds):
        """Compute evaluation metrics for the model.
        
        Args:
            eval_preds: An EvalPrediction object containing predictions and labels.
            
        Returns:
            A dictionary of metrics.
        """
        # Access predictions and labels from the EvalPrediction object
        logits = eval_preds.predictions  # Shape: [batch_size, seq_length, vocab_size]
        labels = eval_preds.label_ids    # Shape: [batch_size, seq_length]
        
        # Calculate loss manually if not available
        loss = None
        if hasattr(eval_preds, 'losses') and eval_preds.losses is not None:
            if isinstance(eval_preds.losses, np.ndarray) and eval_preds.losses.size > 0:
                loss = float(np.mean(eval_preds.losses))
        else:
            # Calculate cross-entropy loss manually
            # Convert logits and labels to torch tensors if they're numpy arrays
            if isinstance(logits, np.ndarray):
                logits_tensor = torch.tensor(logits)
            else:
                logits_tensor = logits
                
            if isinstance(labels, np.ndarray):
                labels_tensor = torch.tensor(labels)
            else:
                labels_tensor = labels
                
            # Create a loss function
            loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
            
            # Reshape for loss calculation
            if len(logits_tensor.shape) == 3:  # [batch_size, seq_length, vocab_size]
                # Shift logits and labels for causal language modeling
                shift_logits = logits_tensor[..., :-1, :].contiguous()
                shift_labels = labels_tensor[..., 1:].contiguous()
                
                # Reshape for loss calculation
                vocab_size = shift_logits.shape[-1]
                loss = loss_fct(shift_logits.view(-1, vocab_size), shift_labels.view(-1))
                loss = loss.item()  # Convert to Python scalar
        
        # For token classification tasks, we need the token-level predictions
        # Get the highest probability token at each position
        if len(logits.shape) == 3:  # [batch_size, seq_length, vocab_size]
            token_predictions = np.argmax(logits, axis=-1)  # [batch_size, seq_length]
        else:  # Already token predictions
            token_predictions = logits
        
        # Shift predictions and labels for language modeling
        # Predictions are for the next token, so we compare preds[:, :-1] with labels[:, 1:]
        shifted_preds = token_predictions[:, :-1]  # Remove last token prediction
        shifted_labels = labels[:, 1:]             # Remove first token (usually BOS)
        
        # Flatten for token-level metrics
        flat_preds = shifted_preds.reshape(-1)
        flat_labels = shifted_labels.reshape(-1)
        
        # Calculate accuracy
        accuracy = np.mean(flat_preds == flat_labels)
        
        # Initialize metrics dictionary
        metrics_dict = {'accuracy': float(accuracy)}
        
        # Calculate perplexity if loss is available
        if loss is not None:
            metrics_dict['perplexity'] = math.exp(loss)
            metrics_dict["loss"] = loss
        
        # Calculate BLEU score on a subset of examples
        # This is computationally expensive, so we limit the number of samples
        batch_size = token_predictions.shape[0]
        max_samples = min(10, batch_size)  # Limit to at most 10 samples
        
        if max_samples > 0:
            bleu_scores = []
            smoothing = SmoothingFunction().method1
            
            for i in range(max_samples):
                # Convert token IDs to actual tokens
                pred_tokens = tokenizer.convert_ids_to_tokens(token_predictions[i].tolist())
                label_tokens = tokenizer.convert_ids_to_tokens(labels[i].tolist())
                
                # Filter out padding tokens if needed
                if tokenizer.pad_token_id is not None:
                    pred_tokens = [t for t in pred_tokens if t != tokenizer.pad_token]
                    label_tokens = [t for t in label_tokens if t != tokenizer.pad_token]
                
                # Calculate BLEU score for this sample
                try:
                    sample_bleu = sentence_bleu([label_tokens], pred_tokens, smoothing_function=smoothing)
                    bleu_scores.append(sample_bleu)
                except Exception as e:
                    logger.warning(f"Error calculating BLEU score: {e}")
                    continue
            
            if bleu_scores:
                metrics_dict['bleu'] = float(np.mean(bleu_scores))
            else:
                metrics_dict['bleu'] = 0.0
        
        return metrics_dict
    
    return compute_metrics

def preprocess_logits_for_metrics(logits, labels):
    """Preprocess the logits before computing metrics.
    
    This function is called before computing metrics with the logits and labels.
    It allows us to modify the logits before they are used to compute metrics.
    
    Args:
        logits: The logits output by the model
        labels: The labels
        
    Returns:
        The processed logits
    """
    if isinstance(logits, tuple):
        # Some models return tuple of (logits, past_key_values)
        logits = logits[0]
    
    # For metrics like accuracy, we need the predicted token IDs
    return logits.argmax(dim=-1)

def load_and_evaluate_model(model_args, data_args, training_args):
    """Load a trained model from HuggingFace and evaluate it on the dataset."""
    # Set seed for reproducibility
    set_seed(training_args.seed)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_file)
    
    # Load model
    try:
        # Try loading from HuggingFace Hub first
        logger.info(f"Attempting to load model from HuggingFace Hub: {training_args.hub_model_id}")
        model = ParallelGPT2LMHeadModel.from_pretrained(
            training_args.hub_model_id,
            trust_remote_code=True,
            use_auth_token=model_args.use_auth_token
        )
    except Exception as e:
        logger.warning(f"Failed to load from HuggingFace Hub: {e}")
        # Fall back to local path
        logger.info(f"Attempting to load model from local path: {model_args.cache_dir}")
        model = ParallelGPT2LMHeadModel.from_pretrained(
            model_args.cache_dir,
            trust_remote_code=True
        )
    
    # Load dataset
    logger.info(f"Loading dataset {data_args.dataset_path}/{data_args.dataset_name}")
    tokenized_eval_dataset = load_dataset(
        f"BluebrainAI/{data_args.dataset_name}-seq{data_args.max_seq_length}-tokenized-grouped",
        token=os.environ.get("HF_TOKEN", None), 
        split="validation",
        keep_in_memory=data_args.keep_in_memory, 
        streaming=data_args.streaming,
        cache_dir=model_args.cache_dir
    )
    
    # Create compute_metrics function
    compute_metrics_fn = create_compute_metrics(tokenizer)
    
    # Data collator
    def data_collator(data):
        batch = default_data_collator(data)
        batch["labels"] = batch["input_ids"].clone()
        return batch
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        eval_dataset=tokenized_eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics_fn,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    )
    
    # Evaluate the model
    logger.info("*** Evaluating model ***")
    metrics = trainer.evaluate()
    
    # Calculate perplexity from loss
    if 'eval_loss' in metrics:
        metrics['eval_perplexity'] = math.exp(metrics['eval_loss'])
        logger.info(f"Calculated perplexity: {metrics['eval_perplexity']}")
    
    # Log and save metrics
    logger.info(f"Evaluation metrics: {metrics}")
    
    # Save metrics to file
    metrics_output_dir = os.path.join(model_args.cache_dir, "eval_metrics.json")
    with open(metrics_output_dir, "w") as f:
        json.dump(metrics, f, indent=4)
    
    logger.info(f"Metrics saved to {metrics_output_dir}")
    
    return metrics

def main():
    # Parse arguments using HfArgumentParser
    parser = HfArgumentParser((ModelArguments, DataEvalArguments, TrainingArguments, EvalOutputArguments))
    
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If a JSON file is provided, parse it
        model_args, data_args, training_args, output_args = parser.parse_json_file(
            json_file=sys.argv[1], allow_extra_keys=True
        )
        # Also load the raw config for any custom fields
        config_parse = json.load(open(sys.argv[1], "r"))
    else:
        # Otherwise, parse command line arguments
        model_args, data_args, training_args, output_args = parser.parse_args_into_dataclasses()
        config_parse = {}
    
    # Set up logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    
    # Set HF token from environment if not provided
    training_args.hub_token = os.environ.get("HF_TOKEN", None)
    
    # Create compute_metrics function
    compute_metrics_fn = create_compute_metrics(AutoTokenizer.from_pretrained(model_args.tokenizer_file))
    
    # Data collator
    def data_collator(data):
        batch = default_data_collator(data)
        batch["labels"] = batch["input_ids"].clone()
        return batch
    
    # Load and evaluate model
    metrics = load_and_evaluate_model(model_args, data_args, training_args)
    return metrics


if __name__ == "__main__":
    main()
