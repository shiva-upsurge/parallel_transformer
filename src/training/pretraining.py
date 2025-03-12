from src.data.tokenization.tokenized_hub import preprocess_and_tokenize_data
from transformers.utils.versions import require_version
from transformers.utils import check_min_version, send_example_telemetry
from transformers.trainer_utils import get_last_checkpoint
import logging
import math
import os
import sys
from dataclasses import dataclass, field
from typing import Optional
import functools

import datasets
import evaluate
import torch
import transformers
from transformers import (
    MODEL_FOR_CAUSAL_LM_MAPPING,
    LlamaConfig,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
    default_data_collator,
    EarlyStoppingCallback
)
import json
from src.models.modeling_gpt2 import GPT2LMHeadModel, GPT2Config
from src.models.modeling_parallel_gpt2 import ParallelGPT2LMHeadModel, ParallelGPT2Config
from src.models.modeling_dd_gpt2 import DDGPT2LMHeadModel, DDGPT2Config
from src.models.modeling_rotating_head_gpt2 import RotatingHeadGPT2LMHeadModel, RotatingHeadGPT2Config

from dotenv import load_dotenv
load_dotenv()
import os

MODEL_CONFIG_CLASSES = list(MODEL_FOR_CAUSAL_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization. Don't set if you want to train a model from scratch."
            )
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={
            "help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)
        },
    )
    config_overrides: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override some existing default config settings when a model is trained from scratch. Example: "
                "n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index"
            )
        },
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_file: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer file path"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Where do you want to store the pretrained models downloaded from huggingface.co"
        },
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={
            "help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."
        },
    )
    model_revision: str = field(
        default="main",
        metadata={
            "help": "The specific model version to use (can be a branch name, tag name or commit id)."
        },
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `huggingface-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )
    torch_dtype: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override the default `torch.dtype` and load the model under this dtype. If `auto` is passed, the "
                "dtype will be automatically derived from the model's weights."
            ),
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )
    max_seq_length: Optional[int] = field(
        default=1024,
        metadata={
            "help": (
                "Optional input sequence length after tokenization. "
                "The training dataset will be truncated in block of this size for training. "
                "Default to the model max input length for single sentence inputs (take into account special tokens)."
            )
        },
    )

    apply_drift: bool = field(default=True, metadata={
                             "help": "Whether to apply drift attention."})
    apply_diffusion: bool = field(default=False, metadata={
                                 "help": "Whether to apply diffusion."})
    baseline_each_head: bool = field(default=True, metadata={
                                "help": "Whether to include a baseline."})
    
    bottleneck_method: str = field(default="mean", metadata={
        "help": "The method to use for the bottleneck. "
    })

    llama_model_config: dict = field(
        default=None, metadata={"help": "Llama model configuration."})

    def __post_init__(self):
        if self.config_overrides is not None and (self.config_name is not None or self.model_name_or_path is not None):
            raise ValueError(
                "--config_overrides can't be used in combination with --config_name or --model_name_or_path"
            )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    streaming: bool = field(default=False, metadata={
                            "help": "Enable streaming mode"})
    block_size: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Optional input sequence length after tokenization. "
                "The training dataset will be truncated in block of this size for training. "
                "Default to the model max input length for single sentence inputs (take into account special tokens)."
            )
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    validation_split_percentage: Optional[int] = field(
        default=5,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None, metadata={"help": "The number of processes to use for the preprocessing."}
    )
    batch_size: Optional[int] = field(
        default=6000,
        metadata={
            "help": (
                "Batch size per GPU/TPU core/CPU for training."
            )
        },
    )
    keep_in_memory: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether or not to keep the dataset in memory."
            )
        },
    )

    def __post_init__(self):
        if self.streaming:
            require_version("datasets>=2.0.0",
                            "The streaming feature requires `datasets>=2.0.0`")


logger = logging.getLogger(__name__)

# --- GPU logging decorator ---


def log_gpu_usage(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if torch.cuda.is_available():
            allocated_before = torch.cuda.memory_allocated()
            reserved_before = torch.cuda.memory_reserved()
            logger.info(
                f"[{func.__name__}] GPU usage before: allocated {allocated_before/1e6:.2f} MB, reserved {reserved_before/1e6:.2f} MB"
            )
        else:
            logger.info(f"[{func.__name__}] No GPU available.")
        result = func(*args, **kwargs)
        if torch.cuda.is_available():
            allocated_after = torch.cuda.memory_allocated()
            reserved_after = torch.cuda.memory_reserved()
            logger.info(
                f"[{func.__name__}] GPU usage after: allocated {allocated_after/1e6:.2f} MB, reserved {reserved_after/1e6:.2f} MB"
            )
        else:
            logger.info(f"[{func.__name__}] No GPU available.")
        return result
    return wrapper


def main():
    # Parse arguments
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(
            sys.argv[1], allow_extra_keys=True)
        config_parse = json.load(open(sys.argv[1], "r"))
    else:
        raise Exception("Provide training config file.")

    # Send telemetry and check token
    send_example_telemetry("run_clm", model_args, data_args)
    training_args.hub_token = os.environ.get("HF_TOKEN", None)
    if training_args.hub_token is None:
        raise Exception("No token key is provided!")

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    if training_args.should_log:
        transformers.utils.logging.set_verbosity_info()
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f" distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)



    @log_gpu_usage
    def preprocess_data(training_args, data_args, config_parse, model_args):
        print(training_args.local_rank, 'start load tokenizer')
        tokenizer, lm_datasets = preprocess_and_tokenize_data(
            training_args,
            data_args,
            config_dataset=config_parse["dataset"].copy(),
            model_args=model_args,
        )
        print(training_args.local_rank, 'end load tokenizer')
        return tokenizer, lm_datasets

    def train_model(trainer, training_args, data_args, train_dataset, last_checkpoint):
        print(training_args.local_rank, 'start train')
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
        return train_result

    def evaluate_model(trainer, training_args, data_args, eval_dataset):
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()
        max_eval_samples = (
            data_args.max_eval_samples if data_args.max_eval_samples is not None else len(
                eval_dataset)
        )
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))
        try:
            perplexity = math.exp(metrics["eval_loss"])
        except OverflowError:
            perplexity = float("inf")
        metrics["perplexity"] = perplexity

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
        return metrics

    # --- GPU-logged helper functions ---
    @log_gpu_usage
    def load_model(training_args, model_call, config):
        print(training_args.local_rank, 'start load model')
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = model_call(config=config)
        model.to(device)
        n_params = sum({p.data_ptr(): p.numel()
                       for p in model.parameters()}.values())
        logger.info(
            f"Training new model from scratch - Total size={n_params/2**20:.2f}M params")
        print(training_args.local_rank, 'end load model')
        return model

    # --- End GPU-logged helper functions ---
    config_kwargs = {
        "cache_dir": model_args.cache_dir
    }
    if model_args.model_type=="gpt2":
        config = GPT2Config.from_pretrained("gpt2-medium", **config_kwargs)
        config.use_cache = False
        model = load_model(training_args, GPT2LMHeadModel, config)
    elif model_args.model_type == "parallel-gpt2":
        config = ParallelGPT2Config.from_pretrained("gpt2-medium", architectures=["ParallelGPT2LMHeadModel"], **config_kwargs)
        ParallelGPT2Config.register_for_auto_class()
        ParallelGPT2LMHeadModel.register_for_auto_class("AutoModel")
        config.model_type = model_args.model_type
        config.bottleneck_method = getattr(model_args, "bottleneck_method", "mean")
        config.use_cache = False
        model = load_model(training_args, ParallelGPT2LMHeadModel, config)
    elif model_args.model_type == "dd-gpt2":
        config = DDGPT2Config.from_pretrained("gpt2-medium", architectures=["DDGPT2LMHeadModel"], **config_kwargs)
        DDGPT2Config.register_for_auto_class()
        DDGPT2LMHeadModel.register_for_auto_class("AutoModel")
        config.model_type = model_args.model_type
        config.apply_drift = model_args.apply_drift
        config.apply_diffusion = model_args.apply_diffusion
        config.baseline_each_head = model_args.baseline_each_head
        config.use_cache = False
        model = load_model(training_args, DDGPT2LMHeadModel, config)
    elif model_args.model_type == "rotating-head-gpt2":
        config = RotatingHeadGPT2Config.from_pretrained("gpt2-medium", architectures=["RotatingHeadGPT2LMHeadModel"], **config_kwargs)
        RotatingHeadGPT2Config.register_for_auto_class()
        RotatingHeadGPT2LMHeadModel.register_for_auto_class("AutoModel")
        config.model_type = model_args.model_type
        config.rotatinghead = getattr(model_args, "rotatinghead", "lr")
        config.use_cache = False
        model = load_model(training_args, RotatingHeadGPT2LMHeadModel, config)
    else:
        raise ValueError(f"Unknown model type: {model_args.model_type}")

    tokenizer, lm_datasets = preprocess_data(training_args, data_args, config_parse, model_args)

    # Resize embeddings if needed.
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) != embedding_size:
        model.resize_token_embeddings(len(tokenizer))

    # Select training dataset.
    print(training_args.local_rank, 'start select train_dataset')
    if training_args.do_train:
        if "train" not in lm_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = lm_datasets["train"]
        if data_args.max_train_samples is not None and not data_args.streaming:
            max_train_samples = min(
                len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))
    print(training_args.local_rank, 'end select train_dataset')

    if training_args.do_eval:
        if "validation" not in lm_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        print(training_args.local_rank, 'start select eval_dataset')
        eval_dataset = lm_datasets["validation"]
        if data_args.max_eval_samples is not None and not data_args.streaming:
            max_eval_samples = min(
                len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))
        print(training_args.local_rank, 'end select eval_dataset')

        def preprocess_logits_for_metrics(logits, labels):
            if isinstance(logits, tuple):
                logits = logits[0]
            return logits.argmax(dim=-1)
        print(training_args.local_rank, 'start load metric')
        metric = evaluate.load("accuracy")
        print(training_args.local_rank, 'end load metric')

        # Import necessary libraries at the module level
        import numpy as np
        import math
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
        import nltk

        # Download nltk data if not already present
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        
        def create_compute_metrics(tokenizer):
            """Create a metrics computation function that uses the given tokenizer.
            
            Args:
                tokenizer: The tokenizer to use for converting token IDs to tokens.
                
            Returns:
                A function that computes metrics given an EvalPrediction object.
            """
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
                
                # Get loss if available
                loss = None
                if hasattr(eval_preds, 'losses') and eval_preds.losses is not None:
                    if isinstance(eval_preds.losses, np.ndarray) and eval_preds.losses.size > 0:
                        loss = float(np.mean(eval_preds.losses))
                
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
                accuracy_metric = metric.compute(predictions=flat_preds, references=flat_labels)
                
                # Initialize metrics dictionary
                metrics_dict = {}
                
                # Add accuracy
                if isinstance(accuracy_metric, dict):
                    metrics_dict.update(accuracy_metric)
                else:
                    metrics_dict['accuracy'] = accuracy_metric
                
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
                            print(f"Error calculating BLEU score: {e}")
                            continue
                    
                    if bleu_scores:
                        metrics_dict['bleu'] = float(np.mean(bleu_scores))
                    else:
                        metrics_dict['bleu'] = 0.0
                
                return metrics_dict
            
            return compute_metrics
        
        # Create the compute_metrics function with the tokenizer
        compute_metrics = create_compute_metrics(tokenizer)

    print(training_args.local_rank, 'Initialize our Trainer')

    def data_collator(data):
        batch = default_data_collator(data)
        batch["labels"] = batch["input_ids"].clone()
        return batch

    early_stopping_callback = EarlyStoppingCallback(
        early_stopping_patience=2,    # Number of evaluations with no improvement after which training will be stopped
        early_stopping_threshold=0.0  # Minimum change to qualify as an improvement
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics if training_args.do_eval else None,
        callbacks=[early_stopping_callback],
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    )

    if training_args.do_train:
        train_result = train_model(
            trainer, training_args, data_args, train_dataset, last_checkpoint)

    if training_args.do_eval:
        evaluate_model(trainer, training_args, data_args, eval_dataset)


if __name__ == "__main__":
    main()
    # torchrun --nproc_per_node=2 ./src/training/pretraining.py ./src/config/gpu1test.json
