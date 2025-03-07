from itertools import chain
from transformers.testing_utils import CaptureLogger
import transformers
from datasets import DatasetDict
from transformers import GPT2TokenizerFast, PreTrainedTokenizerFast
import os
from datasets import load_dataset
from dotenv import load_dotenv
load_dotenv()


def preprocess_and_tokenize_data(training_args, data_args, config_dataset, model_args):
    hub_dataset_name = f"BluebrainAI/{config_dataset['name']}-seq{model_args.max_seq_length}-tokenized-grouped"
    columns_to_remove = []
    raw_datasets = DatasetDict()
    config_dataset.pop("split", None)  # Remove "split" if present
    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
    }

    tokenizer = GPT2TokenizerFast.from_pretrained(
        "gpt2-medium",
        model_max_length=model_args.max_seq_length,
        **tokenizer_kwargs
    )
    if not isinstance(tokenizer, PreTrainedTokenizerFast):
        raise Exception(
            "The tokenizer should inherit from PreTrainedTokenizerFast")
    if data_args.max_train_samples is not None:
        train_split = f"train[:{data_args.max_train_samples}]"
        val_split = f"train[:{data_args.max_train_samples}]"
    else:
        val_split = "validation"
        train_split = "train"
    if not data_args.overwrite_cache:
        print("Loading Dataset from Huggingface Hub!")
        raw_datasets["train"] = load_dataset(
            hub_dataset_name, split=train_split, token=os.environ.get("HF_TOKEN", None), keep_in_memory=data_args.keep_in_memory, streaming=data_args.streaming)
        raw_datasets["validation"] = load_dataset(
            hub_dataset_name, split=val_split, token=os.environ.get("HF_TOKEN", None), keep_in_memory=data_args.keep_in_memory, streaming=data_args.streaming)

        # def create_labels(examples):
        #     examples["labels"] = examples["input_ids"].copy()
        #     return examples
        # raw_datasets = raw_datasets.map(
        #     create_labels,
        #     batched=True,
        #     batch_size=data_args.batch_size,
        #     load_from_cache_file=not data_args.overwrite_cache,
        #     desc="Running labelling on dataset",
        # )

        return tokenizer, raw_datasets
    else:
        print(
            f"Creating and saving Dataset to Huggingface Hub! - {hub_dataset_name}")
        # 1. Load the dataset normally (this might load metadata into memory, but not all text)
        if data_args.max_train_samples is not None:
            train_split = f"train[:{data_args.max_train_samples}]"
            val_split = f"train[:{data_args.max_train_samples}]"
        else:
            train_split = f"train[{data_args.validation_split_percentage}%:]"
            val_split = f"train[:{data_args.validation_split_percentage}%]"
        raw_datasets["validation"] = load_dataset(
            split=val_split,
            **config_dataset
        )
        raw_datasets["train"] = load_dataset(
            split=train_split,
            **config_dataset
        )
        # for key in raw_datasets.keys():
        #     raw_datasets[key] = raw_datasets[key].remove_columns(
        #         columns_to_remove)

        # First we tokenize all the texts.
        if training_args.do_train:
            if data_args.streaming:
                dataset_head = raw_datasets["train"].take(3)
                print(list(dataset_head))
                column_names = list(list(dataset_head)[0].keys())
            else:
                column_names = list(raw_datasets["train"].features)
        else:
            if data_args.streaming:
                dataset_head = raw_datasets["validation"].take(3)
                column_names = list(list(dataset_head)[0].keys())
            else:
                column_names = list(raw_datasets["validation"].features)
        print(column_names)
        text_column_name = "text" if "text" in column_names else column_names[0]

        # since this will be pickled to avoid _LazyModule error in Hasher force logger loading before tokenize_function
        tok_logger = transformers.utils.logging.get_logger(
            "transformers.tokenization_utils_base")

        def tokenize_function(examples):
            with CaptureLogger(tok_logger) as cl:
                output = tokenizer(
                    [item for item in examples[text_column_name]])
            return output
        print("Start Tokenization --")
        with training_args.main_process_first(desc="dataset map tokenization"):
            if not data_args.streaming:
                tokenized_datasets = raw_datasets.map(
                    tokenize_function,
                    batched=True,
                    num_proc=data_args.preprocessing_num_workers,
                    remove_columns=column_names,
                    load_from_cache_file=True,
                    desc="Running tokenizer on dataset",
                )
            else:
                tokenized_datasets = raw_datasets.map(
                    tokenize_function,
                    batched=True,
                    remove_columns=column_names,
                    load_from_cache_file=True,
                    batch_size=data_args.batch_size,
                )

        block_size = min(data_args.block_size, tokenizer.model_max_length)

        # Main data processing function that will concatenate all texts from our dataset and generate chunks of block_size.
        def group_texts(examples):
            # Concatenate all texts.
            concatenated_examples = {
                k: list(chain(*examples[k])) for k in examples.keys()}
            # concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
            total_length = len(concatenated_examples[list(examples.keys())[0]])
            # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
            # customize this part to your needs.
            if total_length >= block_size:
                total_length = (total_length // block_size) * block_size
            # Split by chunks of max_len.
            result = {
                k: [t[i: i + block_size]
                    for i in range(0, total_length, block_size)]
                for k, t in concatenated_examples.items()
            }
            
            return result

        # Note that with `batched=True`, this map processes 1,000 texts together, so group_texts throws away a remainder
        # for each of those groups of 1,000 texts. You can adjust that batch_size here but a higher value might be slower
        # to preprocess.
        #
        # To speed up this part, we use multiprocessing. See the documentation of the map method for more information:
        # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.map
        print("Start grouping --")
        with training_args.main_process_first(desc="grouping texts together"):
            if not data_args.streaming:
                lm_datasets = tokenized_datasets.map(
                    group_texts,
                    batched=True,
                    num_proc=data_args.preprocessing_num_workers,
                    load_from_cache_file=not data_args.overwrite_cache,
                    desc=f"Grouping texts in chunks of {block_size}",
                    batch_size=data_args.batch_size
                )
            else:
                lm_datasets = tokenized_datasets.map(
                    group_texts,
                    batched=True,
                    batch_size=data_args.batch_size,
                    load_from_cache_file=True,
                    # cache_file_name="./output/cache.arrow"
                    # batch_size=4,
                )
        if data_args.max_train_samples is None:  # intentional False, never overwrite
            # Push the processed dataset to the Hugging Face Hub
            lm_datasets.push_to_hub(repo_id=hub_dataset_name,
                                    max_shard_size="5GB", private=False, token=os.environ.get("HF_TOKEN", None))
        return tokenizer, lm_datasets
