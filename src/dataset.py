import os
import json
from datasets import load_dataset, Dataset
from src.utils import setup_logging

logger = setup_logging()

def load_raw_data(data_path_or_hub_name):
    """
    Loads raw dataset from Hugging Face hub or local file path (JSON/JSONL/CSV).
    """
    logger.info(f"Loading dataset from: {data_path_or_hub_name}")
    
    if os.path.exists(data_path_or_hub_name):
        ext = os.path.splitext(data_path_or_hub_name)[-1].lower()
        if ext == '.json' or ext == '.jsonl':
            return load_dataset('json', data_files=data_path_or_hub_name)
        elif ext == '.csv':
            return load_dataset('csv', data_files=data_path_or_hub_name)
        else:
            raise ValueError(f"Unsupported file format: {ext}")
    else:
        # Assume it's a Hugging Face Hub dataset
        return load_dataset(data_path_or_hub_name)

def format_to_chatml(messages):
    """
    Standardizes a conversation list into Qwen ChatML string:
    <|im_start|>system
    ...<|im_end|>
    <|im_start|>user
    ...<|im_end|>
    <|im_start|>assistant
    ...<|im_end|>
    """
    formatted_text = ""
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        formatted_text += f"<|im_start|>{role}\n{content}<|im_end|>\n"
    return formatted_text

def preprocess_dataset(
    dataset,
    tokenizer,
    max_seq_length,
    text_field="messages",
    prompt_field=None,
    response_field=None,
):
    """
    Applies ChatML templates and tokenizes conversations.
    """
    def tokenize_function(examples):
        texts = []

        if text_field in examples:
            batch_values = examples[text_field]
            for value in batch_values:
                if isinstance(value, list):
                    texts.append(format_to_chatml(value))
                elif isinstance(value, str):
                    texts.append(value)
                else:
                    texts.append(str(value))
        elif prompt_field and response_field and prompt_field in examples and response_field in examples:
            prompts = examples[prompt_field]
            responses = examples[response_field]
            for prompt, response in zip(prompts, responses):
                texts.append(
                    f"<|im_start|>user\n{prompt}<|im_end|>\n"
                    f"<|im_start|>assistant\n{response}<|im_end|>\n"
                )
        else:
            available_fields = list(examples.keys())
            raise ValueError(
                "Unable to build training text. Provide dataset.text_field or "
                "dataset.prompt_field + dataset.response_field in config. "
                f"Available fields: {available_fields}"
            )

        tokenized = tokenizer(
            texts,
            truncation=True,
            max_length=max_seq_length,
            padding=False,  # Usually padded dynamically during batching by DataCollator
        )
        return tokenized

    return dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names if isinstance(dataset, Dataset) else dataset["train"].column_names
    )
