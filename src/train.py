import os
import sys
import json
import torch

# Add project root directory to python path to resolve imports when running directly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig
from src.utils import load_config, setup_logging
from src.tokenizer import load_qwen_tokenizer
from src.dataset import load_raw_data, format_dataset_to_text

logger = setup_logging()

def main():
    # 1. Load Configurations
    config = load_config()
    model_cfg = config.get("model", {})
    quant_cfg = config.get("quantization", {})
    lora_cfg = config.get("lora", {})
    train_cfg = config.get("training", {})
    dataset_cfg = config.get("dataset", {})
    
    logger.info("Initializing Fine-Tuning Pipeline...")
    
    # 2. Setup Quantization Configuration (QLoRA) & Precision
    bnb_config = None
    use_fp16 = False
    use_bf16 = False
    
    if torch.cuda.is_available():
        if torch.cuda.is_bf16_supported():
            compute_dtype = torch.bfloat16
            use_bf16 = True
            logger.info("GPU supports BF16. Training in BFloat16 precision.")
        else:
            compute_dtype = torch.float16
            use_fp16 = True
            logger.info("GPU does not support BF16. Training in Float16 precision.")
            
        if quant_cfg.get("load_in_4bit", False):
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type=quant_cfg.get("bnb_4bit_quant_type", "nf4"),
                bnb_4bit_use_double_quant=quant_cfg.get("bnb_4bit_use_double_quant", True),
                bnb_4bit_compute_dtype=compute_dtype
            )
            logger.info("QLoRA 4-bit Quantization Configured")
    else:
        logger.warning("CUDA GPU not available; loading model in default precision on CPU.")
        
    # 3. Load Tokenizer & Model
    model_id = model_cfg.get("base_model_id", "Qwen/Qwen2.5-1.5B-Instruct")
    tokenizer = load_qwen_tokenizer(model_id)
    
    logger.info(f"Loading Base Model: {model_id}")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=model_cfg.get("trust_remote_code", True)
    )
    
    # 4. Prepare Model for Peft Training
    if bnb_config is not None:
        model = prepare_model_for_kbit_training(model)
        logger.info("Prepared model for k-bit quantized training.")
        
    peft_config = LoraConfig(
        r=lora_cfg.get("r", 16),
        lora_alpha=lora_cfg.get("lora_alpha", 32),
        lora_dropout=lora_cfg.get("lora_dropout", 0.05),
        target_modules=lora_cfg.get("target_modules", ["q_proj", "v_proj"]),
        bias=lora_cfg.get("bias", "none"),
        task_type=lora_cfg.get("task_type", "CAUSAL_LM")
    )
    
    # 5. Load Dataset
    dataset_name = dataset_cfg.get("name", "data/raw/sample.json")

    # Generate dummy data if using the default local sample path and it's missing
    if dataset_name == "data/raw/sample.json" and not os.path.exists(dataset_name):
        logger.info("Creating dummy dataset sample.json for testing...")
        os.makedirs(os.path.dirname(dataset_name), exist_ok=True)
        dummy_data = [
            {
                "messages": [
                    {"role": "system", "content": "You are a CS tutor."},
                    {"role": "user", "content": "What is binary search?"},
                    {"role": "assistant", "content": "Binary search is an O(log n) algorithm for finding elements in sorted lists."}
                ]
            },
            {
                "messages": [
                    {"role": "system", "content": "You are a CS tutor."},
                    {"role": "user", "content": "What is time complexity?"},
                    {"role": "assistant", "content": "Time complexity represents the compute steps an algorithm takes as input size grows."}
                ]
            }
        ]
        with open(dataset_name, 'w') as f:
            json.dump(dummy_data, f, indent=2)

    # If it looks like a local file path but doesn't exist, raise a clear error
    local_data_exts = (".json", ".jsonl", ".csv")
    if dataset_name.endswith(local_data_exts) and not os.path.exists(dataset_name):
        raise FileNotFoundError(
            f"Local dataset file not found: {dataset_name}. "
            "If using Hugging Face Hub, set dataset.name to a dataset ID "
            "like 'angrygiraffe/claude-opus-4.6-4.7-reasoning-8.7k'."
        )

    raw_dataset = load_raw_data(dataset_name)

    # Format the dataset to have a 'text' column with ChatML formatted strings
    # SFTTrainer will handle tokenization internally
    train_dataset = format_dataset_to_text(
        raw_dataset,
        text_field=dataset_cfg.get("text_field", "messages"),
        prompt_field=dataset_cfg.get("prompt_field"),
        response_field=dataset_cfg.get("response_field")
    )
    
    # 6. Configure SFT Training Arguments
    # Using SFTConfig (extends TrainingArguments) for full TRL compatibility
    output_dir = train_cfg.get("output_dir", "./outputs")
    if train_cfg.get("overwrite_output_dir", True) and os.path.exists(output_dir):
        import shutil
        logger.info(f"Cleaning existing output directory: {output_dir}")
        shutil.rmtree(output_dir, ignore_errors=True)

    # Dynamic optimizer check: bitsandbytes optimizers like paged_adamw_8bit require CUDA.
    # If on CPU, fall back to adamw_torch.
    optim_name = train_cfg.get("optim", "paged_adamw_8bit")
    if not torch.cuda.is_available():
        optim_name = "adamw_torch"
        logger.info("CUDA not available. Falling back to 'adamw_torch' optimizer for local CPU run.")

    sft_config = SFTConfig(
        output_dir=output_dir,
        max_length=train_cfg.get("max_seq_length", 1024), # Configured under max_length in SFTConfig
        num_train_epochs=train_cfg.get("num_train_epochs", 3),
        max_steps=train_cfg.get("max_steps", -1),
        per_device_train_batch_size=train_cfg.get("per_device_train_batch_size", 2),
        gradient_accumulation_steps=train_cfg.get("gradient_accumulation_steps", 4),
        learning_rate=float(train_cfg.get("learning_rate", 2e-4)),
        weight_decay=train_cfg.get("weight_decay", 0.01),
        optim=optim_name,
        lr_scheduler_type=train_cfg.get("lr_scheduler_type", "cosine"),
        warmup_ratio=train_cfg.get("warmup_ratio", 0.03),
        logging_steps=train_cfg.get("logging_steps", 10),
        save_strategy=train_cfg.get("save_strategy", "steps"),
        save_steps=train_cfg.get("save_steps", 50),
        fp16=use_fp16,
        bf16=use_bf16,
        use_cpu=not torch.cuda.is_available(), # Allow running training locally on CPU
        report_to="none",
    )
    
    # 7. SFT Trainer Initialize
    logger.info("Initializing Trainer...")
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        peft_config=peft_config,
        args=sft_config,
    )
    
    # 8. Run Training
    logger.info("Starting training loop...")
    trainer.train()
    
    # 9. Save Trained Adapter
    adapter_path = "./adapters/qwen-lora-adapter"
    logger.info(f"Training completed. Saving adapter weights to: {adapter_path}")
    os.makedirs(adapter_path, exist_ok=True)
    trainer.model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)
    logger.info("Adapter saved successfully. Fine-tuning complete!")

if __name__ == "__main__":
    main()
