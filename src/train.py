import os
import sys
import torch

# Add project root directory to python path to resolve imports when running directly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
from src.utils import load_config, setup_logging
from src.tokenizer import load_qwen_tokenizer
from src.dataset import load_raw_data, preprocess_dataset

logger = setup_logging()

def main():
    # 1. Load Configurations
    config = load_config()
    model_cfg = config.get("model", {})
    quant_cfg = config.get("quantization", {})
    lora_cfg = config.get("lora", {})
    train_cfg = config.get("training", {})
    
    logger.info("Initializing Fine-Tuning Pipeline...")
    
    # 2. Setup Quantization Configuration (QLoRA)
    bnb_config = None
    if torch.cuda.is_available() and quant_cfg.get("load_in_4bit", False):
        compute_dtype = getattr(torch, quant_cfg.get("bnb_4bit_compute_dtype", "float16"))
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=quant_cfg.get("bnb_4bit_quant_type", "nf4"),
            bnb_4bit_use_double_quant=quant_cfg.get("bnb_4bit_use_double_quant", True),
            bnb_4bit_compute_dtype=compute_dtype
        )
        logger.info("QLoRA 4-bit Quantization Configured")
    else:
        logger.warn("CUDA GPU not available or quantization disabled; loading model in default precision.")
        
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
    dataset_cfg = config.get("dataset", {})
    dataset_name = dataset_cfg.get("name", "data/raw/sample.json")

    local_data_exts = (".json", ".jsonl", ".csv")
    if dataset_name.endswith(local_data_exts) and not os.path.exists(dataset_name):
        raise FileNotFoundError(
            f"Local dataset file not found: {dataset_name}. "
            "If using Hugging Face Hub, set dataset.name to a dataset ID "
            "like 'angrygiraffe/claude-opus-4.6-4.7-reasoning-8.7k'."
        )

    raw_dataset = load_raw_data(dataset_name)
    processed_dataset = preprocess_dataset(
        raw_dataset, 
        tokenizer, 
        dataset_cfg.get("max_length", 1024),
        text_field=dataset_cfg.get("text_field", "messages"),
        prompt_field=dataset_cfg.get("prompt_field"),
        response_field=dataset_cfg.get("response_field")
    )
    
    # 6. Configure Training Arguments
    output_dir = train_cfg.get("output_dir", "./outputs")
    if train_cfg.get("overwrite_output_dir", True) and os.path.exists(output_dir):
        import shutil
        logger.info(f"Cleaning existing output directory: {output_dir}")
        shutil.rmtree(output_dir, ignore_errors=True)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=train_cfg.get("num_train_epochs", 3),
        max_steps=train_cfg.get("max_steps", -1), # Allows capping training at specific steps
        per_device_train_batch_size=train_cfg.get("per_device_train_batch_size", 2),
        gradient_accumulation_steps=train_cfg.get("gradient_accumulation_steps", 4),
        learning_rate=float(train_cfg.get("learning_rate", 2e-4)),
        weight_decay=train_cfg.get("weight_decay", 0.01),
        optim=train_cfg.get("optim", "paged_adamw_8bit"),
        lr_scheduler_type=train_cfg.get("lr_scheduler_type", "cosine"),
        logging_steps=train_cfg.get("logging_steps", 10),
        save_strategy=train_cfg.get("save_strategy", "steps"),
        save_steps=train_cfg.get("save_steps", 50),
        fp16=train_cfg.get("fp16", True) if torch.cuda.is_available() else False,
        bf16=train_cfg.get("bf16", False) if torch.cuda.is_available() else False,
        report_to="none" # Disable W&B logging for simplicity unless configured
    )
    
    # 7. SFT Trainer Initialize
    logger.info("Initializing Trainer...")
    trainer = SFTTrainer(
        model=model,
        train_dataset=processed_dataset["train"] if "train" in processed_dataset else processed_dataset,
        peft_config=peft_config,
        dataset_text_field="input_ids", # Pre-tokenized
        args=training_args
    )
    
    # 8. Run Training
    logger.info("Starting training loop...")
    trainer.train()
    
    # 9. Save Trained Adapter
    adapter_path = "./adapters/qwen-lora-adapter"
    logger.info(f"Training completed. Saving adapter weights to: {adapter_path}")
    trainer.model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)
    logger.info("Adapter saved successfully. Fine-tuning complete!")

if __name__ == "__main__":
    main()
