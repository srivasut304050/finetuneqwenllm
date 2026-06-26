import math
import torch
from torch.utils.data import DataLoader
from transformers import DataCollatorForLanguageModeling
from src.utils import load_config, setup_logging
from src.tokenizer import load_qwen_tokenizer
from src.dataset import load_raw_data, preprocess_dataset
from src.inference import load_inference_model

logger = setup_logging()

def evaluate_model(model, tokenizer, eval_dataset, batch_size=4):
    """
    Computes average loss and perplexity on the validation/test set.
    """
    model.eval()
    
    # We use HF standard DataCollator to pad batches dynamically
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    dataloader = DataLoader(eval_dataset, batch_size=batch_size, collate_fn=data_collator)
    
    total_loss = 0.0
    total_steps = 0
    
    logger.info("Starting model evaluation...")
    
    with torch.no_grad():
        for step, batch in enumerate(dataloader):
            # Move tensors to GPU if available
            inputs = {k: v.to(model.device) for k, v in batch.items()}
            
            outputs = model(**inputs)
            loss = outputs.loss
            
            total_loss += loss.item()
            total_steps += 1
            
            if (step + 1) % 10 == 0:
                logger.info(f"Step {step+1}/{len(dataloader)}")
                
    avg_loss = total_loss / max(total_steps, 1)
    perplexity = math.exp(avg_loss) if avg_loss < 20 else float('inf')
    
    return avg_loss, perplexity

def main():
    config = load_config()
    model_cfg = config.get("model", {})
    base_model_id = model_cfg.get("base_model_id", "Qwen/Qwen2.5-1.5B-Instruct")
    adapter_path = "./adapters/qwen-lora-adapter"
    
    # Load model and tokenizer
    model, tokenizer = load_inference_model(base_model_id, adapter_path)
    
    # Load validation data (using same sample for demonstration)
    dataset_name = config.get("dataset", {}).get("name", "data/raw/sample.json")
    raw_dataset = load_raw_data(dataset_name)
    processed_dataset = preprocess_dataset(
        raw_dataset, 
        tokenizer, 
        config.get("dataset", {}).get("max_length", 1024)
    )
    
    eval_set = processed_dataset["train"] if "train" in processed_dataset else processed_dataset
    
    loss, perplexity = evaluate_model(model, tokenizer, eval_set)
    
    print("\n" + "="*50)
    print("Evaluation Metrics:")
    print(f"Average Loss: {loss:.4f}")
    print(f"Perplexity:   {perplexity:.4f}")
    print("="*50)

if __name__ == "__main__":
    main()
