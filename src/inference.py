import os
import sys
import torch

# Add project root directory to python path to resolve imports when running directly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from src.utils import load_config, setup_logging

logger = setup_logging()

def load_inference_model(base_model_id, adapter_path=None):
    """
    Loads base model and optionall overlays the LoRA adapter weights.
    """
    logger.info(f"Loading tokenizer: {base_model_id}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
    
    logger.info(f"Loading base model: {base_model_id}")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True
    )
    
    if adapter_path and os.path.exists(adapter_path):
        logger.info(f"Overlaying LoRA adapter weights from: {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)
    else:
        logger.info("No adapter path found or provided. Running base model inference.")
        
    return model, tokenizer

def generate_response(model, tokenizer, prompt, max_new_tokens=256):
    """
    Constructs ChatML format, tokenizes it, runs forward pass, and decodes.
    """
    formatted_prompt = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    inputs = tokenizer(formatted_prompt, return_tensors="pt")
    
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}
        
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            do_sample=True,
            temperature=0.7,
            top_p=0.9
        )
        
    # Extract only the newly generated tokens
    input_length = inputs["input_ids"].shape[1]
    generated_tokens = outputs[0][input_length:]
    return tokenizer.decode(generated_tokens, skip_special_tokens=True)

def main():
    config = load_config()
    model_cfg = config.get("model", {})
    base_model_id = model_cfg.get("base_model_id", "Qwen/Qwen2.5-1.5B-Instruct")
    adapter_path = "./adapters/qwen-lora-adapter"
    
    logger.info("Initializing inference engine...")
    model, tokenizer = load_inference_model(base_model_id, adapter_path)
    
    print("\n" + "="*50)
    print("Qwen Chat Inference (Type 'quit' to exit)")
    print("="*50)
    
    while True:
        try:
            user_input = input("\nYou: ")
            if user_input.strip().lower() in ['quit', 'exit']:
                break
                
            if not user_input.strip():
                continue
                
            print("Assistant: ", end="", flush=True)
            response = generate_response(model, tokenizer, user_input)
            print(response)
            
        except KeyboardInterrupt:
            break
            
    print("\nExited inference.")

if __name__ == "__main__":
    main()
