from transformers import AutoTokenizer
from src.utils import setup_logging

logger = setup_logging()

def load_qwen_tokenizer(model_id):
    """
    Loads the tokenizer associated with the Qwen model.
    """
    logger.info(f"Loading tokenizer for model: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True,
        padding_side="right" # Right padding is standard for autoregressive generation fine-tuning
    )
    
    # Qwen models usually have an EOS token. Let's make sure PAD token is defined.
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
            logger.info("Set pad_token to eos_token")
        else:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            logger.info("Added new pad_token [PAD]")
            
    return tokenizer
