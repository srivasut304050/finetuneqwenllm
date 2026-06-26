import os
import logging
import yaml

def setup_logging(level=logging.INFO):
    """Sets up standard logger format."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    return logging.getLogger("qwen-finetuning")

def load_config(config_path="configs/qwen.yaml"):
    """Loads YAML configurations for training and model parameters."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at {config_path}")
        
    with open(config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
            return config
        except yaml.YAMLError as exc:
            raise ValueError(f"Error parsing YAML file: {exc}")
