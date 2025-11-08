"""
Local Model Training Script
For Google Colab free tier - GPT-2 Medium or Phi-3-Mini fine-tuning
"""

import os
import json
import time
from typing import Optional, List, Dict, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train_gpt2_model(
    training_data_path: str,
    output_dir: str = "./models",
    model_name: str = "gpt2-medium",
    epochs: int = 3,
    batch_size: int = 4,
    learning_rate: float = 5e-5,
    max_length: int = 512,
    save_steps: int = 500,
):
    """
    Train GPT-2 model on Google Colab
    
    Args:
        training_data_path: Path to training data JSON
        output_dir: Where to save the trained model
        model_name: Base model (gpt2, gpt2-medium, gpt2-large)
        epochs: Number of training epochs
        batch_size: Batch size (keep small for Colab free tier)
        learning_rate: Learning rate
        max_length: Maximum sequence length
        save_steps: Save checkpoint every N steps
    """
    try:
        from transformers import (
            GPT2LMHeadModel,
            GPT2Tokenizer,
            TextDataset,
            DataCollatorForLanguageModeling,
            Trainer,
            TrainingArguments,
        )
        import torch
    except ImportError:
        logger.error("❌ Install required packages: pip install transformers torch accelerate")
        return
    
    logger.info(f"🚀 Starting GPT-2 training with {model_name}")
    start_time = time.time()
    
    # Check for GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    # Load tokenizer and model
    logger.info("Loading tokenizer and model...")
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = GPT2LMHeadModel.from_pretrained(model_name)
    model.to(device)
    
    # Prepare dataset
    logger.info(f"Loading training data from {training_data_path}")
    
    # Convert JSON to text file for TextDataset
    temp_text_file = "/tmp/training_data.txt"
    with open(training_data_path, 'r') as f:
        data = json.load(f)
    
    with open(temp_text_file, 'w') as f:
        if isinstance(data, list):
            for item in data:
                # Assume each item has 'prompt' and 'lyrics' or just 'text'
                text = item.get('lyrics') or item.get('text') or str(item)
                f.write(text + "\n\n")
        elif isinstance(data, dict):
            for key, value in data.items():
                f.write(str(value) + "\n\n")
    
    # Create dataset
    dataset = TextDataset(
        tokenizer=tokenizer,
        file_path=temp_text_file,
        block_size=max_length,
    )
    
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        save_steps=save_steps,
        save_total_limit=2,
        learning_rate=learning_rate,
        logging_steps=100,
        logging_dir=f"{output_dir}/logs",
        prediction_loss_only=True,
        fp16=device == "cuda",  # Use mixed precision on GPU
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset,
    )
    
    # Train
    logger.info("🔥 Starting training...")
    train_result = trainer.train()
    
    # Save model
    logger.info(f"💾 Saving model to {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Calculate duration
    duration = time.time() - start_time
    
    logger.info(f"✅ Training complete!")
    logger.info(f"   Duration: {duration:.2f} seconds ({duration/60:.2f} minutes)")
    logger.info(f"   Final loss: {train_result.training_loss:.4f}")
    logger.info(f"   Model saved to: {output_dir}")
    
    # Clean up
    os.remove(temp_text_file)
    
    return {
        "model_path": output_dir,
        "final_loss": train_result.training_loss,
        "duration_seconds": int(duration),
    }


def train_phi3_model(
    training_data_path: str,
    output_dir: str = "./models",
    model_name: str = "microsoft/Phi-3-mini-4k-instruct",
    epochs: int = 3,
    batch_size: int = 2,
    learning_rate: float = 2e-5,
    max_length: int = 512,
):
    """
    Train Phi-3 model on Google Colab
    Phi-3-Mini is smaller and faster than GPT-2-Medium
    
    Args:
        training_data_path: Path to training data JSON
        output_dir: Where to save the trained model
        model_name: Base model
        epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        max_length: Maximum sequence length
    """
    try:
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            Trainer,
            TrainingArguments,
            DataCollatorForLanguageModeling,
        )
        from datasets import Dataset
        import torch
    except ImportError:
        logger.error("❌ Install required packages: pip install transformers datasets torch")
        return
    
    logger.info(f"🚀 Starting Phi-3 training with {model_name}")
    start_time = time.time()
    
    # Check for GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    # Load tokenizer and model
    logger.info("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    )
    model.to(device)
    
    # Load and prepare data
    logger.info(f"Loading training data from {training_data_path}")
    with open(training_data_path, 'r') as f:
        data = json.load(f)
    
    # Format data
    texts = []
    if isinstance(data, list):
        for item in data:
            text = item.get('lyrics') or item.get('text') or str(item)
            texts.append({"text": text})
    
    dataset = Dataset.from_list(texts)
    
    # Tokenize
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )
    
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"],
    )
    
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        save_steps=500,
        save_total_limit=2,
        learning_rate=learning_rate,
        logging_steps=50,
        fp16=device == "cuda",
        gradient_accumulation_steps=4,  # Simulate larger batch
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=tokenized_dataset,
    )
    
    # Train
    logger.info("🔥 Starting training...")
    train_result = trainer.train()
    
    # Save
    logger.info(f"💾 Saving model to {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    duration = time.time() - start_time
    
    logger.info(f"✅ Training complete!")
    logger.info(f"   Duration: {duration:.2f} seconds ({duration/60:.2f} minutes)")
    logger.info(f"   Final loss: {train_result.training_loss:.4f}")
    
    return {
        "model_path": output_dir,
        "final_loss": train_result.training_loss,
        "duration_seconds": int(duration),
    }


def convert_to_gguf(model_path: str, output_path: str, quantization: str = "Q4_K_M"):
    """
    Convert trained model to GGUF format for llama.cpp
    Requires llama.cpp installed
    
    Args:
        model_path: Path to HF model
        output_path: Output GGUF file path
        quantization: Quantization type (Q4_K_M, Q5_K_M, Q8_0, etc.)
    """
    logger.info(f"Converting model to GGUF format...")
    logger.info(f"Quantization: {quantization}")
    
    # This requires llama.cpp's convert.py and quantize tools
    logger.info("""
To convert to GGUF:

1. Clone llama.cpp:
   git clone https://github.com/ggerganov/llama.cpp
   cd llama.cpp && make

2. Convert to GGUF:
   python convert.py {model_path} --outfile {output_path}.tmp

3. Quantize:
   ./quantize {output_path}.tmp {output_path} {quantization}

This will create a small, CPU-friendly model file.
    """.format(
        model_path=model_path,
        output_path=output_path,
        quantization=quantization,
    ))


def prepare_training_data_template():
    """Create training data template"""
    template = {
        "metadata": {
            "description": "Training data for AI Rapper System",
            "format": "Each entry should be a complete verse or set of bars",
            "guidelines": [
                "Include your best written lyrics",
                "Focus on your unique style and voice",
                "Include various topics and moods",
                "Aim for 100-1000 examples for fine-tuning",
            ]
        },
        "examples": [
            {
                "prompt": "Write aggressive battle rap bars",
                "lyrics": "I'm the definition of determination\nEvery line I spit is a new revelation\nYou can't match my flow or my dedication\nI'm building my empire, no hesitation"
            },
            {
                "prompt": "Write motivational bars",
                "lyrics": "Started from the bottom, now I'm reaching for the peak\nEvery setback made me stronger, never weak\nI turn my pain into power, that's my technique\nThe future that I'm building is unique"
            },
        ],
        "training_data": [
            # Add your lyrics here
        ]
    }
    
    output_path = "./data/training_lyrics.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(template, f, indent=2)
    
    logger.info(f"✅ Training data template created at {output_path}")
    logger.info("   Fill in the 'training_data' array with your lyrics!")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train local rapper model")
    parser.add_argument("--action", choices=["prepare", "train-gpt2", "train-phi3"], required=True)
    parser.add_argument("--data", default="./data/training_lyrics.json")
    parser.add_argument("--output", default="./models/trained_model")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    
    args = parser.parse_args()
    
    if args.action == "prepare":
        prepare_training_data_template()
    elif args.action == "train-gpt2":
        train_gpt2_model(
            training_data_path=args.data,
            output_dir=args.output,
            epochs=args.epochs,
            batch_size=args.batch_size,
        )
    elif args.action == "train-phi3":
        train_phi3_model(
            training_data_path=args.data,
            output_dir=args.output,
            epochs=args.epochs,
            batch_size=args.batch_size,
        )
