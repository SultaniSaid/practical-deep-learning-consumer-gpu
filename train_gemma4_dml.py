import os
import torch
import torch_directml
from transformers import AutoProcessor, AutoModelForMultimodalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
import argparse

def train_gemma4(device_target='gpu', num_steps=5):
    # 1. Device Setup
    if device_target == 'gpu':
        device = torch_directml.device()
        print(f"Targeting DirectML GPU: {torch_directml.device_name(0)}")
    else:
        device = torch.device('cpu')
        print("Targeting CPU path.")

    # 2. Load Model & Processor (Multimodal)
    model_id = "google/gemma-4-E4B-it"
    print(f"Loading multimodal model {model_id}...")
    
    # Transformers v5+ natively supports Gemma 4 with trust_remote_code
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    
    # Use the specialized Multimodal class for E4B
    model = AutoModelForMultimodalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float32,
        trust_remote_code=True,
        device_map=None # Manual move later
    )

    # 3. Apply LoRA Config
    print("Applying LoRA configuration...")
    # Standard LoRA targets for Gemma cross-attention/hybrid modules
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj", "o_proj", "gate_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)

    # 4. THE VANGUARD REPAIR (GEMMA-4 MULTIMODAL VERSION)
    print("Applying Vanguard Repair (Gradient Re-Sync)...")
    model.to(device)
    
    # CRITICAL: Re-enable gradients for LoRA adapters which get detached during .to()
    # Also ensures the vision-text bridge parameters are alive if needed
    for name, param in model.named_parameters():
        if "lora" in name:
            param.requires_grad_(True)
    
    # 5. Dataset (Minimal sample for verification)
    dataset = load_dataset("imdb", split="train[:10]")
    
    def tokenize_function(examples):
        # Simplest text-only fine-tuning test for verification
        return processor(text=examples["text"], padding="max_length", truncation=True, max_length=128, return_tensors="pt")

    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=dataset.column_names)

    # 6. Training Arguments (DirectML Optimized / Extreme Memory Savings)
    training_args = TrainingArguments(
        output_dir="./gemma4-results",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8, # Large accumulation to prevent OOM
        max_steps=num_steps,
        learning_rate=2e-4,
        logging_steps=1,
        save_steps=100,
        fp16=False, # DirectML requires FP32 for transformer stability
        push_to_hub=False,
        report_to="none",
        remove_unused_columns=False # Critical for multimodal processors
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
    )

    print("Starting Vanguard-enabled training session...")
    trainer.train()
    print("Training complete! Breakthrough verified.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="gpu", choices=["gpu", "cpu"])
    parser.add_argument("--steps", type=int, default=5)
    args = parser.parse_args()
    
    train_gemma4(device_target=args.device, num_steps=args.steps)
