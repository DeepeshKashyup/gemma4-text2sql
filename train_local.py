"""
Fine-tune Gemma 4 31B on Text-to-SQL using QLoRA (local single-GPU).

Requirements:
  - NVIDIA GPU with ≥24GB VRAM (H100/A100 for 31B, RTX 4090 for E4B)
  - Run data/prepare_dataset.py first

Usage:
    python train/train_local.py
    python train/train_local.py --model google/gemma-4-E4B-it  # smaller variant
"""

import argparse
import os
import torch
from datasets import load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig as PeftLoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import ModelConfig, LoraConfig, TrainingConfig, DataConfig


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None, help="Override model ID")
    parser.add_argument("--no-4bit", action="store_true", help="Use 16-bit LoRA instead of QLoRA")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    return parser.parse_args()


def format_chat(example, tokenizer):
    """Apply Gemma 4 chat template to messages."""
    messages = example["messages"]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    return {"text": text}


def main():
    args = parse_args()
    model_cfg = ModelConfig()
    lora_cfg = LoraConfig()
    train_cfg = TrainingConfig()
    data_cfg = DataConfig()

    # Override from CLI
    if args.model:
        model_cfg.model_id = args.model
    if args.no_4bit:
        model_cfg.load_in_4bit = False
        model_cfg.load_in_16bit = True
    if args.epochs:
        train_cfg.num_train_epochs = args.epochs
    if args.lr:
        train_cfg.learning_rate = args.lr
    if args.output_dir:
        train_cfg.output_dir = args.output_dir

    print(f"Model: {model_cfg.model_id}")
    print(f"QLoRA: {model_cfg.load_in_4bit} | LoRA rank: {lora_cfg.r}")

    # -----------------------------------------------------------------------
    # Load dataset
    # -----------------------------------------------------------------------
    print("Loading processed dataset...")
    ds = load_from_disk("./data/processed/gemma4_text2sql")
    print(f"Train: {len(ds['train'])} | Val: {len(ds['validation'])}")

    # -----------------------------------------------------------------------
    # Load tokenizer
    # -----------------------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(
        model_cfg.model_id,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Apply chat template
    print("Formatting dataset with Gemma 4 chat template...")
    ds = ds.map(
        lambda ex: format_chat(ex, tokenizer),
        remove_columns=["messages", "db_id"],
    )

    # -----------------------------------------------------------------------
    # Load model with quantization
    # -----------------------------------------------------------------------
    bnb_config = None
    if model_cfg.load_in_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    print(f"Loading {model_cfg.model_id}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_cfg.model_id,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation=model_cfg.attn_implementation,
    )

    if model_cfg.load_in_4bit:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=train_cfg.gradient_checkpointing
        )

    # -----------------------------------------------------------------------
    # Attach LoRA adapters
    # -----------------------------------------------------------------------
    peft_config = PeftLoraConfig(
        r=lora_cfg.r,
        lora_alpha=lora_cfg.lora_alpha,
        lora_dropout=lora_cfg.lora_dropout,
        target_modules=lora_cfg.target_modules,
        bias=lora_cfg.bias,
        task_type=lora_cfg.task_type,
    )

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # -----------------------------------------------------------------------
    # Training
    # -----------------------------------------------------------------------
    sft_config = SFTConfig(
        output_dir=train_cfg.output_dir,
        num_train_epochs=train_cfg.num_train_epochs,
        per_device_train_batch_size=train_cfg.per_device_train_batch_size,
        gradient_accumulation_steps=train_cfg.gradient_accumulation_steps,
        learning_rate=train_cfg.learning_rate,
        warmup_ratio=train_cfg.warmup_ratio,
        lr_scheduler_type=train_cfg.lr_scheduler_type,
        bf16=train_cfg.bf16,
        logging_steps=train_cfg.logging_steps,
        eval_strategy=train_cfg.eval_strategy,
        eval_steps=train_cfg.eval_steps,
        save_strategy=train_cfg.save_strategy,
        save_steps=train_cfg.save_steps,
        save_total_limit=train_cfg.save_total_limit,
        max_grad_norm=train_cfg.max_grad_norm,
        optim=train_cfg.optim,
        report_to=train_cfg.report_to,
        gradient_checkpointing=train_cfg.gradient_checkpointing,
        group_by_length=train_cfg.group_by_length,
        max_seq_length=model_cfg.max_seq_length,
        dataset_text_field="text",
        push_to_hub=train_cfg.push_to_hub,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        args=sft_config,
    )

    print("Starting training...")
    trainer.train()

    # Save final adapter
    final_path = os.path.join(train_cfg.output_dir, "final_adapter")
    trainer.save_model(final_path)
    tokenizer.save_pretrained(final_path)
    print(f"Training complete! Adapter saved to: {final_path}")

    # -----------------------------------------------------------------------
    # Optional: merge adapter into base model for easier deployment
    # -----------------------------------------------------------------------
    print("Merging adapter into base model...")
    merged_model = model.merge_and_unload()
    merged_path = os.path.join(train_cfg.output_dir, "merged")
    merged_model.save_pretrained(merged_path)
    tokenizer.save_pretrained(merged_path)
    print(f"Merged model saved to: {merged_path}")


if __name__ == "__main__":
    main()
