"""
Central configuration for Gemma 4 Text-to-SQL fine-tuning.
Adjust these values based on your hardware and requirements.
"""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelConfig:
    model_id: str = "google/gemma-4-31B-it"
    max_seq_length: int = 2048
    load_in_4bit: bool = True          # QLoRA; set False + load_in_16bit for LoRA
    load_in_16bit: bool = False
    torch_dtype: str = "bfloat16"
    attn_implementation: str = "flash_attention_2"


@dataclass
class LoraConfig:
    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: list = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])
    bias: str = "none"
    task_type: str = "CAUSAL_LM"


@dataclass
class TrainingConfig:
    output_dir: str = "./output/gemma4-text2sql"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    bf16: bool = True
    logging_steps: int = 10
    eval_strategy: str = "steps"
    eval_steps: int = 100
    save_strategy: str = "steps"
    save_steps: int = 100
    save_total_limit: int = 3
    max_grad_norm: float = 0.3
    optim: str = "paged_adamw_8bit"
    report_to: str = "wandb"
    gradient_checkpointing: bool = True
    group_by_length: bool = True
    push_to_hub: bool = False
    hub_model_id: Optional[str] = None


@dataclass
class DataConfig:
    spider_dataset: str = "xlangai/spider"
    gretel_dataset: str = "gretelai/synthetic_text_to_sql"
    use_gretel_augmentation: bool = True
    gretel_sample_size: int = 10000       # Sample from Gretel to augment
    max_input_length: int = 1536
    max_output_length: int = 512
    val_split_ratio: float = 0.1
    seed: int = 42


@dataclass
class VertexConfig:
    project_id: str = "YOUR_PROJECT_ID"
    region: str = "us-central1"
    staging_bucket: str = "gs://YOUR_BUCKET/gemma4-text2sql"
    machine_type: str = "a3-highgpu-1g"
    accelerator_type: str = "NVIDIA_H100_80GB"
    accelerator_count: int = 1
    boot_disk_size_gb: int = 500
    container_uri: str = "us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.2-3:latest"
    display_name: str = "gemma4-text2sql-sft"


@dataclass
class EvalConfig:
    spider_dev_db_path: str = "./data/spider/database"
    num_eval_samples: int = 1034   # Spider dev set size
    temperature: float = 0.0
    max_new_tokens: int = 512
    gpt4o_mini_model: str = "gpt-4o-mini"
