# Gemma 4 Text-to-SQL Fine-Tuning on Vertex AI

> Fine-tune Google's Gemma 4 31B on the Spider dataset using LoRA/QLoRA,
> evaluate against GPT-4o-mini baseline, and deploy on Vertex AI.

## Project Structure

```
gemma4-text2sql/
├── README.md
├── requirements.txt
├── config.py                  # All hyperparameters & paths
├── data/
│   └── prepare_dataset.py     # Download & format Spider for Gemma 4
├── train/
│   ├── train_local.py         # Local LoRA fine-tuning (single GPU)
│   └── train_vertex.py        # Vertex AI Training Job launcher
├── eval/
│   ├── evaluate.py            # Execution accuracy + exact match
│   └── benchmark.py           # Side-by-side: Gemma4-base vs fine-tuned vs GPT-4o-mini
├── serve/
│   └── deploy_vertex.py       # Deploy fine-tuned model to Vertex AI endpoint
└── Dockerfile                 # Custom container for Vertex AI training
```

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Prepare Spider dataset in Gemma 4 chat format
python data/prepare_dataset.py

# 3. Fine-tune locally (needs ≥24GB VRAM for 31B QLoRA)
python train/train_local.py

# 4. Or launch on Vertex AI (H100)
python train/train_vertex.py

# 5. Evaluate
python eval/evaluate.py --model-path ./output/gemma4-text2sql

# 6. Benchmark against baselines
python eval/benchmark.py

# 7. Deploy to Vertex AI
python serve/deploy_vertex.py
```

## Datasets

| Dataset | Size | Why |
|---------|------|-----|
| **Spider 1.0** (`xlangai/spider`) | 10,181 questions, 200 DBs | Gold standard cross-domain text-to-SQL |
| **Gretel Synthetic** (`gretelai/synthetic_text_to_sql`) | 100K+ samples | Augment with diverse SQL patterns |
| **Spider-Syn** | Adversarial NL variants | Robustness testing |
