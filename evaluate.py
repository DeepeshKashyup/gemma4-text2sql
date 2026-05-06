"""
Evaluate fine-tuned Gemma 4 on Spider dev set.

Metrics:
  - Exact Match (EM): predicted SQL == gold SQL (after normalization)
  - Execution Accuracy (EX): predicted SQL produces same result set as gold SQL

Usage:
    python eval/evaluate.py --model-path ./output/gemma4-text2sql/merged
    python eval/evaluate.py --model-path google/gemma-4-31B-it  # baseline
"""

import argparse
import json
import os
import re
import sqlite3
import sys
from pathlib import Path
from typing import Optional

import sqlparse
import torch
from tqdm import tqdm
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import ModelConfig, EvalConfig


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--dataset-path", type=str, default="./data/processed/gemma4_text2sql")
    parser.add_argument("--db-path", type=str, default=None, help="Path to Spider database dir")
    parser.add_argument("--output", type=str, default="./eval/results.json")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--use-4bit", action="store_true")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# SQL normalization for exact match comparison
# ---------------------------------------------------------------------------

def normalize_sql(sql: str) -> str:
    """Normalize SQL for comparison: lowercase, strip whitespace, remove aliases."""
    sql = sql.strip().rstrip(";").strip()
    sql = re.sub(r"\s+", " ", sql).lower()
    sql = sqlparse.format(sql, reindent=False, keyword_case="lower")
    return sql.strip()


# ---------------------------------------------------------------------------
# Execution accuracy: run both queries and compare result sets
# ---------------------------------------------------------------------------

def execute_sql(db_path: str, sql: str, timeout: int = 30) -> Optional[set]:
    """Execute SQL against a SQLite database and return result as a set of tuples."""
    try:
        conn = sqlite3.connect(db_path)
        conn.execute(f"PRAGMA busy_timeout = {timeout * 1000}")
        cursor = conn.cursor()
        cursor.execute(sql)
        results = set(cursor.fetchall())
        conn.close()
        return results
    except Exception as e:
        return None


def check_execution_accuracy(
    db_path: str, predicted_sql: str, gold_sql: str
) -> bool:
    """Compare execution results of predicted vs gold SQL."""
    pred_results = execute_sql(db_path, predicted_sql)
    gold_results = execute_sql(db_path, gold_sql)

    if pred_results is None or gold_results is None:
        return False

    return pred_results == gold_results


# ---------------------------------------------------------------------------
# Model inference
# ---------------------------------------------------------------------------

def load_model(model_path: str, use_4bit: bool = False):
    """Load model and tokenizer for inference."""
    bnb_config = None
    if use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model.eval()
    return model, tokenizer


def generate_sql(model, tokenizer, messages: list, max_new_tokens: int = 512) -> str:
    """Generate SQL from chat messages using Gemma 4."""
    # Only include system + user messages (not the gold answer)
    prompt_messages = [m for m in messages if m["role"] != "model"]
    prompt = tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.0,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )

    # Decode only the generated portion
    generated = outputs[0][inputs["input_ids"].shape[1]:]
    sql = tokenizer.decode(generated, skip_special_tokens=True).strip()

    # Clean up: extract just the SQL if model adds explanation
    sql = sql.split(";")[0].strip() + ";" if ";" in sql else sql.strip()
    sql = re.sub(r"^```sql\s*", "", sql)
    sql = re.sub(r"\s*```$", "", sql)

    return sql


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    eval_cfg = EvalConfig()

    print(f"Loading model from: {args.model_path}")
    model, tokenizer = load_model(args.model_path, use_4bit=args.use_4bit)

    print(f"Loading dataset from: {args.dataset_path}")
    ds = load_from_disk(args.dataset_path)
    val_ds = ds["validation"]

    if args.max_samples:
        val_ds = val_ds.select(range(min(args.max_samples, len(val_ds))))

    db_path = args.db_path or eval_cfg.spider_dev_db_path

    results = []
    exact_match_count = 0
    execution_accuracy_count = 0
    total = 0
    errors = 0

    print(f"Evaluating on {len(val_ds)} examples...")
    for idx, example in enumerate(tqdm(val_ds)):
        messages = example["messages"]

        # Extract gold SQL from the model turn
        gold_sql = ""
        for msg in messages:
            if msg["role"] == "model":
                gold_sql = msg["content"]
                break

        db_id = example.get("db_id", "unknown")

        # Generate prediction
        try:
            predicted_sql = generate_sql(model, tokenizer, messages)
        except Exception as e:
            predicted_sql = ""
            errors += 1

        # Exact match
        em = normalize_sql(predicted_sql) == normalize_sql(gold_sql)
        if em:
            exact_match_count += 1

        # Execution accuracy (if DB files available)
        ex = False
        db_file = os.path.join(db_path, db_id, f"{db_id}.sqlite")
        if os.path.exists(db_file):
            ex = check_execution_accuracy(db_file, predicted_sql, gold_sql)
            if ex:
                execution_accuracy_count += 1

        total += 1

        results.append({
            "idx": idx,
            "db_id": db_id,
            "gold_sql": gold_sql,
            "predicted_sql": predicted_sql,
            "exact_match": em,
            "execution_accuracy": ex,
        })

        # Print progress every 100 examples
        if (idx + 1) % 100 == 0:
            print(
                f"  [{idx+1}/{len(val_ds)}] "
                f"EM: {exact_match_count/total:.3f} | "
                f"EX: {execution_accuracy_count/total:.3f}"
            )

    # Summary
    em_score = exact_match_count / total if total > 0 else 0
    ex_score = execution_accuracy_count / total if total > 0 else 0

    summary = {
        "model_path": args.model_path,
        "total_examples": total,
        "exact_match": em_score,
        "execution_accuracy": ex_score,
        "errors": errors,
    }

    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    print(f"  Model:               {args.model_path}")
    print(f"  Total examples:      {total}")
    print(f"  Exact Match (EM):    {em_score:.4f} ({exact_match_count}/{total})")
    print(f"  Execution Acc (EX):  {ex_score:.4f} ({execution_accuracy_count}/{total})")
    print(f"  Errors:              {errors}")
    print("=" * 50)

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump({"summary": summary, "results": results}, f, indent=2)
    print(f"\nDetailed results saved to: {output_path}")


if __name__ == "__main__":
    main()
