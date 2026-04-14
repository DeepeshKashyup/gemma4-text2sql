"""
Side-by-side benchmark: Gemma 4 base vs fine-tuned vs GPT-4o-mini.

Produces a comparison table and per-difficulty breakdown (easy/medium/hard/extra).

Usage:
    python eval/benchmark.py
    python eval/benchmark.py --skip-gpt  # Skip OpenAI API calls
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import EvalConfig, ModelConfig


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", type=str, default="google/gemma-4-31B-it")
    parser.add_argument("--finetuned-model", type=str, default="./output/gemma4-text2sql/merged")
    parser.add_argument("--skip-gpt", action="store_true")
    parser.add_argument("--max-samples", type=int, default=200, help="Subset for quick benchmark")
    parser.add_argument("--output", type=str, default="./eval/benchmark_results.json")
    return parser.parse_args()


SYSTEM_PROMPT = (
    "You are an expert SQL assistant. Given a database schema and a natural "
    "language question, generate the correct SQL query. Output ONLY the SQL "
    "query with no explanation."
)


# ---------------------------------------------------------------------------
# GPT-4o-mini baseline
# ---------------------------------------------------------------------------

def query_gpt4o_mini(schema: str, question: str, model_name: str = "gpt-4o-mini") -> str:
    """Query OpenAI's GPT-4o-mini for SQL generation."""
    from openai import OpenAI

    client = OpenAI()  # Uses OPENAI_API_KEY env var

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"{schema}\n\n-- Question: {question}"},
        ],
        temperature=0.0,
        max_tokens=512,
    )

    return response.choices[0].message.content.strip()


# ---------------------------------------------------------------------------
# Gemma inference (reuse from evaluate.py)
# ---------------------------------------------------------------------------

def load_gemma_model(model_path: str):
    """Load a Gemma model for inference."""
    from evaluate import load_model
    return load_model(model_path, use_4bit=True)


def query_gemma(model, tokenizer, schema: str, question: str) -> str:
    """Generate SQL using a Gemma model."""
    from evaluate import generate_sql

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"{schema}\n\n-- Question: {question}"},
    ]
    return generate_sql(model, tokenizer, messages)


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

def classify_difficulty(sql: str) -> str:
    """Heuristic difficulty classification based on SQL complexity."""
    sql_lower = sql.lower()
    has_subquery = "select" in sql_lower[sql_lower.find("from"):]
    has_join = "join" in sql_lower
    has_group = "group by" in sql_lower
    has_having = "having" in sql_lower
    has_union = "union" in sql_lower or "intersect" in sql_lower or "except" in sql_lower

    complexity = sum([has_subquery, has_join, has_group, has_having, has_union])

    if complexity == 0:
        return "easy"
    elif complexity == 1:
        return "medium"
    elif complexity == 2:
        return "hard"
    else:
        return "extra"


def main():
    args = parse_args()
    eval_cfg = EvalConfig()

    from datasets import load_from_disk
    from evaluate import normalize_sql, check_execution_accuracy

    # Load dataset
    ds = load_from_disk("./data/processed/gemma4_text2sql")
    val_ds = ds["validation"]

    if args.max_samples:
        val_ds = val_ds.select(range(min(args.max_samples, len(val_ds))))

    # Load models
    models = {}

    print("Loading Gemma 4 base model...")
    base_model, base_tokenizer = load_gemma_model(args.base_model)
    models["gemma4_base"] = (base_model, base_tokenizer)

    if os.path.exists(args.finetuned_model):
        print("Loading fine-tuned model...")
        ft_model, ft_tokenizer = load_gemma_model(args.finetuned_model)
        models["gemma4_finetuned"] = (ft_model, ft_tokenizer)
    else:
        print(f"Fine-tuned model not found at {args.finetuned_model}, skipping.")

    # Run benchmark
    results = []

    for idx, example in enumerate(tqdm(val_ds, desc="Benchmarking")):
        messages = example["messages"]
        gold_sql = ""
        schema_str = ""
        question_str = ""

        for msg in messages:
            if msg["role"] == "model":
                gold_sql = msg["content"]
            elif msg["role"] == "user":
                # Parse schema and question from user message
                parts = msg["content"].rsplit("-- Question:", 1)
                if len(parts) == 2:
                    schema_str = parts[0].strip()
                    question_str = parts[1].strip()

        db_id = example.get("db_id", "unknown")
        difficulty = classify_difficulty(gold_sql)

        row = {
            "idx": idx,
            "db_id": db_id,
            "difficulty": difficulty,
            "gold_sql": gold_sql,
        }

        # Gemma 4 base
        try:
            base_pred = query_gemma(base_model, base_tokenizer, schema_str, question_str)
            row["gemma4_base_pred"] = base_pred
            row["gemma4_base_em"] = normalize_sql(base_pred) == normalize_sql(gold_sql)
        except Exception as e:
            row["gemma4_base_pred"] = f"ERROR: {e}"
            row["gemma4_base_em"] = False

        # Gemma 4 fine-tuned
        if "gemma4_finetuned" in models:
            try:
                ft_pred = query_gemma(ft_model, ft_tokenizer, schema_str, question_str)
                row["gemma4_ft_pred"] = ft_pred
                row["gemma4_ft_em"] = normalize_sql(ft_pred) == normalize_sql(gold_sql)
            except Exception as e:
                row["gemma4_ft_pred"] = f"ERROR: {e}"
                row["gemma4_ft_em"] = False

        # GPT-4o-mini
        if not args.skip_gpt:
            try:
                gpt_pred = query_gpt4o_mini(schema_str, question_str)
                row["gpt4o_mini_pred"] = gpt_pred
                row["gpt4o_mini_em"] = normalize_sql(gpt_pred) == normalize_sql(gold_sql)
            except Exception as e:
                row["gpt4o_mini_pred"] = f"ERROR: {e}"
                row["gpt4o_mini_em"] = False

        results.append(row)

    # ---------------------------------------------------------------------------
    # Aggregate results
    # ---------------------------------------------------------------------------
    df = pd.DataFrame(results)

    print("\n" + "=" * 70)
    print("BENCHMARK RESULTS — Gemma 4 Text-to-SQL")
    print("=" * 70)

    # Overall scores
    overall = {}
    for col, label in [
        ("gemma4_base_em", "Gemma 4 31B (base)"),
        ("gemma4_ft_em", "Gemma 4 31B (fine-tuned)"),
        ("gpt4o_mini_em", "GPT-4o-mini"),
    ]:
        if col in df.columns:
            score = df[col].mean()
            overall[label] = f"{score:.3f}"
            print(f"  {label:30s}  EM: {score:.3f}")

    # Per-difficulty breakdown
    print("\nPer-Difficulty Breakdown (Exact Match):")
    print("-" * 70)

    for diff in ["easy", "medium", "hard", "extra"]:
        subset = df[df["difficulty"] == diff]
        if len(subset) == 0:
            continue
        line = f"  {diff:8s} (n={len(subset):3d})"
        for col in ["gemma4_base_em", "gemma4_ft_em", "gpt4o_mini_em"]:
            if col in df.columns:
                score = subset[col].mean()
                line += f"  | {score:.3f}"
        print(line)

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump({
            "overall": overall,
            "per_difficulty": df.groupby("difficulty").agg({
                col: "mean" for col in ["gemma4_base_em", "gemma4_ft_em", "gpt4o_mini_em"]
                if col in df.columns
            }).to_dict(),
            "results": results,
        }, f, indent=2, default=str)

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
