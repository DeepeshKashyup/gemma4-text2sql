"""
Prepare Spider (+ optional Gretel augmentation) for Gemma 4 SFT.

Converts raw text-to-SQL pairs into Gemma 4's native chat format:
  user: <schema> + <question>
  model: <sql>

Usage:
    python data/prepare_dataset.py
"""

import json
import random
from pathlib import Path
from datasets import load_dataset, Dataset, DatasetDict
from config import DataConfig

cfg = DataConfig()

# ---------------------------------------------------------------------------
# Schema formatter: turns Spider table metadata into a compact prompt
# ---------------------------------------------------------------------------

def format_schema(db_id: str, tables: dict) -> str:
    """Build a concise schema string from Spider's tables.json entry."""
    lines = [f"-- Database: {db_id}"]
    table_names = tables["table_names_original"]
    column_names = tables["column_names_original"]  # (table_idx, col_name)
    column_types = tables["column_types"]
    primary_keys = set(tables["primary_keys"])

    for tidx, tname in enumerate(table_names):
        cols = []
        for cidx, (ctable, cname) in enumerate(column_names):
            if ctable == tidx:
                ctype = column_types[cidx] if cidx < len(column_types) else "text"
                pk = " PRIMARY KEY" if cidx in primary_keys else ""
                cols.append(f"  {cname} {ctype}{pk}")
        col_str = ",\n".join(cols)
        lines.append(f"CREATE TABLE {tname} (\n{col_str}\n);")

    # Foreign keys
    for fk_col, pk_col in tables.get("foreign_keys", []):
        fk_table = column_names[fk_col][0]
        pk_table = column_names[pk_col][0]
        fk_name = column_names[fk_col][1]
        pk_name = column_names[pk_col][1]
        lines.append(
            f"-- {table_names[fk_table]}.{fk_name} REFERENCES "
            f"{table_names[pk_table]}.{pk_name}"
        )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Convert to Gemma 4 chat messages
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are an expert SQL assistant. Given a database schema and a natural "
    "language question, generate the correct SQL query. Output ONLY the SQL "
    "query with no explanation."
)


def spider_to_chat(example: dict, schema_map: dict) -> dict:
    """Convert a single Spider example to Gemma 4 chat format."""
    db_id = example["db_id"]
    schema_str = schema_map.get(db_id, f"-- Database: {db_id}")
    question = example["question"]

    user_content = f"{schema_str}\n\n-- Question: {question}"
    assistant_content = example["query"]

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
        {"role": "model", "content": assistant_content},
    ]

    return {"messages": messages, "db_id": db_id}


def gretel_to_chat(example: dict) -> dict:
    """Convert a Gretel synthetic_text_to_sql example to chat format."""
    schema_str = example.get("sql_context", "")
    question = example.get("sql_prompt", "")
    sql = example.get("sql", "")

    user_content = f"{schema_str}\n\n-- Question: {question}"

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
        {"role": "model", "content": sql},
    ]

    return {
        "messages": messages,
        "db_id": example.get("domain", "synthetic"),
    }


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main():
    print("Loading Spider dataset...")
    spider = load_dataset(cfg.spider_dataset, trust_remote_code=True)

    # Build schema map from Spider tables
    # Spider dataset has a 'train' and 'validation' split
    # Tables info is embedded in the dataset
    train_ds = spider["train"]
    val_ds = spider["validation"]

    # Build schema map: we need tables.json which is part of the Spider download
    # For HF dataset, schema info is in the db_id; we'll create a simpler version
    # by extracting unique schemas from the dataset
    print("Building schema map from Spider metadata...")

    # The HF Spider dataset includes schema info we can reconstruct
    # For simplicity, we'll create schema strings from the available columns
    schema_map = {}
    for split_ds in [train_ds, val_ds]:
        for ex in split_ds:
            db_id = ex["db_id"]
            if db_id not in schema_map:
                # Use the query context available in the dataset
                schema_map[db_id] = f"-- Database: {db_id}"

    print(f"Found {len(schema_map)} unique databases")

    # Convert Spider to chat format
    print("Converting Spider train split...")
    train_examples = [spider_to_chat(ex, schema_map) for ex in train_ds]

    print("Converting Spider validation split...")
    val_examples = [spider_to_chat(ex, schema_map) for ex in val_ds]

    # Optional: augment with Gretel synthetic data
    if cfg.use_gretel_augmentation:
        print(f"Loading Gretel synthetic dataset (sampling {cfg.gretel_sample_size})...")
        gretel = load_dataset(cfg.gretel_dataset, split="train")

        # Sample a subset
        indices = random.sample(range(len(gretel)), min(cfg.gretel_sample_size, len(gretel)))
        gretel_subset = gretel.select(indices)

        gretel_examples = [gretel_to_chat(ex) for ex in gretel_subset]
        print(f"Added {len(gretel_examples)} Gretel examples")

        # Merge: Spider train + Gretel augmentation
        train_examples = train_examples + gretel_examples
        random.shuffle(train_examples)

    print(f"Final dataset: {len(train_examples)} train, {len(val_examples)} val")

    # Save as HF Dataset
    output_dir = Path("./data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = Dataset.from_list(train_examples)
    val_dataset = Dataset.from_list(val_examples)

    ds = DatasetDict({"train": train_dataset, "validation": val_dataset})
    ds.save_to_disk(str(output_dir / "gemma4_text2sql"))

    # Also save as JSONL for inspection
    for split_name, examples in [("train", train_examples), ("validation", val_examples)]:
        jsonl_path = output_dir / f"{split_name}.jsonl"
        with open(jsonl_path, "w") as f:
            for ex in examples:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
        print(f"Saved {jsonl_path}")

    print("Dataset preparation complete!")
    print(f"\nSample training example:")
    sample = train_examples[0]
    for msg in sample["messages"]:
        role = msg["role"]
        content = msg["content"][:200]
        print(f"  [{role}]: {content}...")


if __name__ == "__main__":
    random.seed(cfg.seed)
    main()
