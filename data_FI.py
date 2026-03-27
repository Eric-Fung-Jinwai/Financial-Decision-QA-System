"""
Download the finance-alpaca dataset from HuggingFace and save as JSONL files
(train/val split) for LoRA fine-tuning.

Dataset: https://huggingface.co/datasets/gbharti/finance-alpaca
"""

import json
import random
import os
import urllib.request

random.seed(42)

SYNTHETIC_FILE = "synthetic_data.json"

DATA_URL = "https://huggingface.co/datasets/gbharti/finance-alpaca/resolve/main/Cleaned_date.json"
RAW_FILE = "finance_alpaca.json"
TRAIN_FILE = "train.jsonl"
VAL_FILE = "val.jsonl"
TRAIN_RATIO = 0.9
MAX_SAMPLES = 5000  # Cap the alpaca dataset to keep training manageable

script_dir = os.path.dirname(os.path.abspath(__file__))
raw_path = os.path.join(script_dir, RAW_FILE)
train_path = os.path.join(script_dir, TRAIN_FILE)
val_path = os.path.join(script_dir, VAL_FILE)


def download_data():
    """Download the raw JSON file from HuggingFace."""
    if os.path.exists(raw_path):
        print(f"Raw file already exists: {raw_path}, skipping download.")
        return
    print(f"Downloading dataset from {DATA_URL} ...")
    urllib.request.urlretrieve(DATA_URL, raw_path)
    print("Download complete.")


def convert_to_jsonl():
    """Load raw JSON, clean, shuffle, split into train/val JSONL files."""
    print("Loading raw data...")
    with open(raw_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Clean: drop empty 'text' field, skip records with empty output
    cleaned = []
    for item in data:
        if not item.get("output", "").strip():
            continue
        cleaned.append({
            "instruction": item.get("instruction", "").strip(),
            "input": item.get("input", "").strip(),
            "output": item.get("output", "").strip(),
        })

    print(f"Loaded {len(cleaned)} valid records (dropped {len(data) - len(cleaned)}).")

    # Truncate to MAX_SAMPLES before adding synthetic data
    if len(cleaned) > MAX_SAMPLES:
        random.shuffle(cleaned)
        cleaned = cleaned[:MAX_SAMPLES]
        print(f"Truncated to {MAX_SAMPLES} samples.")

    # Append synthetic data for advanced financial analysis
    synthetic_path = os.path.join(script_dir, SYNTHETIC_FILE)
    with open(synthetic_path, "r", encoding="utf-8") as f:
        synthetic_categories = json.load(f)
    synthetic_count = 0
    for category, records in synthetic_categories.items():
        for record in records:
            record["category"] = category
            cleaned.append(record)
            synthetic_count += 1
        print(f"  - {category}: {len(records)} records")
    print(f"Added {synthetic_count} synthetic records. Total: {len(cleaned)}.")

    # Shuffle and split
    random.shuffle(cleaned)
    split_idx = int(len(cleaned) * TRAIN_RATIO)
    train_data = cleaned[:split_idx]
    val_data = cleaned[split_idx:]

    # Write JSONL files
    for path, records, label in [
        (train_path, train_data, "Train"),
        (val_path, val_data, "Val"),
    ]:
        with open(path, "w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"{label} set: {len(records)} records -> {path}")


if __name__ == "__main__":
    download_data()
    convert_to_jsonl()
    print("Done!")