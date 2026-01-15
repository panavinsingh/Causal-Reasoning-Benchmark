#!/usr/bin/env python3
"""
build_train_4k.py - Deterministic 4K CCR.GB Training Subset

Implements the refined sampling strategy:
- 4,000 CCR.GB samples via sha256(prompt) % 17 == 3
- Excludes test IDs
- Deterministic, no RNG

Usage:
    python src/build_train_4k.py
"""

import hashlib
import json
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).parent.parent
TRAIN_DIR = BASE_DIR / "train"
TEST_DIR = BASE_DIR / "test"

# CCR.GB source
CCRGB_URL = "https://huggingface.co/datasets/jmaasch/compositional_causal_reasoning/resolve/main/clinical_notes_v0/clinical_notes_v0.csv"

# Sampling parameters
TARGET_TRAIN = 4_000
SEED_RULE = 17
TRAIN_MODULO = 3  # sha256 % 17 == 3 for train
DEV_MODULO = 7    # sha256 % 17 == 7 for dev
DEV_TARGET = 300


def sha256_int(s: str) -> int:
    """Get integer from SHA-256 hash."""
    return int(hashlib.sha256(s.encode('utf-8')).hexdigest(), 16)


def extract_task_id_numeric(task_id: str) -> int:
    """Extract numeric index from Task ID."""
    try:
        parts = str(task_id).split('.')
        return int(parts[0]) * 10000 + int(parts[1]) * 100 + int(parts[2])
    except:
        return -1


def main():
    print("=" * 60)
    print("DETERMINISTIC 4K CCR.GB SAMPLING")
    print("=" * 60)
    
    # Load test IDs
    test_ids_file = TEST_DIR / "ccrgb" / "ids_test_400.txt"
    
    # First, create test IDs file if not exists
    test_jsonl = TEST_DIR / "ccrgb" / "ccrgb_test.jsonl"
    if test_jsonl.exists():
        print("\n[1/5] Extracting test IDs...")
        test_ids = set()
        with open(test_jsonl, 'r', encoding='utf-8') as f:
            for line in f:
                row = json.loads(line)
                test_ids.add(row['Task ID'])
        
        # Save test IDs
        with open(test_ids_file, 'w') as f:
            for tid in sorted(test_ids):
                f.write(f"{tid}\n")
        print(f"    Saved {len(test_ids)} test IDs to {test_ids_file.name}")
    else:
        print("    ERROR: Test file not found")
        return
    
    # Download full CCR.GB
    print("\n[2/5] Downloading CCR.GB...")
    df = pd.read_csv(CCRGB_URL)
    print(f"    Loaded {len(df)} rows")
    
    # Sort by Task ID
    df = df.sort_values(by="Task ID", key=lambda x: x.apply(extract_task_id_numeric))
    df = df.reset_index(drop=True)
    
    # Get prompt column for hashing
    prompt_col = "Sample context" if "Sample context" in df.columns else df.columns[0]
    
    # Deterministic sampling
    print(f"\n[3/5] Sampling with sha256 % {SEED_RULE} == {TRAIN_MODULO}...")
    
    train_rows = []
    dev_rows = []
    
    for _, row in df.iterrows():
        tid = row['Task ID']
        
        # Skip test IDs
        if tid in test_ids:
            continue
        
        # Get hash of prompt
        prompt = str(row[prompt_col])
        h = sha256_int(prompt)
        
        # Deterministic selection
        if h % SEED_RULE == TRAIN_MODULO:
            if len(train_rows) < TARGET_TRAIN:
                train_rows.append(row.to_dict())
        elif h % SEED_RULE == DEV_MODULO:
            if len(dev_rows) < DEV_TARGET:
                dev_rows.append(row.to_dict())
    
    print(f"    Train: {len(train_rows)} samples")
    print(f"    Dev: {len(dev_rows)} samples")
    
    # Save train
    print("\n[4/5] Saving files...")
    train_file = TRAIN_DIR / "ccrgb" / "ccrgb_train.jsonl"
    with open(train_file, 'w', encoding='utf-8') as f:
        for row in train_rows:
            f.write(json.dumps(row, default=str) + "\n")
    print(f"    Saved: {train_file.name}")
    
    # Save dev
    dev_dir = BASE_DIR / "dev"
    dev_dir.mkdir(exist_ok=True)
    dev_file = dev_dir / "dev_300.jsonl"
    with open(dev_file, 'w', encoding='utf-8') as f:
        for row in dev_rows:
            f.write(json.dumps(row, default=str) + "\n")
    print(f"    Saved: {dev_file.name}")
    
    # Update metadata
    metadata = {
        "name": "CCR.GB Train",
        "source": "https://huggingface.co/datasets/jmaasch/compositional_causal_reasoning",
        "samples": len(train_rows),
        "license": "MIT",
        "selection_rule": f"sha256(prompt) % {SEED_RULE} == {TRAIN_MODULO}, exclude test IDs",
        "description": "Deterministic 4K training subset"
    }
    with open(TRAIN_DIR / "ccrgb" / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Dev metadata
    dev_metadata = {
        "name": "Dev Split",
        "samples": len(dev_rows),
        "license": "MIT",
        "selection_rule": f"sha256(prompt) % {SEED_RULE} == {DEV_MODULO}",
        "description": "Deterministic dev split from CCR.GB"
    }
    with open(dev_dir / "metadata.json", 'w') as f:
        json.dump(dev_metadata, f, indent=2)
    
    print("\n[5/5] Summary")
    print("=" * 60)
    print(f"  Train CCR.GB: {len(train_rows)}")
    print(f"  Dev: {len(dev_rows)}")
    print(f"  Test CCR.GB: {len(test_ids)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
