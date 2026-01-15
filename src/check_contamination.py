#!/usr/bin/env python3
"""
check_contamination.py - Data Contamination Detection

Verifies zero overlap between train, dev, and test splits.
This script should be run before any training or evaluation.

Usage:
    python src/check_contamination.py
    python src/check_contamination.py --verbose  # Show sample counts
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Set, Dict, List

BASE_DIR = Path(__file__).parent.parent
TRAIN_DIR = BASE_DIR / "train"
DEV_DIR = BASE_DIR / "dev"
TEST_DIR = BASE_DIR / "test"


def sha256_string(s: str) -> str:
    """Compute SHA-256 hash of a string."""
    return hashlib.sha256(s.encode('utf-8')).hexdigest()


def extract_id(row: dict) -> str:
    """Extract unique identifier from a data row."""
    id_fields = ['Task ID', 'question_id', 'id', 'dag_id', 'sample_id', 'idx']
    for field in id_fields:
        if field in row and row[field] is not None:
            return str(row[field])
    # Fallback: hash the entire row
    return sha256_string(json.dumps(row, sort_keys=True, default=str))


def extract_prompt(row: dict) -> str:
    """Extract prompt/question text from a data row."""
    prompt_fields = ['prompt', 'question', 'Sample context', 'given_info', 'input', 'text']
    for field in prompt_fields:
        if field in row and row[field]:
            return str(row[field]).strip()
    return json.dumps(row, sort_keys=True, default=str)


def load_samples(filepath: Path) -> List[dict]:
    """Load samples from JSONL or CSV file."""
    samples = []
    
    if filepath.suffix == '.jsonl':
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    samples.append(json.loads(line))
    elif filepath.suffix == '.csv':
        import pandas as pd
        df = pd.read_csv(filepath)
        samples = df.to_dict('records')
    
    return samples


def get_ids_and_hashes(samples: List[dict], dataset_name: str = "") -> tuple[Set[str], Set[str]]:
    """Extract IDs and prompt hashes from samples."""
    ids = set()
    hashes = set()
    
    for row in samples:
        # Prefix ID with dataset name to avoid cross-dataset false positives
        raw_id = extract_id(row)
        prefixed_id = f"{dataset_name}:{raw_id}" if dataset_name else raw_id
        ids.add(prefixed_id)
        hashes.add(sha256_string(extract_prompt(row)))
    
    return ids, hashes


def load_split(split_dir: Path, split_name: str) -> Dict[str, tuple[Set[str], Set[str]]]:
    """Load all datasets in a split directory."""
    result = {}
    
    if not split_dir.exists():
        return result
    
    for subdir in split_dir.iterdir():
        if subdir.is_dir():
            dataset_name = subdir.name
            for filepath in subdir.glob("*.jsonl"):
                samples = load_samples(filepath)
                ids, hashes = get_ids_and_hashes(samples, dataset_name)
                result[f"{split_name}/{dataset_name}"] = (ids, hashes)
            for filepath in subdir.glob("*.csv"):
                samples = load_samples(filepath)
                ids, hashes = get_ids_and_hashes(samples, dataset_name)
                result[f"{split_name}/{dataset_name}"] = (ids, hashes)
    
    # Also check for files directly in the split directory
    for filepath in split_dir.glob("*.jsonl"):
        samples = load_samples(filepath)
        ids, hashes = get_ids_and_hashes(samples, filepath.stem)
        result[f"{split_name}/{filepath.stem}"] = (ids, hashes)
    
    return result


def check_contamination(verbose: bool = False) -> bool:
    """Check for contamination between all splits."""
    print("=" * 60)
    print("DATA CONTAMINATION CHECK")
    print("=" * 60)
    
    # Load all splits
    print("\n[1/3] Loading datasets...")
    
    train_data = load_split(TRAIN_DIR, "train")
    dev_data = load_split(DEV_DIR, "dev")
    test_data = load_split(TEST_DIR, "test")
    
    # Aggregate by split
    train_ids: Set[str] = set()
    train_hashes: Set[str] = set()
    for name, (ids, hashes) in train_data.items():
        train_ids.update(ids)
        train_hashes.update(hashes)
        if verbose:
            print(f"  {name}: {len(ids)} samples")
    
    dev_ids: Set[str] = set()
    dev_hashes: Set[str] = set()
    for name, (ids, hashes) in dev_data.items():
        dev_ids.update(ids)
        dev_hashes.update(hashes)
        if verbose:
            print(f"  {name}: {len(ids)} samples")
    
    test_ids: Set[str] = set()
    test_hashes: Set[str] = set()
    for name, (ids, hashes) in test_data.items():
        test_ids.update(ids)
        test_hashes.update(hashes)
        if verbose:
            print(f"  {name}: {len(ids)} samples")
    
    print(f"\n  Train: {len(train_ids)} unique IDs, {len(train_hashes)} unique prompts")
    print(f"  Dev: {len(dev_ids)} unique IDs, {len(dev_hashes)} unique prompts")
    print(f"  Test: {len(test_ids)} unique IDs, {len(test_hashes)} unique prompts")
    
    # Check ID overlaps
    print("\n[2/3] Checking ID overlaps...")
    
    train_dev_id = train_ids & dev_ids
    train_test_id = train_ids & test_ids
    dev_test_id = dev_ids & test_ids
    
    print(f"  Train ∩ Dev (IDs): {len(train_dev_id)}")
    print(f"  Train ∩ Test (IDs): {len(train_test_id)}")
    print(f"  Dev ∩ Test (IDs): {len(dev_test_id)}")
    
    # Check prompt hash overlaps
    print("\n[3/3] Checking prompt overlaps...")
    
    train_dev_hash = train_hashes & dev_hashes
    train_test_hash = train_hashes & test_hashes
    dev_test_hash = dev_hashes & test_hashes
    
    print(f"  Train ∩ Dev (prompts): {len(train_dev_hash)}")
    print(f"  Train ∩ Test (prompts): {len(train_test_hash)}")
    print(f"  Dev ∩ Test (prompts): {len(dev_test_hash)}")
    
    # Summary
    print("\n" + "=" * 60)
    
    id_contamination = len(train_dev_id) + len(train_test_id) + len(dev_test_id)
    prompt_contamination = len(train_dev_hash) + len(train_test_hash) + len(dev_test_hash)
    
    if id_contamination == 0 and prompt_contamination == 0:
        print("✓ PASS: Zero contamination detected")
        print("  - All splits have disjoint IDs")
        print("  - All splits have disjoint prompts")
        print("=" * 60)
        return True
    else:
        print("✗ FAIL: Contamination detected!")
        if id_contamination > 0:
            print(f"  - {id_contamination} overlapping IDs")
        if prompt_contamination > 0:
            print(f"  - {prompt_contamination} overlapping prompts")
        print("=" * 60)
        return False


def main():
    parser = argparse.ArgumentParser(description="Check for data contamination")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Show detailed sample counts")
    args = parser.parse_args()
    
    success = check_contamination(args.verbose)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
