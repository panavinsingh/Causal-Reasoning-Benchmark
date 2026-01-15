#!/usr/bin/env python3
"""
prepare_data.py - Dataset Preparation Script

Downloads and prepares all datasets from upstream sources.
Supports --force-download to rebuild from scratch.

Usage:
    python src/prepare_data.py                    # Prepare if missing
    python src/prepare_data.py --force-download  # Force re-download
    python src/prepare_data.py --verify          # Verify integrity only
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

try:
    from datasets import load_dataset
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    print("Warning: 'datasets' library not installed. Run: pip install datasets")

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).parent.parent
TRAIN_DIR = BASE_DIR / "train"
TEST_DIR = BASE_DIR / "test"
CHECKSUMS_DIR = BASE_DIR / "checksums"

# Dataset sources
SOURCES = {
    "ccrgb": {
        "hf_repo": "jmaasch/compositional_causal_reasoning",
        "subset": "clinical_notes_v0",
        "url": "https://huggingface.co/datasets/jmaasch/compositional_causal_reasoning/resolve/main/clinical_notes_v0/clinical_notes_v0.csv",
        "license": "MIT",
    },
    "counterbench": {
        "hf_repo": "CounterBench/CounterBench",
        "license": "MIT",
    },
    "cladder": {
        "hf_repo": "causalnlp/CLadder",
        "split": "full_v1.5_default",
        "license": "Apache-2.0",
    },
    "corr2cause": {
        "hf_repo": "causalnlp/corr2cause",
        "split": "test",
        "license": "CC-BY-4.0",
    },
    "causcibench": {
        "url": "https://raw.githubusercontent.com/causalNLP/CauSciBench/main/data/real_info.csv",
        "license": "Apache-2.0",
    },
}

# CCR.GB curriculum-aware split ranges
CCRGB_TEST_RANGES = [
    (0, 39), (40, 79), (80, 119),      # Associational
    (120, 159), (160, 239), (240, 319), # Interventional
    (320, 339), (340, 379), (380, 399), # Counterfactual
]


# ─────────────────────────────────────────────────────────────────────────────
# Utility Functions
# ─────────────────────────────────────────────────────────────────────────────

def sha256_file(filepath: Path) -> str:
    """Compute SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def count_jsonl(filepath: Path) -> int:
    """Count lines in JSONL file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return sum(1 for _ in f)


def count_csv(filepath: Path) -> int:
    """Count rows in CSV file (excluding header)."""
    df = pd.read_csv(filepath)
    return len(df)


def save_metadata(directory: Path, name: str, source: str, samples: int, 
                  license: str, description: str = ""):
    """Save metadata.json for a dataset."""
    metadata = {
        "name": name,
        "source": source,
        "samples": samples,
        "license": license,
        "description": description
    }
    with open(directory / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)


# ─────────────────────────────────────────────────────────────────────────────
# Download Functions
# ─────────────────────────────────────────────────────────────────────────────

def download_counterbench(force: bool = False):
    """Download CounterBench dataset."""
    print("\n[CounterBench] Downloading...")
    
    output_dir = TRAIN_DIR / "counterbench"
    output_file = output_dir / "counterbench.jsonl"
    
    if output_file.exists() and not force:
        print(f"  Exists: {output_file}")
        return
    
    if not HF_AVAILABLE:
        print("  ✗ Skipped: datasets library required")
        return
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        ds = load_dataset("CounterBench/CounterBench", split="train")
        df = ds.to_pandas()
        
        # Save as JSONL
        with open(output_file, 'w', encoding='utf-8') as f:
            for _, row in df.iterrows():
                f.write(json.dumps(row.to_dict(), default=str) + "\n")
        
        samples = count_jsonl(output_file)
        print(f"  ✓ Downloaded {samples} samples")
        
        save_metadata(
            output_dir, "CounterBench", 
            "https://huggingface.co/datasets/CounterBench/CounterBench",
            samples, "MIT",
            "Counterfactual reasoning benchmark with 1.2K questions"
        )
        
    except Exception as e:
        print(f"  ✗ Error: {e}")


def download_ccrgb(force: bool = False):
    """Download and split CCR.GB dataset."""
    print("\n[CCR.GB] Downloading and splitting...")
    
    train_dir = TRAIN_DIR / "ccrgb"
    test_dir = TEST_DIR / "ccrgb"
    train_file = train_dir / "ccrgb_train.jsonl"
    test_file = test_dir / "ccrgb_test.jsonl"
    
    if train_file.exists() and test_file.exists() and not force:
        print(f"  Exists: {train_file}")
        print(f"  Exists: {test_file}")
        return
    
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)
    
    url = SOURCES["ccrgb"]["url"]
    print(f"  Source: {url}")
    
    try:
        df = pd.read_csv(url)
        print(f"  Loaded {len(df)} rows")
        
        # Sort by Task ID
        def extract_id(task_id):
            try:
                parts = str(task_id).split('.')
                return int(parts[0]) * 10000 + int(parts[1]) * 100 + int(parts[2])
            except:
                return -1
        
        df = df.sort_values(by="Task ID", key=lambda x: x.apply(extract_id))
        df = df.reset_index(drop=True)
        
        # Apply curriculum-aware split
        test_indices = []
        for start, end in CCRGB_TEST_RANGES:
            test_indices.extend(range(start, min(end + 1, len(df))))
        
        test_df = df.iloc[test_indices].copy()
        train_df = df[~df.index.isin(test_indices)].copy()
        
        # Save train
        with open(train_file, 'w', encoding='utf-8') as f:
            for _, row in train_df.iterrows():
                f.write(json.dumps(row.to_dict(), default=str) + "\n")
        
        # Save test
        with open(test_file, 'w', encoding='utf-8') as f:
            for _, row in test_df.iterrows():
                f.write(json.dumps(row.to_dict(), default=str) + "\n")
        
        train_count = count_jsonl(train_file)
        test_count = count_jsonl(test_file)
        
        print(f"  ✓ Train: {train_count} samples")
        print(f"  ✓ Test: {test_count} samples")
        
        save_metadata(
            train_dir, "CCR.GB Train",
            "https://huggingface.co/datasets/jmaasch/compositional_causal_reasoning",
            train_count, "MIT",
            "Compositional causal reasoning training split"
        )
        
        save_metadata(
            test_dir, "CCR.GB Test",
            "https://huggingface.co/datasets/jmaasch/compositional_causal_reasoning",
            test_count, "MIT",
            "Curriculum-aware test split (400 samples)"
        )
        
    except Exception as e:
        print(f"  ✗ Error: {e}")


def generate_checksums():
    """Generate SHA-256 checksums for all data files."""
    print("\n[Checksums] Generating...")
    
    CHECKSUMS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Train checksums
    train_manifest = []
    for subdir in TRAIN_DIR.iterdir():
        if subdir.is_dir():
            for f in subdir.glob("*.jsonl"):
                h = sha256_file(f)
                # Use forward slashes for cross-platform compatibility
                rel_path = str(f.relative_to(BASE_DIR)).replace("\\", "/")
                train_manifest.append(f"{h}  {rel_path}")
    
    with open(CHECKSUMS_DIR / "train.sha256", 'w') as f:
        f.write("\n".join(train_manifest) + "\n")
    print(f"  ✓ train.sha256 ({len(train_manifest)} files)")
    
    # Test checksums
    test_manifest = []
    for subdir in TEST_DIR.iterdir():
        if subdir.is_dir():
            for ext in ["*.jsonl", "*.csv"]:
                for f in subdir.glob(ext):
                    h = sha256_file(f)
                    # Use forward slashes for cross-platform compatibility
                    rel_path = str(f.relative_to(BASE_DIR)).replace("\\", "/")
                    test_manifest.append(f"{h}  {rel_path}")
    
    with open(CHECKSUMS_DIR / "test.sha256", 'w') as f:
        f.write("\n".join(test_manifest) + "\n")
    print(f"  ✓ test.sha256 ({len(test_manifest)} files)")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Prepare causal reasoning datasets")
    parser.add_argument("--force-download", action="store_true",
                        help="Force re-download all datasets")
    parser.add_argument("--verify", action="store_true",
                        help="Verify checksums only")
    args = parser.parse_args()
    
    print("=" * 60)
    print("CAUSAL REASONING BENCHMARK - DATA PREPARATION")
    print("=" * 60)
    
    if args.verify:
        # Verification mode
        from verify import verify_checksums
        success = verify_checksums()
        sys.exit(0 if success else 1)
    
    # Download datasets
    download_counterbench(args.force_download)
    download_ccrgb(args.force_download)
    
    # Generate checksums
    generate_checksums()
    
    print("\n" + "=" * 60)
    print("✓ COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
