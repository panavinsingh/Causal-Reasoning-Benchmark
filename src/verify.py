#!/usr/bin/env python3
"""
verify.py - Dataset Verification

Verifies dataset integrity through checksum validation and overlap detection.

Usage:
    python src/verify.py                     # Run all checks
    python src/verify.py --check-structure   # Verify file structure
    python src/verify.py --check-integrity   # Verify checksums
    python src/verify.py --check-overlap     # Check train/test overlap
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Set

BASE_DIR = Path(__file__).parent.parent
TRAIN_DIR = BASE_DIR / "train"
TEST_DIR = BASE_DIR / "test"
CHECKSUMS_DIR = BASE_DIR / "checksums"

# Expected structure
EXPECTED_TRAIN = ["ccrgb", "counterbench"]
EXPECTED_TEST = ["ccrgb", "cladder", "corr2cause", "causcibench", "synthetic"]


def sha256_file(filepath: Path) -> str:
    """Compute SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def check_structure() -> bool:
    """Verify expected directory structure exists."""
    print("\n[Structure Check]")
    all_ok = True
    
    # Check train directories
    for name in EXPECTED_TRAIN:
        path = TRAIN_DIR / name
        if path.exists():
            print(f"  ✓ train/{name}/")
        else:
            print(f"  ✗ train/{name}/ MISSING")
            all_ok = False
    
    # Check test directories
    for name in EXPECTED_TEST:
        path = TEST_DIR / name
        if path.exists():
            print(f"  ✓ test/{name}/")
        else:
            print(f"  ✗ test/{name}/ MISSING")
            all_ok = False
    
    return all_ok


def verify_checksums() -> bool:
    """Verify all file checksums match manifests."""
    print("\n[Checksum Verification]")
    all_ok = True
    
    for manifest_name in ["train.sha256", "test.sha256"]:
        manifest_path = CHECKSUMS_DIR / manifest_name
        if not manifest_path.exists():
            print(f"  ✗ {manifest_name}: Not found")
            all_ok = False
            continue
        
        with open(manifest_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                expected_hash, rel_path = line.split("  ", 1)
                # Normalize path separators for cross-platform compatibility
                rel_path = rel_path.replace("\\", "/")
                filepath = BASE_DIR / rel_path
                
                if not filepath.exists():
                    print(f"  ✗ {rel_path}: File missing")
                    all_ok = False
                    continue
                
                actual_hash = sha256_file(filepath)
                if actual_hash == expected_hash:
                    print(f"  ✓ {rel_path}")
                else:
                    print(f"  ✗ {rel_path}: Hash mismatch")
                    all_ok = False
    
    return all_ok


def load_ids_from_jsonl(filepath: Path) -> Set[str]:
    """Extract unique IDs from JSONL file."""
    ids = set()
    id_fields = ['Task ID', 'id', 'question_id', 'dag_id', 'sample_id']
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                for field in id_fields:
                    if field in row and row[field] is not None:
                        ids.add(str(row[field]))
                        break
    return ids


def check_overlap() -> bool:
    """Check for train/test ID overlap."""
    print("\n[Overlap Check]")
    
    # Collect train IDs
    train_ids: Set[str] = set()
    for subdir in TRAIN_DIR.iterdir():
        if subdir.is_dir():
            for f in subdir.glob("*.jsonl"):
                ids = load_ids_from_jsonl(f)
                train_ids.update(ids)
                print(f"  Train IDs from {f.name}: {len(ids)}")
    
    # Collect test IDs
    test_ids: Set[str] = set()
    for subdir in TEST_DIR.iterdir():
        if subdir.is_dir():
            for f in subdir.glob("*.jsonl"):
                ids = load_ids_from_jsonl(f)
                test_ids.update(ids)
                print(f"  Test IDs from {f.name}: {len(ids)}")
    
    # Check overlap
    overlap = train_ids & test_ids
    if len(overlap) == 0:
        print(f"\n  ✓ Zero ID overlap")
        return True
    else:
        print(f"\n  ✗ {len(overlap)} overlapping IDs detected!")
        return False


def main():
    parser = argparse.ArgumentParser(description="Verify dataset integrity")
    parser.add_argument("--check-structure", action="store_true",
                        help="Check directory structure")
    parser.add_argument("--check-integrity", action="store_true",
                        help="Verify checksums")
    parser.add_argument("--check-overlap", action="store_true",
                        help="Check train/test overlap")
    args = parser.parse_args()
    
    print("=" * 60)
    print("CAUSAL REASONING BENCHMARK - VERIFICATION")
    print("=" * 60)
    
    # If no specific checks, run all
    run_all = not (args.check_structure or args.check_integrity or args.check_overlap)
    
    results = []
    
    if run_all or args.check_structure:
        results.append(("Structure", check_structure()))
    
    if run_all or args.check_integrity:
        results.append(("Checksums", verify_checksums()))
    
    if run_all or args.check_overlap:
        results.append(("Overlap", check_overlap()))
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    all_passed = True
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False
    
    print("=" * 60)
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
