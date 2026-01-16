#!/usr/bin/env python3
"""Final audit for ICML/ICLR readiness."""

import json
import hashlib
from pathlib import Path
from collections import Counter

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()

print("=" * 70)
print("ICML/ICLR FINAL AUDIT")
print("=" * 70)

issues = []

# 1. File existence
print("\n[1/8] File Existence...")
files = [
    "train/ccrgb/ccrgb_lima1k.jsonl",
    "train/lima_500.jsonl", 
    "train/VERSION.txt",
    "checksums/lima1k.sha256",
    "checksums/lima500.sha256",
]
for f in files:
    status = "OK" if Path(f).exists() else "MISSING"
    print(f"  {status}: {f}")
    if not Path(f).exists():
        issues.append(f"Missing: {f}")

# 2. Row counts
print("\n[2/8] Row Counts...")
lima1k = [json.loads(l) for l in open("train/ccrgb/ccrgb_lima1k.jsonl", encoding="utf-8")]
lima500 = [json.loads(l) for l in open("train/lima_500.jsonl", encoding="utf-8")]
print(f"  LIMA-1K: {len(lima1k)} (expected 1000)")
print(f"  LIMA-500: {len(lima500)} (expected 500)")
if len(lima1k) != 1000: issues.append("LIMA-1K count wrong")
if len(lima500) != 500: issues.append("LIMA-500 count wrong")

# 3. Source distribution
print("\n[3/8] LIMA-500 Sources...")
sources = Counter(r.get("source") for r in lima500)
for s, c in sources.items():
    print(f"  {s}: {c}")

# 4. Checksums
print("\n[4/8] Checksums...")
h1k = sha256_file("train/ccrgb/ccrgb_lima1k.jsonl")
h500 = sha256_file("train/lima_500.jsonl")
exp1k = open("checksums/lima1k.sha256").read().split()[0]
exp500 = open("checksums/lima500.sha256").read().split()[0]
m1k = "MATCH" if h1k == exp1k else "MISMATCH"
m500 = "MATCH" if h500 == exp500 else "MISMATCH"
print(f"  LIMA-1K: {m1k}")
print(f"  LIMA-500: {m500}")
if m1k != "MATCH": issues.append("LIMA-1K checksum fail")
if m500 != "MATCH": issues.append("LIMA-500 checksum fail")

# 5. ID uniqueness
print("\n[5/8] ID Uniqueness...")
ids1k = set(r.get("Task ID") for r in lima1k)
print(f"  LIMA-1K unique: {len(ids1k)}/{len(lima1k)}")
if len(ids1k) != len(lima1k): issues.append("Duplicate IDs in LIMA-1K")

# 6. Test exclusion
print("\n[6/8] Test Exclusion...")
test_ids = set(l.strip() for l in open("test/ccrgb/ids_test_400.txt"))
overlap = ids1k & test_ids
print(f"  LIMA-1K vs Test: {len(overlap)} overlap")
if overlap: issues.append("Test contamination!")

# 7. Disjointness
print("\n[7/8] LIMA-1K/500 Disjoint...")
ids500 = set(r.get("Task ID") for r in lima500 if r.get("source") == "CCR.GB")
dupe = ids1k & ids500
print(f"  Overlap: {len(dupe)}")
if dupe: issues.append("LIMA-1K/500 overlap!")

# 8. VERSION.txt
print("\n[8/8] Source Pinning...")
ver = Path("train/VERSION.txt").read_text()
print(f"  Commit hash: {'Yes' if '564e27ae' in ver else 'No'}")
print(f"  License info: {'Yes' if 'GPL' in ver else 'No'}")

print("\n" + "=" * 70)
if issues:
    print(f"FAILED: {len(issues)} issue(s)")
    for i in issues: print(f"  - {i}")
else:
    print("ALL CHECKS PASSED - Ready for ICML/ICLR")
print("=" * 70)
