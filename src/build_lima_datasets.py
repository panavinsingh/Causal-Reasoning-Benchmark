#!/usr/bin/env python3
"""
build_lima_datasets.py - Generate LIMA-Quality Training Datasets

Creates two publication-grade datasets:
- LIMA-1K: 1,000 high-quality CCR.GB samples (quality scoring + hash filtering)
- LIMA-500: 500 mixed samples (250 CCR.GB + 250 CounterBench)

Scoring formula: score = 0.5*difficulty + 0.3*pearl_level + 0.2*novelty

Requirements:
    pip install pandas sentence-transformers scikit-learn

Usage:
    python src/build_lima_datasets.py
"""

import hashlib
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

# Optional: sentence-transformers for novelty scoring
try:
    from sentence_transformers import SentenceTransformer
    HAS_EMBEDDINGS = True
except ImportError:
    HAS_EMBEDDINGS = False
    print("WARNING: sentence-transformers not installed. Novelty will default to 0.5.")

BASE_DIR = Path(__file__).parent.parent
TRAIN_DIR = BASE_DIR / "train"
TEST_DIR = BASE_DIR / "test"
CHECKSUMS_DIR = BASE_DIR / "checksums"

# Source URLs
CCRGB_URL = "https://huggingface.co/datasets/jmaasch/compositional_causal_reasoning/resolve/main/clinical_notes_v0/clinical_notes_v0.csv"

# Sampling parameters
LIMA_1K_TARGET = 1_000
LIMA_250_TARGET = 250
CB_250_TARGET = 250
SEED_RULE = 17
TRAIN_MODULO = 3  # sha256 % 17 == 3 for selection

# Weights for quality scoring
W_DIFFICULTY = 0.5
W_PEARL = 0.3
W_NOVELTY = 0.2


def sha256_int(s: str) -> int:
    """Get integer from SHA-256 hash."""
    return int(hashlib.sha256(s.encode('utf-8')).hexdigest(), 16)


def sha256_file(path: Path) -> str:
    """Compute SHA-256 checksum of a file."""
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()


def extract_task_id_numeric(task_id: str) -> int:
    """Extract numeric index from Task ID for sorting."""
    try:
        parts = str(task_id).split('.')
        return int(parts[0]) * 10000 + int(parts[1]) * 100 + int(parts[2])
    except:
        return -1


def get_pearl_level(row: pd.Series) -> int:
    """Extract Pearl level from row (1=Association, 2=Intervention, 3=Counterfactual)."""
    # Check Question type column
    qtype = str(row.get('Question type', '')).lower()
    if 'counterfactual' in qtype:
        return 3
    elif 'intervention' in qtype:
        return 2
    else:
        return 1


def get_difficulty(row: pd.Series) -> float:
    """Compute difficulty score from DAG complexity (0-1 normalized)."""
    # Use DAG node count as proxy for difficulty
    dag_str = str(row.get('DAG', ''))
    node_count = dag_str.count('->')
    # Normalize: assume 1-10 edges typical
    return min(1.0, node_count / 10.0)


def compute_novelty_scores(prompts: list[str], k: int = 5) -> np.ndarray:
    """Compute novelty scores using sentence embeddings.
    
    Novelty = 1 - average cosine similarity to k nearest neighbors.
    """
    if not HAS_EMBEDDINGS:
        return np.full(len(prompts), 0.5)
    
    print("    Loading embedding model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    print(f"    Encoding {len(prompts)} prompts...")
    embeddings = model.encode(prompts, show_progress_bar=True, batch_size=64)
    
    print(f"    Computing k={k} nearest neighbor similarities...")
    nn = NearestNeighbors(n_neighbors=k+1, metric='cosine')
    nn.fit(embeddings)
    distances, _ = nn.kneighbors(embeddings)
    
    # Average distance to k neighbors (excluding self)
    avg_distances = distances[:, 1:].mean(axis=1)
    
    # Novelty = distance (higher distance = more novel)
    # Normalize to 0-1
    novelty = (avg_distances - avg_distances.min()) / (avg_distances.max() - avg_distances.min() + 1e-8)
    
    return novelty


def passes_hash_filter(prompt: str) -> bool:
    """Check if prompt passes deterministic hash filter."""
    return sha256_int(prompt) % SEED_RULE == TRAIN_MODULO


def main():
    print("=" * 70)
    print("LIMA DATASET GENERATION")
    print(f"  LIMA-1K Target: {LIMA_1K_TARGET} CCR.GB samples")
    print(f"  LIMA-500 Target: {LIMA_250_TARGET} CCR.GB + {CB_250_TARGET} CounterBench")
    print(f"  Scoring: {W_DIFFICULTY}*difficulty + {W_PEARL}*pearl + {W_NOVELTY}*novelty")
    print("=" * 70)
    
    # =========================================================================
    # STEP 1: Load test IDs to exclude
    # =========================================================================
    print("\n[1/8] Loading test IDs...")
    test_ids_file = TEST_DIR / "ccrgb" / "ids_test_400.txt"
    
    if test_ids_file.exists():
        with open(test_ids_file, 'r') as f:
            test_ids = set(line.strip() for line in f if line.strip())
        print(f"    Loaded {len(test_ids)} test IDs to exclude")
    else:
        print("    ERROR: Test IDs file not found!")
        return
    
    # =========================================================================
    # STEP 2: Download and prepare CCR.GB
    # =========================================================================
    print("\n[2/8] Downloading CCR.GB from HuggingFace...")
    df = pd.read_csv(CCRGB_URL)
    print(f"    Loaded {len(df)} total rows")
    
    # Sort deterministically by Task ID
    df = df.sort_values(by="Task ID", key=lambda x: x.apply(extract_task_id_numeric))
    df = df.reset_index(drop=True)
    
    # Exclude test IDs
    df_train = df[~df['Task ID'].isin(test_ids)].copy()
    print(f"    After excluding test IDs: {len(df_train)} candidates")
    
    # Get prompt column
    prompt_col = "Sample context" if "Sample context" in df_train.columns else df_train.columns[0]
    
    # =========================================================================
    # STEP 3: Compute quality scores
    # =========================================================================
    print("\n[3/8] Computing quality scores...")
    
    # Difficulty
    df_train['difficulty'] = df_train.apply(get_difficulty, axis=1)
    print(f"    Difficulty: min={df_train['difficulty'].min():.2f}, max={df_train['difficulty'].max():.2f}")
    
    # Pearl level (normalize to 0-1: L1=0.33, L2=0.67, L3=1.0)
    df_train['pearl_level'] = df_train.apply(get_pearl_level, axis=1)
    df_train['pearl_norm'] = df_train['pearl_level'] / 3.0
    print(f"    Pearl levels: {df_train['pearl_level'].value_counts().to_dict()}")
    
    # Novelty (via embeddings)
    prompts = df_train[prompt_col].fillna('').tolist()
    novelty_scores = compute_novelty_scores(prompts)
    df_train['novelty'] = novelty_scores
    print(f"    Novelty: min={novelty_scores.min():.3f}, max={novelty_scores.max():.3f}")
    
    # Combined score
    df_train['quality_score'] = (
        W_DIFFICULTY * df_train['difficulty'] +
        W_PEARL * df_train['pearl_norm'] +
        W_NOVELTY * df_train['novelty']
    )
    print(f"    Quality scores: min={df_train['quality_score'].min():.3f}, max={df_train['quality_score'].max():.3f}")
    
    # =========================================================================
    # STEP 4: Apply hash filter and select LIMA-1K
    # =========================================================================
    print("\n[4/8] Selecting LIMA-1K CCR.GB...")
    
    # Filter by hash
    df_train['passes_hash'] = df_train[prompt_col].apply(passes_hash_filter)
    df_filtered = df_train[df_train['passes_hash']].copy()
    print(f"    Samples passing hash filter: {len(df_filtered)}")
    
    # Sort by quality score descending
    df_filtered = df_filtered.sort_values('quality_score', ascending=False)
    
    # Select top 1000
    lima1k_df = df_filtered.head(LIMA_1K_TARGET)
    lima1k_ids = set(lima1k_df['Task ID'].tolist())
    print(f"    Selected LIMA-1K: {len(lima1k_df)} samples")
    
    # =========================================================================
    # STEP 5: Select next 250 for LIMA-500 CCR.GB portion
    # =========================================================================
    print("\n[5/8] Selecting LIMA-500 CCR.GB portion (next 250)...")
    
    remaining_df = df_filtered[~df_filtered['Task ID'].isin(lima1k_ids)]
    lima250_ccrgb_df = remaining_df.head(LIMA_250_TARGET)
    print(f"    Selected LIMA-250 CCR.GB: {len(lima250_ccrgb_df)} samples")
    
    # =========================================================================
    # STEP 6: Select 250 CounterBench samples
    # =========================================================================
    print("\n[6/8] Selecting CounterBench 250...")
    
    cb_file = TRAIN_DIR / "counterbench" / "counterbench.jsonl"
    if not cb_file.exists():
        print(f"    ERROR: {cb_file} not found!")
        return
    
    cb_rows = []
    with open(cb_file, 'r', encoding='utf-8') as f:
        for line in f:
            cb_rows.append(json.loads(line))
    
    print(f"    Loaded {len(cb_rows)} CounterBench samples")
    
    # First, try hash filter - but if insufficient, take deterministically by hash order
    cb_hash_qualified = []
    cb_remaining = []
    for row in cb_rows:
        question = row.get('question', '')
        if passes_hash_filter(question):
            cb_hash_qualified.append(row)
        else:
            cb_remaining.append(row)
    
    print(f"    Hash-qualified: {len(cb_hash_qualified)} samples")
    
    # If we have enough hash-qualified, use those
    if len(cb_hash_qualified) >= CB_250_TARGET:
        cb_selected = cb_hash_qualified[:CB_250_TARGET]
    else:
        # Use all hash-qualified + fill with sorted remaining (by hash for determinism)
        cb_remaining_sorted = sorted(
            cb_remaining,
            key=lambda r: sha256_int(r.get('question', ''))
        )
        needed = CB_250_TARGET - len(cb_hash_qualified)
        cb_selected = cb_hash_qualified + cb_remaining_sorted[:needed]
    
    print(f"    Selected CounterBench: {len(cb_selected)} samples")
    
    # =========================================================================
    # STEP 7: Save all datasets
    # =========================================================================
    print("\n[7/8] Saving datasets...")
    
    # LIMA-1K CCR.GB
    lima1k_file = TRAIN_DIR / "ccrgb" / "ccrgb_lima1k.jsonl"
    with open(lima1k_file, 'w', encoding='utf-8') as f:
        for _, row in lima1k_df.iterrows():
            # Remove scoring columns before saving
            row_dict = row.drop(['difficulty', 'pearl_level', 'pearl_norm', 'novelty', 'quality_score', 'passes_hash']).to_dict()
            f.write(json.dumps(row_dict, default=str) + "\n")
    print(f"    Saved: {lima1k_file.name} ({len(lima1k_df)} samples)")
    
    # LIMA-250 CCR.GB
    lima250_ccrgb_file = TRAIN_DIR / "ccrgb" / "ccrgb_lima250.jsonl"
    with open(lima250_ccrgb_file, 'w', encoding='utf-8') as f:
        for _, row in lima250_ccrgb_df.iterrows():
            row_dict = row.drop(['difficulty', 'pearl_level', 'pearl_norm', 'novelty', 'quality_score', 'passes_hash']).to_dict()
            f.write(json.dumps(row_dict, default=str) + "\n")
    print(f"    Saved: {lima250_ccrgb_file.name} ({len(lima250_ccrgb_df)} samples)")
    
    # CounterBench 250
    cb_250_file = TRAIN_DIR / "counterbench" / "cb_250.jsonl"
    with open(cb_250_file, 'w', encoding='utf-8') as f:
        for row in cb_selected:
            f.write(json.dumps(row) + "\n")
    print(f"    Saved: {cb_250_file.name} ({len(cb_selected)} samples)")
    
    # Scores CSV (full candidate pool)
    scores_file = TRAIN_DIR / "ccrgb" / "ccrgb_scores.csv"
    df_train[['Task ID', 'difficulty', 'pearl_level', 'novelty', 'quality_score', 'passes_hash']].to_csv(
        scores_file, index=False
    )
    print(f"    Saved: {scores_file.name} ({len(df_train)} rows)")
    
    # =========================================================================
    # STEP 8: Generate checksums
    # =========================================================================
    print("\n[8/8] Generating checksums...")
    CHECKSUMS_DIR.mkdir(exist_ok=True)
    
    # LIMA-1K checksum
    lima1k_checksum = sha256_file(lima1k_file)
    with open(CHECKSUMS_DIR / "lima1k.sha256", 'w') as f:
        f.write(f"{lima1k_checksum}  train/ccrgb/ccrgb_lima1k.jsonl\n")
    print(f"    LIMA-1K: {lima1k_checksum[:16]}...")
    
    # LIMA-500 checksums (combined)
    lima250_checksum = sha256_file(lima250_ccrgb_file)
    cb250_checksum = sha256_file(cb_250_file)
    with open(CHECKSUMS_DIR / "lima500.sha256", 'w') as f:
        f.write(f"{lima250_checksum}  train/ccrgb/ccrgb_lima250.jsonl\n")
        f.write(f"{cb250_checksum}  train/counterbench/cb_250.jsonl\n")
    print(f"    LIMA-250 CCR.GB: {lima250_checksum[:16]}...")
    print(f"    CB-250: {cb250_checksum[:16]}...")
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "=" * 70)
    print("GENERATION COMPLETE")
    print("=" * 70)
    print(f"  LIMA-1K CCR.GB:      {len(lima1k_df):,} samples")
    print(f"  LIMA-250 CCR.GB:     {len(lima250_ccrgb_df):,} samples")
    print(f"  CounterBench 250:    {len(cb_selected):,} samples")
    print(f"  LIMA-500 Total:      {len(lima250_ccrgb_df) + len(cb_selected):,} samples")
    print("-" * 70)
    print("Output files:")
    print(f"  - {lima1k_file}")
    print(f"  - {lima250_ccrgb_file}")
    print(f"  - {cb_250_file}")
    print(f"  - {scores_file}")
    print(f"  - {CHECKSUMS_DIR / 'lima1k.sha256'}")
    print(f"  - {CHECKSUMS_DIR / 'lima500.sha256'}")
    print("=" * 70)


if __name__ == "__main__":
    main()
