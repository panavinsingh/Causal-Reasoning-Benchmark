# LIMA Dataset Selection Report

## Overview

This document describes the selection methodology for the LIMA-quality training datasets.

| Dataset | Samples | Source |
|---------|---------|--------|
| LIMA-1K | 1,000 | CCR.GB |
| LIMA-500 | 500 | CCR.GB (250) + CounterBench (250) |

## Selection Methodology

### CCR.GB Quality Scoring

All 49,600 candidate samples (50K - 400 test IDs) were scored using:

```
quality_score = 0.5 × difficulty + 0.3 × pearl_level + 0.2 × novelty
```

| Component | Description | Range |
|-----------|-------------|-------|
| Difficulty | DAG complexity (edge count) | 0.0 - 1.0 |
| Pearl Level | Causal reasoning level (L1=0.33, L2=0.67, L3=1.0) | 0.33 - 1.0 |
| Novelty | Semantic uniqueness via k-NN distance (all-MiniLM-L6-v2) | 0.0 - 1.0 |

### Hash-Based Filtering

Deterministic selection using:
```
SHA256(prompt) % 17 == 3
```

This yielded 2,815 qualified samples from 49,600 candidates.

### LIMA-1K Selection

1. Filter 49,600 candidates by hash rule → 2,815 samples
2. Sort by quality_score descending
3. Select top 1,000 samples

### LIMA-500 Selection

**CCR.GB (250 samples):**
- Take next 250 samples from hash-qualified pool (not in LIMA-1K)

**CounterBench (250 samples):**
- Apply hash filter: 77 samples qualified
- Fill remaining 173 via deterministic hash-ordering

## Contamination Verification

| Check | Result |
|-------|--------|
| LIMA-1K ∩ Test IDs | 0 |
| LIMA-500 ∩ Test IDs | 0 |
| LIMA-1K ∩ LIMA-500 CCR.GB | 0 (disjoint) |

## Output Files

| File | Samples | Checksum (SHA-256) |
|------|---------|-------------------|
| `train/ccrgb/ccrgb_lima1k.jsonl` | 1,000 | `596db117b93bd090...` |
| `train/lima_500.jsonl` | 500 | `1ff1156145db2dba...` |
| `train/ccrgb/ccrgb_scores.csv` | 49,600 | Quality scores for all candidates |

## Reproducibility

All selections are deterministic:
- No random sampling (hash-based filtering)
- Fixed embedding model: `all-MiniLM-L6-v2`
- Source versions pinned in `train/VERSION.txt`
