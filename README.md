<p align="center">
  <h1 align="center">🔬 Causal Reasoning Benchmark</h1>
  <p align="center">
    <strong>A comprehensive benchmark for evaluating causal reasoning in Large Language Models</strong>
  </p>
</p>

<p align="center">
  <a href="#installation"><img src="https://img.shields.io/badge/python-3.10+-blue.svg" alt="Python"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-CC%20BY%204.0-green.svg" alt="License"></a>
  <a href="#data"><img src="https://img.shields.io/badge/datasets-7-orange.svg" alt="Datasets"></a>
  <img src="https://img.shields.io/badge/samples-8K-purple.svg" alt="Samples">
</p>

---

## Overview

This repository provides curated training and evaluation datasets for causal reasoning research. All datasets are organized with reproducible splits, SHA-256 checksums, and automated integrity verification.

### Pearl's Causal Hierarchy Coverage

| Level | Description | Datasets |
|-------|-------------|----------|
| **L1** | Associational | CLadder, CCR.GB, Corr2Cause |
| **L2** | Interventional | CLadder, CCR.GB, CauSciBench |
| **L3** | Counterfactual | CLadder, CCR.GB, CounterBench, Synthetic |

---

## Data Summary

| Split | Dataset | Rows | License | Selection Rule |
|-------|---------|------|---------|----------------|
| **Train** | CCR.GB | 4,000 | GPL-3.0 | `sha256(prompt) % 17 == 3`, exclude test IDs |
| **Train** | CounterBench | 1,000 | MIT | Full dataset |
| **Train** | **LIMA-1K** *(CCR.GB subset)* | 1,000 | GPL-3.0 | Quality scoring + hash filter (top 1K from 50K CCR.GB) |
| **Train** | **LIMA-500** | 500 | GPL-3.0 + MIT | 250 CCR.GB + 250 CounterBench |
| **Dev** | CCR.GB | 300 | GPL-3.0 | `sha256(prompt) % 17 == 7` |
| **Test** | CCR.GB | 400 | GPL-3.0 | Contiguous curriculum blocks |
| **Test** | CLadder | 1,278 | Apache-2.0 | `id % 8 == 0` |
| **Test** | Corr2Cause | 400 | CC-BY-4.0 | Stratified by label (seed 42) |
| **Test** | CauSciBench | 155 | Apache-2.0 | Full benchmark |
| **Test** | Synthetic | 600 | Apache-2.0 | Stratified by rung (seed 42) |

| | | | | |
|---|---|---|---|---|
| **Train Total** | | **6,500** | | |
| **Dev Total** | | **300** | | |
| **Test Total** | | **2,833** | | |

---

## Repository Structure

```
├── train/                      # Training data
│   ├── ccrgb/
│   │   ├── ccrgb_train.jsonl   # 4,000 samples
│   │   ├── ccrgb_lima1k.jsonl  # 1,000 LIMA-quality samples
│   │   ├── ccrgb_scores.csv    # Quality scores for 49,600 candidates
│   │   └── metadata.json
│   ├── counterbench/
│   │   └── counterbench.jsonl  # 1,000 samples
│   ├── lima_500.jsonl          # 500 mixed (250 CCR.GB + 250 CounterBench)
│   └── VERSION.txt             # Source commit pinning
│
├── dev/                        # Development set (300 samples)
│   └── dev_300.jsonl
│
├── test/                       # Evaluation data (2,833 samples)
│   ├── ccrgb/                  # 400 samples
│   ├── cladder/                # 1,278 samples
│   ├── corr2cause/             # 400 samples
│   ├── causcibench/            # 155 samples
│   └── synthetic/              # 600 samples
│
├── src/
│   ├── build_lima_datasets.py  # LIMA dataset generation
│   ├── build_train_4k.py
│   ├── verify.py
│   ├── check_contamination.py
│   └── final_audit.py          # ICML/ICLR audit
│
├── checksums/
│   ├── lima1k.sha256
│   ├── lima500.sha256
│   └── *.sha256
│
└── docs/
    ├── selection_report.md     # LIMA methodology
    ├── DATA_CARD.md
    └── CONTRIBUTING.md
```

---

## Quick Start

```bash
git clone https://github.com/panavinsingh/Causal-Reasoning-Benchmark.git
cd Causal-Reasoning-Benchmark
pip install pandas datasets

# Verify integrity
python src/verify.py

# Check for contamination
python src/check_contamination.py

# Rebuild from scratch (optional)
python src/prepare_data.py --force-download
python src/build_train_4k.py
```

---

## Sampling Rationale

### LIMA Datasets (Recommended for Fine-tuning)

> **What is LIMA?** The [LIMA paper](https://arxiv.org/abs/2305.11206) ("Less Is More for Alignment", Zhou et al., 2023) demonstrated that fine-tuning on a small number of high-quality examples can match or exceed training on much larger datasets. Our LIMA subsets apply this principle: instead of training on all 50K CCR.GB samples, we use quality scoring to select the most valuable 1,000.

| Dataset | Samples | Selection Method |
|---------|---------|------------------|
| **LIMA-1K** | 1,000 | Top quality scores from ~50K CCR.GB |
| **LIMA-500** | 500 | 250 CCR.GB + 250 CounterBench |

**Quality Scoring Formula:**
```
score = 0.5 × difficulty + 0.3 × pearl_level + 0.2 × novelty
```

- **Difficulty**: DAG complexity (number of causal edges)
- **Pearl Level**: Causal hierarchy level (L1=Association, L2=Intervention, L3=Counterfactual)
- **Novelty**: Semantic uniqueness computed via sentence embeddings (`all-MiniLM-L6-v2`)

See [`docs/selection_report.md`](docs/selection_report.md) for full methodology.

### CCR.GB Test Split

Curriculum-aware contiguous blocks preserving compositional complexity:

| Level | Difficulty | Samples |
|-------|------------|---------|
| Associational | easy/medium/hard | 120 |
| Interventional | easy/medium/hard | 200 |
| Counterfactual | easy/medium/hard | 80 |

---

## Reproducibility

| Guarantee | Implementation |
|-----------|----------------|
| **Deterministic** | Hash-based sampling, no RNG |
| **Verifiable** | SHA-256 checksums for all files |
| **Leak-free** | ID-based overlap verification |
| **Traceable** | HuggingFace commit hashes in metadata |
| **Rebuild** | `build_train_4k.py` regenerates exact splits |

```bash
# Verify checksums
python src/verify.py --check-integrity

# Check overlap
python src/verify.py --check-overlap

# Full contamination check
python src/check_contamination.py
```

---

## Dataset Sources

| Dataset | Source | Commit/Version |
|---------|--------|----------------|
| CCR.GB | [jmaasch/compositional_causal_reasoning](https://huggingface.co/datasets/jmaasch/compositional_causal_reasoning) | `564e27ae` |
| CounterBench | [CounterBench/CounterBench](https://huggingface.co/datasets/CounterBench/CounterBench) | `c6225dfa` |
| CLadder | [causalnlp/CLadder](https://huggingface.co/datasets/causalnlp/CLadder) | v1.5 |
| Corr2Cause | [causalnlp/corr2cause](https://huggingface.co/datasets/causalnlp/corr2cause) | test split |
| CauSciBench | [causalNLP/CauSciBench](https://github.com/causalNLP/CauSciBench) | main |

---

## License

| Dataset | License | SPDX |
|---------|---------|------|
| CCR.GB | GPL-3.0 | `GPL-3.0-only` |
| CounterBench | MIT | `MIT` |
| CLadder | Apache-2.0 | `Apache-2.0` |
| Corr2Cause | CC-BY-4.0 | `CC-BY-4.0` |
| CauSciBench | Apache-2.0 | `Apache-2.0` |
| Synthetic | Apache-2.0 | `Apache-2.0` |

The benchmark infrastructure and documentation are licensed under [CC-BY-4.0](LICENSE).

---

## Citation

```bibtex
@misc{causal-reasoning-benchmark,
  title={Causal Reasoning Benchmark},
  author={Singh, Panavin},
  year={2026},
  url={https://github.com/panavinsingh/Causal-Reasoning-Benchmark}
}
```
