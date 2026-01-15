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
| **Train** | CCR.GB | 4,000 | GPL-3.0 | `sha256(prompt) % 12 == 3`, exclude test IDs |
| **Train** | CounterBench | 1,000 | MIT | First 1,000 samples |
| **Dev** | CCR.GB | 300 | GPL-3.0 | `sha256(prompt) % 12 == 7` |
| **Test** | CCR.GB | 400 | GPL-3.0 | Contiguous curriculum blocks |
| **Test** | CLadder | 1,278 | Apache-2.0 | `id % 8 == 0` |
| **Test** | Corr2Cause | 400 | CC-BY-4.0 | Stratified by label (seed 42) |
| **Test** | CauSciBench | 155 | Apache-2.0 | Full benchmark |
| **Test** | Synthetic | 600 | Apache-2.0 | Stratified by rung (seed 42) |

| | | | | |
|---|---|---|---|---|
| **Train Total** | | **5,000** | | |
| **Dev Total** | | **300** | | |
| **Test Total** | | **2,833** | | |

---

## Repository Structure

```
├── train/                      # Training data (5,000 samples)
│   ├── ccrgb/                  # 4,000 samples
│   │   ├── ccrgb_train.jsonl
│   │   └── metadata.json
│   └── counterbench/           # 1,000 samples
│       ├── counterbench.jsonl
│       └── metadata.json
│
├── dev/                        # Development set (300 samples)
│   ├── dev_300.jsonl
│   └── metadata.json
│
├── test/                       # Evaluation data (2,833 samples)
│   ├── ccrgb/                  # 400 samples
│   ├── cladder/                # 1,278 samples
│   ├── corr2cause/             # 400 samples
│   ├── causcibench/            # 155 samples
│   └── synthetic/              # 600 samples
│
├── src/                        # Source code
│   ├── prepare_data.py
│   ├── build_train_4k.py
│   ├── verify.py
│   └── check_contamination.py
│
├── checksums/                  # SHA-256 manifests
│   ├── train.sha256
│   └── test.sha256
│
└── docs/
    ├── CONTRIBUTING.md
    └── DATA_CARD.md
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

### Why 5K training samples?

| Concern | Our approach |
|---------|--------------|
| **Compute budget** | Pilot experiments show accuracy plateaus after ~5K examples on A100-80GB |
| **Class balance** | 4K CCR.GB samples stratified to match test split difficulty distribution |
| **Domain coverage** | 1K CounterBench (20%) adds explicit counterfactual practice |
| **Determinism** | Hash-based selection (`sha256 % 12`) ensures bit-perfect reproducibility |

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
