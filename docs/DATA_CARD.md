# Data Card

## Dataset Overview

The Causal Reasoning Benchmark provides curated datasets for training and evaluating causal reasoning capabilities in Large Language Models.

## Training Data

### Standard Splits

| Dataset | Samples | Format | Task Type |
|---------|---------|--------|-----------|
| CCR.GB | 4,000 | JSONL | Compositional causal reasoning |
| CounterBench | 1,000 | JSONL | Counterfactual questions |

### LIMA-Quality Subsets (Recommended for Fine-tuning)

| Dataset | Samples | Selection | Location |
|---------|---------|-----------|----------|
| **LIMA-1K** | 1,000 | Top 1K from 50K CCR.GB (quality scored) | `train/ccrgb/ccrgb_lima1k.jsonl` |
| **LIMA-500** | 500 | 250 CCR.GB + 250 CounterBench | `train/lima_500.jsonl` |

**LIMA Selection Methodology:**
- Quality score: `0.5 × difficulty + 0.3 × pearl_level + 0.2 × novelty`
- Novelty computed via sentence-transformer embeddings (`all-MiniLM-L6-v2`)
- Hash filter: `SHA256(prompt) % 17 == 3`
- Full methodology: [`docs/selection_report.md`](selection_report.md)

## Evaluation Data (2,833 samples)

| Dataset | Samples | Format | Task Type |
|---------|---------|--------|-----------|
| CCR.GB Test | 400 | JSONL | Compositional causal reasoning |
| CLadder | 1,278 | CSV | Causal hierarchy questions |
| Corr2Cause | 400 | CSV | Correlation to causation |
| CauSciBench | 155 | CSV | Scientific causal reasoning |
| Synthetic | 600 | JSONL | Generated causal scenarios |

## Data Sources

| Dataset | Original Source | License | Commit |
|---------|-----------------|---------|--------|
| CCR.GB | [jmaasch/compositional_causal_reasoning](https://huggingface.co/datasets/jmaasch/compositional_causal_reasoning) | GPL-3.0 | `564e27ae` |
| CounterBench | [CounterBench/CounterBench](https://huggingface.co/datasets/CounterBench/CounterBench) | MIT | `c6225dfa` |
| CLadder | [causalnlp/CLadder](https://huggingface.co/datasets/causalnlp/CLadder) | Apache-2.0 | v1.5 |
| Corr2Cause | [causalnlp/corr2cause](https://huggingface.co/datasets/causalnlp/corr2cause) | CC-BY-4.0 | test split |
| CauSciBench | [causalNLP/CauSciBench](https://github.com/causalNLP/CauSciBench) | Apache-2.0 | main |

## Intended Use

- **Primary**: Evaluating LLM causal reasoning capabilities
- **Secondary**: Fine-tuning models for causal tasks (use LIMA subsets)

## Data Integrity

| Verification | Location |
|--------------|----------|
| LIMA-1K checksum | `checksums/lima1k.sha256` |
| LIMA-500 checksum | `checksums/lima500.sha256` |
| Contamination check | `python src/check_contamination.py` |
| Full audit | `python src/final_audit.py` |

## Limitations

- CCR.GB uses simulated clinical notes, not real patient data
- Synthetic dataset may not capture all real-world causal complexities
- Counterfactual questions assume ground-truth SCM

## Ethics

All datasets are used in accordance with their original licenses. No personally identifiable information is included.

## Maintenance

Report issues via GitHub. Run integrity checks before training/evaluation.
