# Data Card

## Dataset Overview

The Causal Reasoning Benchmark provides curated datasets for training and evaluating causal reasoning capabilities in Large Language Models.

## Dataset Composition

### Training Data (51,800 samples)

| Dataset | Samples | Format | Task Type |
|---------|---------|--------|-----------|
| CCR.GB | 49,600 | JSONL | Compositional causal reasoning |
| CounterBench | 1,200 | JSONL | Counterfactual questions |

### Evaluation Data (2,833 samples)

| Dataset | Samples | Format | Task Type |
|---------|---------|--------|-----------|
| CCR.GB Test | 400 | JSONL | Compositional causal reasoning |
| CLadder | 1,278 | CSV | Causal hierarchy questions |
| Corr2Cause | 400 | CSV | Correlation to causation |
| CauSciBench | 155 | CSV | Scientific causal reasoning |
| Synthetic | 600 | JSONL | Generated causal scenarios |

## Data Sources

| Dataset | Original Source | Access |
|---------|-----------------|--------|
| CCR.GB | Maasch et al. | HuggingFace |
| CounterBench | CounterBench Team | HuggingFace |
| CLadder | causalNLP | HuggingFace |
| Corr2Cause | causalNLP | HuggingFace |
| CauSciBench | causalNLP | GitHub |

## Intended Use

- **Primary**: Evaluating LLM causal reasoning capabilities
- **Secondary**: Fine-tuning models for causal tasks

## Limitations

- CCR.GB uses simulated clinical notes, not real patient data
- Synthetic dataset may not capture all real-world causal complexities
- Counterfactual questions assume ground-truth SCM

## Ethics

All datasets are used in accordance with their original licenses. No personally identifiable information is included.

## Maintenance

Checksums are provided for data integrity verification. Report issues via GitHub.
