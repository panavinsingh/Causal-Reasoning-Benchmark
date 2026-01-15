# Contributing

Thank you for your interest in contributing to the Causal Reasoning Benchmark.

## Guidelines

### Adding a New Dataset

1. Add the dataset files to the appropriate directory:
   - Training data: `train/<dataset_name>/`
   - Evaluation data: `test/<dataset_name>/`

2. Create a `metadata.json` file:
   ```json
   {
     "name": "Dataset Name",
     "source": "https://source-url",
     "samples": 1000,
     "license": "License-SPDX",
     "description": "Brief description"
   }
   ```

3. Update `src/prepare_data.py` to support downloading from upstream.

4. Run verification:
   ```bash
   python src/verify.py
   ```

5. Regenerate checksums:
   ```bash
   python src/prepare_data.py --force-download
   ```

### Code Style

- Python 3.10+
- Follow PEP 8
- Use type hints
- Document functions with docstrings

### Pull Request Process

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run all verification checks
5. Submit a pull request

## Questions?

Open an issue for discussion.
