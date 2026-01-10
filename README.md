# Hazelnut Defect Detection

Summary
- Small Python project for training and testing models that detect defects in hazelnut images. Contains scripts for quick experiments, autoencoder training, and a smoke-run utility to verify the environment.

Repository layout
- `hazelnut/` - dataset folders (train, test, ground_truth categories).
- `train_quick.py` - lightweight training script for quick experiments.
- `train_autoencoder.py` - train an autoencoder model (larger / longer runs).
- `smoke_run.py` - quick environment / inference smoke test.
- `check_imports.py` - helper to validate required imports.
- `code.py` - utilities and model code (project-specific logic).
- `requirements.txt` - Python dependencies.

Requirements
- Python 3.8+ (confirm with your environment)
- Install dependencies:

```bash
pip install -r requirements.txt
```

Quick start
1. Install dependencies (see above).
2. Run a smoke test to ensure environment and basic inference work:

```bash
python smoke_run.py
```

Training
- Quick experiment:

```bash
python train_quick.py
```

- Full autoencoder training:

```bash
python train_autoencoder.py
```

Notes
- Dataset is under `hazelnut/` and includes `train/`, `test/`, and `ground_truth/` subfolders organized by defect type.
- Adjust script arguments or configuration in the scripts as needed for paths, hyperparameters, and GPUs.

Troubleshooting
- If imports fail, run:

```bash
python check_imports.py
```

Contact
- If you want changes to this README or additional examples, tell me what to include.
