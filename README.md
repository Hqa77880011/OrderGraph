# OrderGraph

Implementation of **Temporal relational order for fine-grained recognition with noisy labels.**

## Installation

Requires Python >=3.10.

```bash
python -m pip install -e .
```

Inspection environment: Python 3.13.5, NumPy 2.1.3 and SciPy 1.15.3. Verification covers Python syntax, JSON/TOML and README CLI options against parser definitions; training and evaluation were not run.

## Data and preprocessing

Use CUB-200-2011, Stanford Cars, FGVC-Aircraft, NABirds or WebFG. A CSV manifest contains `image,label,split` with zero-based labels, relative image paths and `train,val,test` roles. Optional `ambiguous` flags come from independent annotation review; `boundary` marks a held-out boundary subset, and `clean_label` preserves known labels for controlled noise. Keep images disjoint across roles.

```bash
python -m ordergraph prepare --input data/images.csv --config config.json --output runs/full
```

## Training and inference

```bash
python -m ordergraph train --config config.json --output runs/full
python -m ordergraph run --config config.json --output runs/full
```

`config.json` records the method settings and seed `13`; the resolved configuration is saved as `runs/full/settings.json`. Keep separate output directories for different seeds and variants.

## Evaluation

```bash
python -m ordergraph evaluate --config config.json --output runs/full
```

## Ablations

Supported controls: `full`, `ce`, `binary`, `no-temporal`, `no-order-loss`, `confidence-only`.

Repeat the same data and configuration in a separate directory, for example:

```bash
python -m ordergraph prepare --input data/images.csv --config config.json --output runs/ce --variant ce
python -m ordergraph train --config config.json --output runs/ce --variant ce
python -m ordergraph run --config config.json --output runs/ce --variant ce
python -m ordergraph evaluate --config config.json --output runs/ce --variant ce
```

## Outputs

All main outputs are written below `runs/full/`.

`manifest.csv` preserves labels and corruption flags. `states.npz` contains state posteriors and survival from the selected checkpoint. `predictions.npz` contains clean-test probabilities. `metrics.csv` reports accuracy, ECE and, when annotations exist, noise AUROC, ambiguity retention and boundary accuracy.

```bash
python -m ordergraph plot --config config.json --output runs/full
```
