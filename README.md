# OrderGraph

Implementation of **Temporal relational order for fine-grained recognition with noisy labels.**

## Install

Use Python 3.11 in an isolated environment, then run from the repository root:

```bash
python -m pip install -e .
```

## Data

Use CUB-200-2011, Stanford Cars, FGVC-Aircraft, NABirds or WebFG. A CSV manifest contains `image,label,split` with zero-based labels, relative image paths and `train,val,test` roles. Optional `ambiguous` flags come from independent annotation review; `boundary` marks a held-out boundary subset, and `clean_label` preserves known labels for controlled noise. Keep images disjoint across roles.

## Workflow

```bash
python -m ordergraph prepare --input data/images.csv
python -m ordergraph train
python -m ordergraph run
python -m ordergraph evaluate
python -m ordergraph plot
```

Main commands accept `--config config.json` and `--output runs/default`. Keep each seed/condition in its own output directory. The resolved configuration is written to `settings.json`.

## Configuration

`neighbors`, `history` and `reference_ema` control local order tracking. `warmup` initializes ordinary supervision before mixture fitting. `noise` accepts `symmetric`, `pair-flip` or `provided`; `noise_rate` applies to eligible training examples. `pseudo_threshold` admits stable-neighbor soft targets. `contrastive_temperature` controls cross-view discrimination. Training epochs, batch size, learning rate and seed use the defaults shown in `config.json` where applicable.

## Outputs

`manifest.csv` preserves labels and corruption flags. `states.npz` contains state posteriors and survival from the selected checkpoint. `predictions.npz` contains clean-test probabilities. `metrics.csv` reports accuracy, ECE and, when annotations exist, noise AUROC, ambiguity retention and boundary accuracy.

## Implementation and controls

ImageNet ResNet-50 supplies representations. Candidate sets remain fixed inside a history block and history resets when membership changes. The standardized mixture signals are order survival, observed-label rank, loss EMA and view agreement. Clean, ambiguous and corrupted probabilities weight observed-label CE, logistic pair-order loss, and instance discrimination plus neighbor soft labels. Ambiguity flags protect controlled-noise generation but never fit the posterior.

Controls: `full`, `ce`, `binary`, `no-temporal`, `no-order-loss`, `confidence-only`. Choose `--variant` and use the same preparation and fitting choices in each separate output directory.

```bash
python -m ordergraph.compare --inputs "runs/*/metrics.csv" --reference full --metrics accuracy ece --unit seed
```

The comparison command averages repeated rows within the specified independent unit, bootstraps units and applies paired Wilcoxon/Holm comparisons per metric.

