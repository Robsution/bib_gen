# BIB family generation

This repository contains a conditional generative baseline for beam-induced-background
(BIB) particle families. A family is represented as a parent state and a variable-size set
of daughter particles. The current baseline factorizes generation into

$$
p(N\mid c)\,p(\{s_i,k_i\}_{i=1}^{N}\mid N,c),
$$

where `c` is the parent position, `N` is daughter multiplicity, `s_i` is particle species,
and `k_i` contains eight continuous particle features.

The first factor is a conditional categorical model. The second is a conditioned
spectral-normalized GAN with a DeepSets discriminator.

## Repository layout

```text
configs/                 Reproducible experiment configurations
src/dataset.py           Processed-family dataset
src/models/              Multiplicity and conditioned-GAN models
src/monitoring.py        Training diagnostics
scripts/data/            Preprocessing, filtering, and validation commands
scripts/analysis/        Real-data and real/generated composition plots
train.py                 Two-stage baseline training
train_multiplicity.py    Development comparison of multiplicity models
evaluate.py              Particle-level real/generated distributions
tests/                   Unit tests using synthetic data
```

Generated datasets, checkpoints, plots, logs, and Python bytecode are intentionally ignored
by Git.

## Environment

Python 3.10 or newer is expected.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

For development checks:

```bash
python -m pip install -r requirements-dev.txt
ruff check .
pytest
```

## Data representation

The raw CSV has one family per row:

- three parent coordinates `(x, y, z)`;
- a variable number of 11-value daughter records.

The current preprocessing retains the daughter PDG code and
`(energy, x, y, z, px, py, pz, t)`. The final two raw daughter values are not used by the
baseline; their offsets are recorded in the scaler metadata.

Processed files use the following layout:

- `bib_data_parents.npy`: `(families, 3)`, MinMax-scaled to `[-1, 1]`;
- `bib_data_daughters.npy`: `(families, max_daughters, 1 + 8 + n_species)`;
  - daughter column 0: activity mask;
  - daughter columns 1–8: Min-max scaled continuous features to `[-1, 1]`;
  - remaining columns: one-hot PDG category;
- `bib_data_scalers.json` and `bib_data_parent_scalers.json`: feature order, ranges, and
  PDG-category mapping.

Create and validate the full processed dataset with

```bash
python -m scripts.data.preprocess RAW.csv --output-dir data/processed/full
python -m scripts.data.validate_dataset --data-dir data/processed/full
```

One might filter their data for a small training dataset via filtering:

```bash
python -m scripts.data.filter_dataset \
  --input-dir data/raw/your_data \
  --output-dir data/processed/your_data_10 \
  --max-daughters 10
```

## Training and evaluation

Train the current baseline with a fixed configuration:

```bash
python train.py --config configs/baseline_10.yaml
python train.py --config configs/baseline_50.yaml
```

Existing multiplicity checkpoints are reused unless `--retrain-multiplicity` is supplied.
Resume a saved GAN checkpoint with `--resume-epoch EPOCH`.

Generate the particle-level comparison dashboard:

```bash
python evaluate.py --config configs/baseline_50.yaml
```

Useful exploratory commands include

```bash
python -m scripts.analysis.family_composition --data-dir data/processed/large50
python -m scripts.analysis.family_composition \
  --data-dir data/processed/large50 --family-size 1
python -m scripts.analysis.parent_geometry --data-dir data/processed/large50
python train_multiplicity.py --config configs/multiplicity_50.yaml
```

`compare_family_composition` conditions the generator on each real family's observed
multiplicity. It therefore tests species/family generation at fixed `N`, not the complete
two-stage pipeline.

## Current methodological limitations

- The preprocessing script fits MinMax ranges on its entire input. Benchmark datasets must
  be split before fitting preprocessing statistics to avoid test-set leakage.
- `train_multiplicity.py` is a development comparison on the training sample. It is not yet
  the held-out multiplicity benchmark.
- Generated slot masks are soft during training and thresholded at 0.5 during evaluation.
- No parent-to-recorded-daughters four-momentum constraint is imposed; the recorded scoring
  surface is not assumed to be a closed physical system.

These limitations should be resolved or explicitly frozen when Benchmark v1.0 is defined.
