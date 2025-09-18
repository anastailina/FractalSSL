# FractalSSL

Modular research toolkit for exploring fractal self-supervised learning on biological time-series.

## Repository layout

```
FractalSSL/
├── configs/                 # JSON experiment descriptors (datasets, models, training)
├── notebooks/
│   ├── pipelines/           # Reproducible end-to-end experiment drivers
│   ├── templates/           # Minimal scaffold for new projects
│   └── legacy/              # Original exploratory notebooks (kept for reference)
├── scripts/                 # Command-line entry points for data prep, training, evaluation
├── src/fractal_ssl/         # Python package with datasets, models, training loops
├── legacy/                  # Archived monolithic scripts from earlier iterations
├── requirements.txt         # Runtime dependencies
└── pyproject.toml           # Optional packaging metadata
```

## Quick start (PAMAP2 → TS2Vec → Fractal-SSL)

1. **Install dependencies** inside a fresh Python ≥3.9 environment:

   ```bash
   pip install -r requirements.txt
   ```

2. **Preprocess the dataset** into cached sliding windows:

   ```bash
   python scripts/prepare_data.py pamap2 \
       --data-root /path/to/PAMAP2_Dataset \
       --cache-dir cache/pamap2
   ```

3. **Pre-train the TS2Vec backbone** using the provided configuration:

   ```bash
   python scripts/train_ssl.py \
       --config configs/pamap2_ts2vec_fractal.json \
       --stage backbone
   ```

4. **Fine-tune with the fractal SSL objective** (provide the best backbone checkpoint if different):

   ```bash
   python scripts/train_ssl.py \
       --config configs/pamap2_ts2vec_fractal.json \
       --stage fractal_ssl \
       --backbone-checkpoint runs/pamap2/ts2vec/ts2vec_epoch050.pt
   ```

5. **Evaluate via a linear probe** on the cached representations:

   ```bash
   python scripts/evaluate_linear.py \
       --config configs/pamap2_ts2vec_fractal.json \
       --checkpoint runs/pamap2/fractal/fractal_epoch100.pt
   ```

The JSON configuration encapsulates dataset, model and training hyperparameters. Copy it and adjust values to sweep over architectures (e.g. `simclr`) or datasets (e.g. `pads`).

## Working with other datasets

* **PADS:** export your preprocessing pipeline to `cache/pads/pads_windows.npz` (see `notebooks/legacy/PADs_FractalSSL.ipynb`), then reuse `configs/pads_ts2vec_fractal.json` with the scripts above.
* **New datasets:** implement a dedicated `prepare` function under `src/fractal_ssl/data/` (see `pamap2.py`), register it in the dataset registry and author a config JSON that points to the new cache.

## Notebooks

* `notebooks/pipelines/pamap2_ts2vec_fractal.ipynb` mirrors the quick-start sequence inside Jupyter and is useful for interactive debugging.
* `notebooks/pipelines/pads_ts2vec_fractal.ipynb` is a ready-to-run template once the PADS cache is prepared.
* `notebooks/templates/experiment_template.ipynb` provides a lightweight scaffold for custom experiments.
* Legacy exploratory notebooks remain available under `notebooks/legacy/` for reference but are no longer part of the primary workflow.

## Extending the toolkit

* Add new backbones or SSL wrappers by registering builders in `src/fractal_ssl/models/__init__.py`.
* Implement custom training schedules inside `src/fractal_ssl/training/loops.py`.
* Create additional evaluation routines under `src/fractal_ssl/evaluation/`.

The codebase is intentionally lightweight so that future experiments—new datasets, augmentations or fractal objectives—can re-use the same infrastructure without rewriting notebooks from scratch.
