# Replication Files

This directory contains Monte Carlo simulation scripts for replicating the results in *"Penalized GMM Framework for Inference on Functionals of Nonparametric Instrumental Variable Estimators"* by Edvard Bakhitov (2026). [[arXiv]](https://arxiv.org/abs/2603.29889)

## Prerequisites

Install the ADMLIV package first:

```bash
cd ..
pip install -e ".[dev]"
```

## Simulations

### Toy Model (Weighted Average Derivative)

Tests ADMLIV on a simple nonparametric model with known ground truth.

```bash
cd toy_model

# Quick test (5 runs)
python mc_wad_serial.py --k 2 --n_obs 100 --n_runs 5

# Full replication (parallel)
bash run_mc_wad.sh

# Summarize results
python summarize_wad_results.py
```

### High-Dimensional Linear Regression

Compares PGMM, RMD Lasso, and sklearn Lasso for sparse linear regression.

```bash
cd linear_model
python mc_hd_linear.py
python summarize_hd_linear_results.py
```

### High-Dimensional Linear IV

Compares PGMM and Double Lasso for linear IV with many instruments.

```bash
cd linear_iv_model
python mc_hd_linear_iv.py
python summarize_hd_linear_iv_results.py
```

### Demand Elasticity (Logit)

Estimates average own-price elasticity using simulated logit demand data.

```bash
cd demand_model/elasticities

# Quick test (5 runs)
python mc_elasticity_logit_serial.py --n_markets 100 --n_products 4 --n_runs 5

# Full replication
bash run_mc_elasticity_serial.sh

# Summarize results
python summarize_elasticity_results.py
```

## Output

Results are saved to `results/` subdirectories within each simulation folder.
These are excluded from version control; users regenerate them by running the scripts.
