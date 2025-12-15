# ML Experimental Results Directory

This directory contains results from machine learning experiments.

## Experiments

### 1. Data Scarcity Analysis (`data_scarcity_results_*.json`)
Tests model performance with varying amounts of training data.

**Key Metrics**:
- R² scores across different data percentages (20%, 40%, 60%, 80%, 100%)
- Training time
- Model convergence

### 2. Random Drop Analysis (`random_drop_results_*.json`)
Tests model robustness to random data point removal.

**Key Metrics**:
- R² scores with different drop rates (0%, 10%, 20%, 30%, 40%, 50%)
- Degradation curves
- Stability metrics

## File Format

```json
{
  "experiment_type": "data_scarcity" | "random_drop",
  "timestamp": "YYYYMMDD_HHMMSS",
  "parameters": { ... },
  "results": {
    "baseline": { "r2": 0.95, ... },
    "variations": [ ... ]
  }
}
```

## Generating Results

```bash
# Data scarcity experiment
python game_theory_data_scarcity.py

# Random drop experiment  
python game_theory_random_drop.py
```

## Visualization

Generate visualizations from results:

```bash
python visualize_results.py
```

Figures will be saved to `output/figures/`

## Notes

⚠️ **Reproducibility**: Results depend on:
- Random seed
- Dataset split
- Hardware (GPU vs CPU)
- PyTorch version

✅ **Sample Results**: Representative samples are included for reference
