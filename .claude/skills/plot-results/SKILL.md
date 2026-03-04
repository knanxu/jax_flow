---
name: plot-results
description: Generate plots and analysis from training logs and evaluation results. Use for visualizing experiment outcomes.
disable-model-invocation: false
allowed-tools: Bash, Read, Write, Grep, Glob
argument-hint: [experiment_dir_or_log_files]
---

# Plot Experiment Results

Generate publication-quality plots from training logs and evaluation results.

## Instructions

1. **Parse arguments**:
   - Extract experiment directory or log file paths from $ARGUMENTS
   - If directory, find all log files recursively
   - Support multiple formats: tensorboard, wandb, CSV, JSON

2. **Identify available metrics**:
   - Training loss curves
   - Evaluation success rates
   - Episode returns
   - Gradient norms
   - Learning rate schedules

3. **Create analysis script**:
   - Generate Python script using matplotlib/seaborn
   - Load data from logs
   - Create standard plots:
     - Learning curves (loss vs steps)
     - Success rate over time
     - Hyperparameter comparison (if multiple runs)
     - Statistical significance tests (if multiple seeds)

4. **Generate plots**:
   ```bash
   python plot_script.py --input $LOGS --output plots/
   ```

5. **Create summary report**:
   - Best performing configuration
   - Statistical comparison between methods
   - Key insights and recommendations

6. **Save outputs**:
   - Save plots to `plots/` directory
   - Generate `RESULTS.md` with embedded plots
   - Create LaTeX table for paper if requested

## Plot Types

- **Learning curves**: Loss/reward vs training steps
- **Success rate**: Evaluation performance over time
- **Ablation study**: Compare different configurations
- **Hyperparameter sweep**: Heatmaps or line plots
- **Statistical tests**: Box plots with significance markers

## Examples

- `/plot-results experiments/lr_sweep/` - Plot all runs in experiment
- `/plot-results logs/train_*.log` - Plot specific log files
- `/plot-results latest` - Plot most recent training run

## Notes

- Generates both PNG (for viewing) and PDF (for papers)
- Includes error bars for multiple seeds
- Follows matplotlib style guidelines for publications
- Can export to wandb for online viewing
