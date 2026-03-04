---
name: experiment
description: Set up and run a complete experiment with multiple hyperparameter configurations. Use for systematic research experiments.
disable-model-invocation: false
allowed-tools: Bash, Read, Write, Edit, Grep, Glob
argument-hint: [experiment_name]
---

# Run Research Experiment

Set up and execute a systematic experiment with multiple configurations.

## Instructions

1. **Parse experiment name**: Extract from $ARGUMENTS (e.g., "flow_ablation", "lr_sweep")

2. **Create experiment directory**:
   ```
   experiments/$EXPERIMENT_NAME/
   ├── configs/          # Config variants
   ├── scripts/          # Launch scripts
   ├── results/          # Checkpoints and logs
   └── README.md         # Experiment description
   ```

3. **Ask user for experiment details**:
   - Base task (lift, can, square, etc.)
   - Hyperparameters to sweep (lr, batch_size, flow_steps, etc.)
   - Number of seeds (default: 3)
   - Total training steps

4. **Generate config files**:
   - Create one config per hyperparameter combination
   - Use systematic naming: `{task}_{param1}_{param2}_seed{N}.py`
   - Inherit from base config and override specific parameters

5. **Create launch script**:
   - Bash script to run all configs in parallel or sequentially
   - Use `tmux` or `screen` for parallel runs
   - Log stdout/stderr to separate files

6. **Create experiment README**:
   - Document experiment purpose
   - List all configurations
   - Provide commands to monitor and analyze results

7. **Ask for confirmation** before launching

8. **Launch experiment** if approved

## Examples

- `/experiment lr_sweep` - Sweep learning rates
- `/experiment flow_ablation` - Compare flow matching variants

## Notes

- Use 3 seeds minimum for statistical significance
- Monitor GPU memory usage for parallel runs
- Results can be analyzed with `/plot-results` skill
