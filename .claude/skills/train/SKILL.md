---
name: train
description: Train a JAX flow model with specified task and configuration. Use when user wants to start training.
disable-model-invocation: false
allowed-tools: Bash, Read, Grep, Glob
argument-hint: [task_name] [config_overrides]
---

# Train JAX Flow Model

Train a behavior cloning or RL model for robotic manipulation.

## Instructions

1. **Parse arguments**: Extract task name and config overrides from $ARGUMENTS
   - First argument: task name (e.g., "lift", "can", "square")
   - Remaining arguments: config overrides (e.g., "--config.lr=1e-4")

2. **Verify setup**:
   - Check if the task config exists in `configs/task/`
   - Verify dataset is available (check robomimic data path)
   - Confirm environment can be created

3. **Select training script**:
   - Use `examples/train_bc.py` for behavior cloning
   - Use `examples/train_acfql.py` for offline-to-online RL
   - Default to BC unless user specifies RL

4. **Run training**:
   ```bash
   python examples/train_bc.py --env_name=$TASK $CONFIG_OVERRIDES
   ```
   - Use `run_in_background: true` for long training runs
   - Monitor initial output for errors

5. **Report status**:
   - Confirm training started successfully
   - Show checkpoint directory path
   - Provide command to monitor progress (e.g., tensorboard)

## Examples

- `/train lift` - Train on lift task with default config
- `/train square --config.lr=1e-4 --config.batch_size=256` - Train with overrides
- `/train can acfql` - Train with ACFQL algorithm

## Notes

- Training runs in background by default for long experiments
- Checkpoints saved to `checkpoints/` directory
- Use `/eval` skill to evaluate trained models
