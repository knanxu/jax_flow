---
name: eval
description: Evaluate a trained policy checkpoint on robotic tasks. Use when user wants to test model performance.
disable-model-invocation: false
allowed-tools: Bash, Read, Grep, Glob
argument-hint: [checkpoint_path] [num_episodes]
---

# Evaluate Trained Policy

Evaluate a trained JAX flow policy on robotic manipulation tasks.

## Instructions

1. **Parse arguments**:
   - First argument: checkpoint path (or "latest" to find most recent)
   - Second argument: number of episodes (default: 50)

2. **Find checkpoint**:
   - If "latest", search `checkpoints/` for most recent `.pkl` file
   - Verify checkpoint file exists
   - Extract task name from checkpoint metadata if possible

3. **Verify environment**:
   - Check if environment can be created
   - Confirm task name matches checkpoint

4. **Run evaluation**:
   ```bash
   python examples/eval_policy.py \
     --checkpoint_path=$CHECKPOINT \
     --env_name=$TASK \
     --num_episodes=$NUM_EPISODES
   ```

5. **Parse and report results**:
   - Success rate
   - Average episode return
   - Episode length statistics
   - Any failure modes observed

6. **Suggest next steps**:
   - If performance is poor, suggest hyperparameter tuning
   - If good, suggest trying harder tasks or online fine-tuning

## Examples

- `/eval latest` - Evaluate most recent checkpoint
- `/eval checkpoints/lift_bc_step_100000.pkl 100` - Evaluate specific checkpoint
- `/eval latest 10` - Quick evaluation with 10 episodes

## Notes

- Evaluation runs with rendering disabled by default for speed
- Use `--render` flag if you want to visualize episodes
- Results are logged to console and optionally to wandb
