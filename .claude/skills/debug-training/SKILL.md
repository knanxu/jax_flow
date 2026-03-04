---
name: debug-training
description: Debug training issues like NaN losses, poor convergence, or crashes. Use when training fails or performs poorly.
disable-model-invocation: false
allowed-tools: Bash, Read, Grep, Glob
argument-hint: [log_file_or_issue_description]
---

# Debug Training Issues

Systematically diagnose and fix training problems in JAX flow models.

## Instructions

1. **Identify the issue**:
   - Parse $ARGUMENTS for log file path or issue description
   - If log file provided, read and analyze it
   - Common issues: NaN losses, slow convergence, OOM errors, shape mismatches

2. **Gather diagnostic information**:
   - Check recent training logs for error messages
   - Verify data loading (shapes, normalization, batch size)
   - Check model architecture (parameter count, layer sizes)
   - Review hyperparameters (lr, gradient clipping, optimizer)

3. **Run diagnostic checks**:
   ```bash
   # Check for NaN in dataset
   python -c "import h5py; f = h5py.File('data.hdf5'); print(f['data/actions'][:].max())"

   # Verify model can forward pass
   python -c "from jax_flow.agents import BCAgent; ..."

   # Check GPU memory
   nvidia-smi
   ```

4. **Identify root cause**:
   - **NaN losses**: Check learning rate, gradient clipping, data normalization
   - **Poor convergence**: Check batch size, network capacity, data quality
   - **OOM errors**: Reduce batch size, use gradient accumulation, check memory leaks
   - **Shape errors**: Verify observation/action dimensions match config

5. **Propose fixes**:
   - Provide specific config changes or code fixes
   - Explain why the fix should work
   - Suggest validation steps

6. **Apply fixes** if user approves

7. **Verify fix**:
   - Run short training test (100 steps)
   - Check that issue is resolved

## Common Fixes

- **NaN losses**: Add gradient clipping, reduce LR, check data normalization
- **Slow convergence**: Increase LR, increase batch size, check data quality
- **OOM**: Reduce batch size, use mixed precision, reduce model size
- **Shape mismatch**: Fix obs_steps/act_steps in config

## Examples

- `/debug-training logs/train.log` - Debug from log file
- `/debug-training "NaN loss after 1000 steps"` - Debug specific issue
- `/debug-training` - Interactive debugging session

## Notes

- Always check data normalization first (most common issue)
- Use gradient clipping for stability
- Start with small learning rates for flow matching
