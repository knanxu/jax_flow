---
name: benchmark
description: Run comprehensive benchmarks across multiple tasks and compare with baselines. Use for paper experiments.
disable-model-invocation: false
allowed-tools: Bash, Read, Write, Edit, Grep, Glob
argument-hint: [task_list] [num_seeds]
---

# Run Benchmark Suite

Execute comprehensive benchmarks across multiple robotic tasks.

## Instructions

1. **Parse arguments**:
   - Task list: comma-separated (e.g., "lift,can,square") or "all"
   - Number of seeds: default 3

2. **Define benchmark tasks**:
   - If "all", use: lift, can, square, transport, tool_hang
   - Verify each task has dataset available

3. **Create benchmark plan**:
   - List all task × seed combinations
   - Estimate total training time
   - Calculate GPU requirements

4. **Ask user for confirmation**:
   - Show benchmark plan
   - Confirm resource availability
   - Get approval to proceed

5. **Set up benchmark directory**:
   ```
   benchmarks/YYYYMMDD_benchmark_name/
   ├── configs/
   ├── scripts/
   ├── results/
   └── README.md
   ```

6. **Generate training configs**:
   - One config per task × seed
   - Use consistent hyperparameters across tasks
   - Document any task-specific settings

7. **Create launch script**:
   - Parallel execution with GPU assignment
   - Automatic checkpointing
   - Error handling and restart logic

8. **Launch benchmark**:
   - Start all training runs
   - Monitor progress
   - Alert on failures

9. **Aggregate results**:
   - Collect success rates from all runs
   - Compute mean and std across seeds
   - Generate comparison table

10. **Create benchmark report**:
    - Summary table with all results
    - Statistical significance tests
    - Plots comparing tasks
    - LaTeX table for paper

## Benchmark Metrics

- **Success rate**: Primary metric (%)
- **Episode return**: Average reward
- **Training time**: Wall-clock time to convergence
- **Sample efficiency**: Steps to reach 80% success
- **Robustness**: Std across seeds

## Examples

- `/benchmark lift,can,square 5` - Benchmark 3 tasks with 5 seeds
- `/benchmark all 3` - Full benchmark suite
- `/benchmark lift 10` - Deep benchmark on single task

## Notes

- Use consistent hyperparameters for fair comparison
- Run at least 3 seeds for statistical validity
- Save all checkpoints for later analysis
- Document any task-specific modifications
