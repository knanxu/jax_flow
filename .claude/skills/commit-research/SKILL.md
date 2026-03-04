---
name: commit-research
description: Create a well-formatted research commit with experiment results and analysis. Use after completing experiments.
disable-model-invocation: true
allowed-tools: Bash, Read, Grep, Glob
---

# Commit Research Progress

Create a structured commit for research experiments with proper documentation.

## Instructions

1. **Review changes**:
   - Run `git status` to see modified files
   - Run `git diff` to review code changes
   - Check for new experiment results, plots, or configs

2. **Categorize changes**:
   - **Code changes**: New features, bug fixes, refactoring
   - **Experiments**: New configs, results, plots
   - **Documentation**: README updates, paper drafts

3. **Create commit message** following this format:
   ```
   [Category] Brief description (< 50 chars)

   ## Changes
   - Bullet point list of key changes
   - Focus on what and why, not how

   ## Results (if applicable)
   - Success rate: X%
   - Key findings: ...

   ## Next steps (if applicable)
   - What to try next

   Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
   ```

4. **Categories**:
   - `[Experiment]` - New experimental results
   - `[Feature]` - New functionality
   - `[Fix]` - Bug fixes
   - `[Refactor]` - Code improvements
   - `[Docs]` - Documentation updates
   - `[Config]` - Configuration changes

5. **Stage and commit**:
   - Stage relevant files (avoid staging unrelated changes)
   - Create commit with formatted message
   - Verify commit with `git log -1`

6. **Suggest next actions**:
   - Push to remote if appropriate
   - Create PR if on feature branch
   - Tag commit if milestone reached

## Examples

Good commit messages:
```
[Experiment] Evaluate flow matching on lift task

## Changes
- Add lift task configuration with lowdim observations
- Train BC agent for 100k steps with lr=3e-4
- Evaluate on 50 episodes

## Results
- Success rate: 92% (46/50 episodes)
- Average return: 1.85
- Converged after ~60k steps

## Next steps
- Try image observations
- Compare with MeanFlow variant
```

```
[Fix] Resolve NaN loss in flow matching training

## Changes
- Add gradient clipping (max_norm=1.0)
- Fix action normalization bounds
- Reduce initial learning rate to 1e-4

## Results
- Training now stable for 100k+ steps
- No NaN losses observed
```

## Notes

- Keep commits focused (one logical change per commit)
- Include quantitative results when available
- Reference issue numbers if applicable
- Don't commit large binary files (checkpoints, datasets)
