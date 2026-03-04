---
name: paper
description: Generate LaTeX tables, figures, and text snippets for research papers from experiment results.
disable-model-invocation: false
allowed-tools: Bash, Read, Write, Grep, Glob
argument-hint: [results_dir] [output_type]
---

# Generate Paper Materials

Create publication-ready LaTeX content from experiment results.

## Instructions

1. **Parse arguments**:
   - Results directory: path to experiment results
   - Output type: "table", "figure", "all" (default: all)

2. **Load experiment results**:
   - Read all result files (CSV, JSON, logs)
   - Parse success rates, returns, training times
   - Compute statistics (mean, std, confidence intervals)

3. **Generate LaTeX table**:
   ```latex
   \begin{table}[t]
   \centering
   \caption{Performance comparison on robotic manipulation tasks.}
   \label{tab:results}
   \begin{tabular}{lcccc}
   \toprule
   Method & Lift & Can & Square & Avg \\
   \midrule
   Flow Matching & 92.3 ± 2.1 & 87.5 ± 3.2 & ... \\
   ...
   \bottomrule
   \end{tabular}
   \end{table}
   ```

4. **Generate figure code**:
   - Create matplotlib script for publication-quality plots
   - Use proper fonts (Times, 10pt)
   - Include error bars and significance markers
   - Export as PDF with tight bounding box

5. **Generate results text**:
   - Draft results section paragraph
   - Include key numbers and comparisons
   - Highlight statistical significance
   - Suggest discussion points

6. **Create paper directory**:
   ```
   paper/
   ├── tables/
   │   └── results.tex
   ├── figures/
   │   ├── learning_curves.pdf
   │   └── success_rates.pdf
   ├── scripts/
   │   └── generate_plots.py
   └── snippets/
       └── results_text.tex
   ```

7. **Compile preview**:
   - If LaTeX installed, compile standalone document
   - Show preview of tables and figures

## Output Types

- **table**: LaTeX table with results
- **figure**: Publication-quality plots
- **text**: Results section draft
- **all**: Complete paper materials

## Examples

- `/paper experiments/benchmark/` - Generate all materials
- `/paper results/ table` - Only generate table
- `/paper latest figure` - Generate figures from latest run

## Notes

- Follows NeurIPS/ICML/CoRL style guidelines
- Includes statistical significance tests
- Generates both color and grayscale versions
- Provides LaTeX source for easy customization
