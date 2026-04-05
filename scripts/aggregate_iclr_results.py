#!/usr/bin/env python3
"""Aggregate ICLR experiment results and generate publication-ready tables.

This script:
1. Finds all experiment result files from bulk_runs/
2. Groups by pipeline/configuration
3. Computes cross-run statistics (mean ± std, 95% CI)
4. Runs significance tests vs baseline
5. Generates LaTeX tables for the paper

Usage:
    python scripts/aggregate_iclr_results.py --output-dir bulk_runs/

Example output structure:
    bulk_runs/
    ├── 2026-01-26_*.csv                    # Individual runs
    ├── aggregated_results.json             # Combined statistics
    ├── table1_main_results.tex             # Main results table
    └── table3_ablation.tex                 # Ablation table
"""

import argparse
import json
import sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Any, Tuple

import numpy as np
import pandas as pd

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.metrics import bootstrap_ci, bootstrap_compare
from evaluation.latex_tables import significance_marker, generate_results_table


def find_experiment_files(output_dir: Path) -> Dict[str, List[Path]]:
    """Find and group experiment result files by configuration.

    Returns:
        Dict mapping config key to list of result file paths
    """
    csv_files = sorted(output_dir.glob("*.csv"))

    # Group by configuration (everything except timestamp and seed/run)
    groups = defaultdict(list)

    for f in csv_files:
        name = f.stem

        # Skip combined/aggregated files
        if "combined" in name or "aggregated" in name:
            continue

        # Parse filename to extract config key
        # Format: timestamp_dataset_model_kN_tT_seedS_runR
        parts = name.split("_")

        # Find seed/run parts and remove them for grouping
        config_parts = []
        for p in parts:
            if p.startswith("seed") or p.startswith("run"):
                continue
            config_parts.append(p)

        config_key = "_".join(config_parts[1:])  # Skip timestamp
        groups[config_key].append(f)

    return dict(groups)


def load_and_aggregate(files: List[Path]) -> Dict[str, Any]:
    """Load multiple result files and compute aggregate statistics.

    Args:
        files: List of CSV result files for same configuration

    Returns:
        Dict with aggregated metrics
    """
    all_dfs = []
    for f in files:
        df = pd.read_csv(f)
        all_dfs.append(df)

    if not all_dfs:
        return {}

    # Metrics to aggregate
    metric_cols = ['semantic_similarity', 'judge_score', 'numeric_accuracy']

    result = {
        'num_runs': len(all_dfs),
        'files': [str(f) for f in files],
    }

    # Compute per-run means
    for col in metric_cols:
        run_means = []
        run_scores = []  # For paired significance tests

        for df in all_dfs:
            if col in df.columns:
                values = df[col].dropna()
                if len(values) > 0:
                    run_means.append(float(values.mean()))
                    run_scores.append(values.tolist())

        if run_means:
            mean = float(np.mean(run_means))
            std = float(np.std(run_means))

            # Bootstrap CI if enough runs
            ci = None
            if len(run_means) >= 2:
                _, ci_lower, ci_upper = bootstrap_ci(run_means, n_bootstrap=1000)
                ci = [ci_lower, ci_upper]

            result[col] = {
                'mean': mean,
                'std': std,
                'ci_95': ci,
                'per_run_means': run_means,
                'scores': run_scores[0] if run_scores else [],  # First run for paired tests
            }

    return result


def run_significance_tests(
    results: Dict[str, Dict],
    baseline_key: str,
    metric: str = 'judge_score'
) -> Dict[str, Dict]:
    """Run pairwise significance tests vs baseline.

    Args:
        results: Dict of config_key -> aggregated metrics
        baseline_key: Key of the baseline configuration
        metric: Metric to test on

    Returns:
        Dict mapping config_key to significance test results
    """
    if baseline_key not in results:
        print(f"Warning: Baseline '{baseline_key}' not found in results")
        return {}

    baseline = results[baseline_key]
    if metric not in baseline or 'scores' not in baseline[metric]:
        print(f"Warning: Metric '{metric}' or scores not in baseline")
        return {}

    baseline_scores = baseline[metric]['scores']

    comparisons = {}
    for config_key, config_results in results.items():
        if config_key == baseline_key:
            continue

        if metric not in config_results or 'scores' not in config_results[metric]:
            continue

        config_scores = config_results[metric]['scores']

        # Ensure same length for paired test
        min_len = min(len(baseline_scores), len(config_scores))
        if min_len == 0:
            continue

        comp = bootstrap_compare(
            baseline_scores[:min_len],
            config_scores[:min_len]
        )

        comparisons[config_key] = {
            'p_value': comp['p_value'],
            'significant': comp['significant'],
            'diff': comp['mean_diff'],
            'ci_95': comp['ci_95'],
            'marker': significance_marker(comp['p_value']),
        }

    return comparisons


def generate_main_table(
    results: Dict[str, Dict],
    significance: Dict[str, Dict],
    output_path: Path,
) -> str:
    """Generate LaTeX table for main results (Table 1).

    Args:
        results: Aggregated results by config
        significance: Significance test results
        output_path: Where to save the table

    Returns:
        LaTeX table string
    """
    # Define row order and display names
    row_order = [
        ('semantic', 'Semantic'),
        ('hybrid', 'Hybrid'),
        ('hybrid_filter', 'Hybrid+Filter'),
        ('hybrid_filter_rerank', 'Hybrid+F+Rerank'),
        ('scrag_b1', 'SC-RAG (B=1)'),
        ('scrag_b2', r'\textbf{SC-RAG (B=2)}'),
    ]

    metrics = ['semantic_similarity', 'judge_score', 'numeric_accuracy']
    metric_headers = ['Sem.Sim', 'LLM Judge', 'Num.EM']

    # Build table
    lines = [
        r'\begin{table}[t]',
        r'\centering',
        r'\small',
        r'\begin{tabular}{lccc}',
        r'\toprule',
        'Configuration & ' + ' & '.join(metric_headers) + r' \\',
        r'\midrule',
    ]

    for config_key, display_name in row_order:
        if config_key not in results:
            continue

        r = results[config_key]
        cells = [display_name]

        for metric in metrics:
            if metric not in r:
                cells.append('--')
                continue

            m = r[metric]
            mean = m['mean']
            std = m['std']

            # Format value
            cell = f'{mean:.2f} $\\pm$ {std:.2f}'

            # Add significance marker
            if config_key in significance:
                marker = significance[config_key].get('marker', '')
                if marker:
                    cell += f'$^{{{marker}}}$'

            # Bold best values
            all_means = [results[k][metric]['mean'] for k in results if metric in results[k]]
            if all_means and mean == max(all_means):
                cell = r'\textbf{' + cell + '}'

            cells.append(cell)

        lines.append(' & '.join(cells) + r' \\')

    lines.extend([
        r'\bottomrule',
        r'\end{tabular}',
        r'\caption{Main results on FinanceBench. Results averaged over 3 runs with different seeds. ',
        r'$\pm$ indicates standard deviation. ',
        r'$^*$ p<0.05, $^{**}$ p<0.01, $^{***}$ p<0.001 vs. best baseline (bootstrap test).}',
        r'\label{tab:main_results}',
        r'\end{table}',
    ])

    latex = '\n'.join(lines)

    # Save
    output_path.write_text(latex)
    print(f"Main results table saved to: {output_path}")

    return latex


def generate_ablation_table(
    results: Dict[str, Dict],
    output_path: Path,
) -> str:
    """Generate LaTeX table for ablation study (Table 3).

    Args:
        results: Aggregated results by config
        output_path: Where to save the table

    Returns:
        LaTeX table string
    """
    # Full system baseline
    full_key = 'scrag_b2'
    if full_key not in results:
        print("Warning: Full SC-RAG (B=2) not found for ablation baseline")
        return ""

    full = results[full_key]

    # Ablation order
    ablations = [
        ('scrag_b2', 'Full SC-RAG'),
        ('scrag_no_judge', r'$-$ Judge'),
        ('scrag_no_retrieval', r'$-$ Retrieval Escalation'),
        ('scrag_no_prompt', r'$-$ Prompt Escalation'),
        ('scrag_no_hyde', r'$-$ HyDE'),
        ('scrag_no_verify', r'$-$ Numeric Verify'),
    ]

    metrics = ['judge_score', 'numeric_accuracy']

    lines = [
        r'\begin{table}[t]',
        r'\centering',
        r'\small',
        r'\begin{tabular}{lcccc}',
        r'\toprule',
        r'Ablation & LLM Judge & $\Delta$ & Num.EM & $\Delta$ \\',
        r'\midrule',
    ]

    for config_key, display_name in ablations:
        if config_key not in results:
            continue

        r = results[config_key]
        cells = [display_name]

        for metric in metrics:
            if metric not in r or metric not in full:
                cells.extend(['--', '--'])
                continue

            mean = r[metric]['mean']
            full_mean = full[metric]['mean']
            delta = mean - full_mean

            cells.append(f'{mean:.2f}')

            if config_key == full_key:
                cells.append('--')
            elif delta < -0.01:
                cells.append(rf'\textcolor{{red}}{{{delta:+.2f}}}')
            elif delta > 0.01:
                cells.append(rf'\textcolor{{green!60!black}}{{{delta:+.2f}}}')
            else:
                cells.append(f'{delta:+.2f}')

        lines.append(' & '.join(cells) + r' \\')

    lines.extend([
        r'\bottomrule',
        r'\end{tabular}',
        r'\caption{Ablation study. Each row removes one component from the full SC-RAG system. ',
        r'$\Delta$ shows the change from full system.}',
        r'\label{tab:ablation}',
        r'\end{table}',
    ])

    latex = '\n'.join(lines)
    output_path.write_text(latex)
    print(f"Ablation table saved to: {output_path}")

    return latex


def main():
    parser = argparse.ArgumentParser(description="Aggregate ICLR experiment results")
    parser.add_argument(
        '--output-dir', type=str, default='bulk_runs',
        help='Directory containing experiment results'
    )
    parser.add_argument(
        '--baseline', type=str, default='hybrid_filter_rerank',
        help='Baseline configuration for significance tests'
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    if not output_dir.exists():
        print(f"Error: Output directory '{output_dir}' does not exist")
        sys.exit(1)

    print("=" * 60)
    print("AGGREGATING ICLR EXPERIMENT RESULTS")
    print("=" * 60)

    # Find experiment files
    print("\nFinding experiment files...")
    groups = find_experiment_files(output_dir)
    print(f"Found {len(groups)} unique configurations:")
    for key, files in groups.items():
        print(f"  {key}: {len(files)} runs")

    # Load and aggregate
    print("\nAggregating results...")
    results = {}
    for config_key, files in groups.items():
        results[config_key] = load_and_aggregate(files)

    # Run significance tests
    print(f"\nRunning significance tests vs baseline: {args.baseline}")
    significance = run_significance_tests(results, args.baseline)

    # Print summary
    print("\n" + "=" * 60)
    print("AGGREGATED RESULTS")
    print("=" * 60)

    for config_key, r in results.items():
        print(f"\n{config_key}:")
        for metric in ['semantic_similarity', 'judge_score', 'numeric_accuracy']:
            if metric in r:
                m = r[metric]
                print(f"  {metric}: {m['mean']:.4f} ± {m['std']:.4f}")
                if config_key in significance:
                    sig = significance[config_key]
                    print(f"    p-value: {sig['p_value']:.4f} {sig['marker']}")

    # Save aggregated JSON
    json_path = output_dir / 'aggregated_results.json'
    with open(json_path, 'w') as f:
        json.dump({
            'results': results,
            'significance': significance,
        }, f, indent=2, default=str)
    print(f"\nAggregated results saved to: {json_path}")

    # Generate tables
    print("\nGenerating LaTeX tables...")
    generate_main_table(results, significance, output_dir / 'table1_main_results.tex')
    generate_ablation_table(results, output_dir / 'table3_ablation.tex')

    print("\n" + "=" * 60)
    print("AGGREGATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
