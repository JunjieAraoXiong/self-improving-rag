#!/usr/bin/env python3
"""Compare results between different experiments with significance testing.

This script loads results from multiple experiments and computes:
1. Pairwise significance tests (bootstrap)
2. Formatted comparison tables
3. LaTeX-ready output for papers

Usage:
    python scripts/compare_experiments.py \
        --baseline results/baseline_run.csv \
        --method results/scrag_run.csv \
        --output comparison.json

    # Or compare multiple aggregated JSON files
    python scripts/compare_experiments.py \
        --json-files results/*_aggregated_*.json \
        --output comparison_table.tex
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional

import pandas as pd
import numpy as np

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.metrics import bootstrap_compare, bootstrap_ci
from evaluation.latex_tables import significance_marker, generate_results_table


def load_results(path: Path) -> pd.DataFrame:
    """Load results from CSV file."""
    return pd.read_csv(path)


def load_aggregated_results(path: Path) -> Dict[str, Any]:
    """Load aggregated results from JSON file."""
    with open(path) as f:
        return json.load(f)


def compare_two_runs(
    baseline_df: pd.DataFrame,
    method_df: pd.DataFrame,
    metrics: List[str] = None,
) -> Dict[str, Dict[str, Any]]:
    """Compare two experiment runs with significance testing.

    Args:
        baseline_df: DataFrame with baseline results
        method_df: DataFrame with method results
        metrics: List of metric columns to compare

    Returns:
        Dict mapping metric names to comparison results
    """
    if metrics is None:
        metrics = ['semantic_similarity', 'judge_score', 'numeric_accuracy']
        metrics = [m for m in metrics if m in baseline_df.columns and m in method_df.columns]

    results = {}
    for metric in metrics:
        baseline_scores = baseline_df[metric].dropna().tolist()
        method_scores = method_df[metric].dropna().tolist()

        # Align by question_id if available
        if 'question_id' in baseline_df.columns and 'question_id' in method_df.columns:
            baseline_df_indexed = baseline_df.set_index('question_id')
            method_df_indexed = method_df.set_index('question_id')
            common_ids = set(baseline_df_indexed.index) & set(method_df_indexed.index)

            baseline_scores = baseline_df_indexed.loc[list(common_ids), metric].dropna().tolist()
            method_scores = method_df_indexed.loc[list(common_ids), metric].dropna().tolist()

        if len(baseline_scores) != len(method_scores):
            print(f"Warning: {metric} has different lengths ({len(baseline_scores)} vs {len(method_scores)})")
            min_len = min(len(baseline_scores), len(method_scores))
            baseline_scores = baseline_scores[:min_len]
            method_scores = method_scores[:min_len]

        if len(baseline_scores) > 0:
            comparison = bootstrap_compare(baseline_scores, method_scores)
            comparison['marker'] = significance_marker(comparison['p_value'])
            results[metric] = comparison

    return results


def format_comparison_summary(
    comparisons: Dict[str, Dict[str, Any]],
    baseline_name: str = "Baseline",
    method_name: str = "SC-RAG",
) -> str:
    """Format comparison results as a human-readable summary."""
    lines = []
    lines.append("=" * 70)
    lines.append(f"SIGNIFICANCE COMPARISON: {baseline_name} vs {method_name}")
    lines.append("=" * 70)

    for metric, result in comparisons.items():
        lines.append(f"\n{metric}:")
        lines.append(f"  {baseline_name}: {result['mean_a']:.4f}")
        lines.append(f"  {method_name}: {result['mean_b']:.4f}")
        lines.append(f"  Difference:  {result['mean_diff']:+.4f}")
        lines.append(f"  95% CI:      [{result['ci_95'][0]:+.4f}, {result['ci_95'][1]:+.4f}]")
        lines.append(f"  p-value:     {result['p_value']:.4f} {result['marker']}")
        lines.append(f"  Significant: {'Yes' if result['significant'] else 'No'}")

    lines.append("\n" + "=" * 70)
    lines.append("Legend: * p<0.05, ** p<0.01, *** p<0.001")
    lines.append("=" * 70)

    return "\n".join(lines)


def format_latex_table_from_comparisons(
    all_comparisons: Dict[str, Dict[str, Dict[str, Any]]],
    metrics: List[str],
    method_order: List[str],
    baseline_name: str,
) -> str:
    """Generate a LaTeX table from multiple comparisons.

    Args:
        all_comparisons: Dict[method_name -> Dict[metric -> comparison_result]]
        metrics: List of metric names to include
        method_order: Order of methods in table
        baseline_name: Name of baseline for significance markers

    Returns:
        LaTeX table string
    """
    metric_display = {
        'semantic_similarity': 'Sem.Sim',
        'judge_score': 'LLM Judge',
        'numeric_accuracy': 'Num.EM',
    }

    # Header
    header_cols = ["Configuration"] + [metric_display.get(m, m) for m in metrics]
    header = " & ".join(header_cols) + r" \\"

    rows = []
    for method in method_order:
        if method == baseline_name:
            # Baseline row - just show means
            cells = [method]
            for metric in metrics:
                if method in all_comparisons and metric in all_comparisons[method]:
                    mean = all_comparisons[method][metric]['mean_a']
                else:
                    mean = 0.0
                cells.append(f"{mean:.2f}")
            rows.append(" & ".join(cells) + r" \\")
        else:
            # Other methods - show mean with significance marker
            cells = [method]
            for metric in metrics:
                comp = all_comparisons.get(method, {}).get(metric, {})
                mean = comp.get('mean_b', 0)
                marker = comp.get('marker', '')

                cell = f"{mean:.2f}"
                if marker:
                    cell += f"$^{{{marker}}}$"
                cells.append(cell)
            rows.append(" & ".join(cells) + r" \\")

    # Assemble table
    col_spec = "l" + "c" * len(metrics)

    table_lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        rf"\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        header,
        r"\midrule",
    ]
    table_lines.extend(rows)
    table_lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{Comparison with statistical significance. $^*p<0.05$, $^{**}p<0.01$, $^{***}p<0.001$.}",
        r"\label{tab:comparison}",
        r"\end{table}",
    ])

    return "\n".join(table_lines)


def main():
    parser = argparse.ArgumentParser(description="Compare experiment results with significance testing")

    parser.add_argument(
        '--baseline', type=str, required=True,
        help='Path to baseline results CSV'
    )
    parser.add_argument(
        '--method', type=str, required=True,
        help='Path to method results CSV'
    )
    parser.add_argument(
        '--baseline-name', type=str, default='Baseline',
        help='Display name for baseline'
    )
    parser.add_argument(
        '--method-name', type=str, default='SC-RAG',
        help='Display name for method'
    )
    parser.add_argument(
        '--metrics', type=str, nargs='+',
        default=['semantic_similarity', 'judge_score', 'numeric_accuracy'],
        help='Metrics to compare'
    )
    parser.add_argument(
        '--output', type=str,
        help='Output file for comparison results (JSON)'
    )
    parser.add_argument(
        '--latex', type=str,
        help='Output file for LaTeX table'
    )

    args = parser.parse_args()

    # Load results
    print(f"Loading baseline: {args.baseline}")
    baseline_df = load_results(Path(args.baseline))

    print(f"Loading method: {args.method}")
    method_df = load_results(Path(args.method))

    print(f"\nBaseline: {len(baseline_df)} questions")
    print(f"Method:   {len(method_df)} questions")

    # Compare
    comparisons = compare_two_runs(
        baseline_df=baseline_df,
        method_df=method_df,
        metrics=args.metrics,
    )

    # Print summary
    print(format_comparison_summary(
        comparisons,
        baseline_name=args.baseline_name,
        method_name=args.method_name,
    ))

    # Save results
    if args.output:
        output = {
            'baseline': args.baseline,
            'method': args.method,
            'baseline_name': args.baseline_name,
            'method_name': args.method_name,
            'comparisons': comparisons,
        }
        with open(args.output, 'w') as f:
            json.dump(output, f, indent=2)
        print(f"\nResults saved to: {args.output}")

    if args.latex:
        # Generate simple LaTeX row
        row_parts = [args.method_name]
        for metric in args.metrics:
            if metric in comparisons:
                comp = comparisons[metric]
                mean = comp['mean_b']
                marker = comp['marker']
                if marker:
                    row_parts.append(f"{mean:.2f}$^{{{marker}}}$")
                else:
                    row_parts.append(f"{mean:.2f}")
            else:
                row_parts.append("--")

        latex_row = " & ".join(row_parts) + r" \\"

        with open(args.latex, 'w') as f:
            f.write(f"% Comparison: {args.method_name} vs {args.baseline_name}\n")
            f.write(latex_row + "\n")
        print(f"LaTeX row saved to: {args.latex}")


if __name__ == "__main__":
    main()
