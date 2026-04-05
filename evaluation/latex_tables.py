"""LaTeX table generation for ICLR-class papers.

Generates publication-ready tables with:
- Bootstrap confidence intervals
- Significance markers (*, **, ***)
- Proper formatting for academic papers
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from .metrics import bootstrap_compare


def significance_marker(p_value: float) -> str:
    """Return significance marker based on p-value.

    Convention:
        * : p < 0.05
        ** : p < 0.01
        *** : p < 0.001
    """
    if p_value < 0.001:
        return "***"
    elif p_value < 0.01:
        return "**"
    elif p_value < 0.05:
        return "*"
    return ""


def format_value_with_std(mean: float, std: float, decimals: int = 2) -> str:
    """Format a value with standard deviation for papers.

    Example: "0.72 ± 0.03"
    """
    if std is None or std == 0:
        return f"{mean:.{decimals}f}"
    return f"{mean:.{decimals}f} ± {std:.{decimals}f}"


def format_value_with_ci(mean: float, ci: Tuple[float, float], decimals: int = 2) -> str:
    """Format a value with 95% CI for papers.

    Example: "0.72 [0.68, 0.76]"
    """
    if ci is None:
        return f"{mean:.{decimals}f}"
    return f"{mean:.{decimals}f} [{ci[0]:.{decimals}f}, {ci[1]:.{decimals}f}]"


def generate_results_table(
    results: Dict[str, Dict[str, Any]],
    metrics: List[str] = None,
    baseline_key: str = None,
    caption: str = "Experimental Results",
    label: str = "tab:results",
    use_std: bool = True,
    bold_best: bool = True,
) -> str:
    """Generate a LaTeX table from experimental results.

    Args:
        results: Dict mapping method names to metric dicts.
                 Each metric dict should have 'mean', 'std', and optionally 'ci_95'.
        metrics: List of metric names to include (default: all found metrics)
        baseline_key: Key of the baseline method for significance testing
        caption: Table caption
        label: LaTeX label for referencing
        use_std: If True, show ± std; if False, show 95% CI
        bold_best: If True, bold the best value in each column

    Returns:
        LaTeX table string

    Example:
        results = {
            'Semantic': {'judge_score': {'mean': 0.52, 'std': 0.03}},
            'SC-RAG': {'judge_score': {'mean': 0.72, 'std': 0.02}},
        }
        latex = generate_results_table(results, baseline_key='Semantic')
    """
    if metrics is None:
        # Collect all metrics from first result
        first_result = next(iter(results.values()))
        metrics = [k for k in first_result.keys() if isinstance(first_result[k], dict)]

    # Nice metric names for display
    metric_display = {
        'semantic_similarity': 'Sem.Sim',
        'judge_score': 'LLM Judge',
        'numeric_accuracy': 'Num.EM',
        'lazarus_rate': 'Lazarus',
        'latency_ms': 'Latency',
    }

    # Build header
    metric_headers = [metric_display.get(m, m) for m in metrics]
    header = " & ".join(["Configuration"] + metric_headers) + r" \\"

    # Determine best values for each metric
    best_values = {}
    for metric in metrics:
        values = []
        for method, method_results in results.items():
            if metric in method_results and 'mean' in method_results[metric]:
                values.append(method_results[metric]['mean'])
        if values:
            best_values[metric] = max(values)  # Assuming higher is better

    # Build rows
    rows = []
    for method_name, method_results in results.items():
        cells = [method_name]

        for metric in metrics:
            if metric not in method_results:
                cells.append("--")
                continue

            m = method_results[metric]
            mean = m.get('mean', 0)
            std = m.get('std', 0)
            ci = m.get('ci_95')

            # Format value
            if use_std:
                cell = format_value_with_std(mean, std)
            else:
                cell = format_value_with_ci(mean, ci)

            # Add significance marker if baseline specified
            if baseline_key and method_name != baseline_key and baseline_key in results:
                baseline_m = results[baseline_key].get(metric, {})
                if 'scores' in m and 'scores' in baseline_m:
                    comp = bootstrap_compare(baseline_m['scores'], m['scores'])
                    marker = significance_marker(comp['p_value'])
                    if marker:
                        cell += f"$^{{{marker}}}$"

            # Bold if best
            if bold_best and abs(mean - best_values.get(metric, 0)) < 1e-6:
                cell = r"\textbf{" + cell + "}"

            cells.append(cell)

        rows.append(" & ".join(cells) + r" \\")

    # Assemble table
    n_cols = len(metrics) + 1
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
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\end{table}",
    ])

    return "\n".join(table_lines)


def generate_ablation_table(
    results: Dict[str, Dict[str, Any]],
    full_system_key: str = "Full SC-RAG",
    metrics: List[str] = None,
    caption: str = "Ablation Study Results",
    label: str = "tab:ablation",
) -> str:
    """Generate an ablation study table showing impact of removing components.

    Similar to generate_results_table but optimized for ablation presentation.
    Shows Δ (delta) from full system for each ablation.
    """
    if metrics is None:
        first_result = next(iter(results.values()))
        metrics = [k for k in first_result.keys() if isinstance(first_result[k], dict)]

    metric_display = {
        'semantic_similarity': 'Sem.Sim',
        'judge_score': 'LLM Judge',
        'numeric_accuracy': 'Num.EM',
    }

    # Header with Δ columns
    headers = ["Ablation"]
    for m in metrics:
        name = metric_display.get(m, m)
        headers.extend([name, f"Δ"])
    header = " & ".join(headers) + r" \\"

    # Get full system values
    full_system = results.get(full_system_key, {})

    rows = []
    for method_name, method_results in results.items():
        cells = [method_name.replace("−", "$-$")]  # LaTeX-safe minus

        for metric in metrics:
            m = method_results.get(metric, {})
            full_m = full_system.get(metric, {})

            mean = m.get('mean', 0)
            full_mean = full_m.get('mean', mean)
            delta = mean - full_mean

            # Format value
            cells.append(f"{mean:.2f}")

            # Format delta with sign and color
            if method_name == full_system_key:
                cells.append("--")
            elif delta < -0.01:
                cells.append(rf"\textcolor{{red}}{{{delta:+.2f}}}")
            elif delta > 0.01:
                cells.append(rf"\textcolor{{green!60!black}}{{{delta:+.2f}}}")
            else:
                cells.append(f"{delta:+.2f}")

        rows.append(" & ".join(cells) + r" \\")

    # Assemble table
    n_cols = 1 + len(metrics) * 2
    col_spec = "l" + "cc" * len(metrics)

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
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\end{table}",
    ])

    return "\n".join(table_lines)


def compute_significance_matrix(
    results: Dict[str, List[float]],
    method_names: List[str] = None,
) -> Dict[str, Dict[str, Dict]]:
    """Compute pairwise significance between all methods.

    Args:
        results: Dict mapping method names to lists of per-question scores
        method_names: Optional order of methods

    Returns:
        Nested dict: results[method_a][method_b] = comparison result
    """
    if method_names is None:
        method_names = list(results.keys())

    matrix = {}
    for method_a in method_names:
        matrix[method_a] = {}
        for method_b in method_names:
            if method_a == method_b:
                matrix[method_a][method_b] = None
            else:
                scores_a = results.get(method_a, [])
                scores_b = results.get(method_b, [])
                if len(scores_a) == len(scores_b) and len(scores_a) > 0:
                    matrix[method_a][method_b] = bootstrap_compare(scores_a, scores_b)
                else:
                    matrix[method_a][method_b] = None

    return matrix
