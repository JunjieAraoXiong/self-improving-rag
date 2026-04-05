"""Evaluation metrics for RAG system."""

from typing import Dict, Any, Optional, List, Tuple
import numpy as np
import pandas as pd


def bootstrap_ci(
    scores: List[float],
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    seed: Optional[int] = 42
) -> Tuple[float, float, float]:
    """Compute bootstrap confidence interval for the mean.

    Bootstrap resampling provides robust confidence intervals without assuming
    normal distribution - important for small sample sizes typical in RAG evaluation.

    Args:
        scores: List of metric scores to compute CI for
        n_bootstrap: Number of bootstrap resamples (default: 1000)
        ci: Confidence level (default: 0.95 for 95% CI)
        seed: Random seed for reproducibility (default: 42)

    Returns:
        Tuple of (mean, lower_bound, upper_bound)

    Example:
        >>> scores = [0.6, 0.7, 0.5, 0.8, 0.65]
        >>> mean, lower, upper = bootstrap_ci(scores)
        >>> print(f"Mean: {mean:.3f} [{lower:.3f}, {upper:.3f}]")
    """
    if not scores:
        return 0.0, 0.0, 0.0

    scores_arr = np.array(scores)
    n = len(scores_arr)

    # Set random seed for reproducibility
    rng = np.random.RandomState(seed)

    # Generate bootstrap samples and compute means
    boot_means = []
    for _ in range(n_bootstrap):
        resample_idx = rng.randint(0, n, size=n)
        resample = scores_arr[resample_idx]
        boot_means.append(np.mean(resample))

    boot_means = np.array(boot_means)

    # Compute percentile-based confidence interval
    alpha = (1 - ci) / 2
    lower = float(np.percentile(boot_means, alpha * 100))
    upper = float(np.percentile(boot_means, (1 - alpha) * 100))
    mean = float(np.mean(scores_arr))

    return mean, lower, upper


def bootstrap_compare(
    scores_a: List[float],
    scores_b: List[float],
    n_bootstrap: int = 1000,
    seed: int = 42
) -> Dict[str, Any]:
    """Compare two methods using paired bootstrap test.

    This performs a paired bootstrap test to determine if there's a statistically
    significant difference between two methods evaluated on the same questions.
    The pairing removes inter-question variance, making the test more powerful.

    Args:
        scores_a: Scores from method A (e.g., baseline)
        scores_b: Scores from method B (e.g., new method)
        n_bootstrap: Number of bootstrap resamples (default: 1000)
        seed: Random seed for reproducibility (default: 42)

    Returns:
        Dict with:
            - mean_a: Mean score of method A
            - mean_b: Mean score of method B
            - mean_diff: Mean difference (B - A), positive means B is better
            - ci_95: 95% confidence interval for the difference
            - p_value: Two-tailed p-value for the null hypothesis (no difference)
            - significant: Boolean indicating significance at α=0.05

    Example:
        >>> baseline = [0.5, 0.6, 0.4, 0.7, 0.55]
        >>> new_method = [0.6, 0.7, 0.5, 0.8, 0.65]
        >>> result = bootstrap_compare(baseline, new_method)
        >>> print(f"Diff: {result['mean_diff']:.3f}, p={result['p_value']:.3f}")
    """
    if len(scores_a) != len(scores_b):
        raise ValueError(
            f"Scores must be paired (same length): got {len(scores_a)} vs {len(scores_b)}"
        )

    if len(scores_a) == 0:
        return {
            'mean_a': 0.0, 'mean_b': 0.0, 'mean_diff': 0.0,
            'ci_95': [0.0, 0.0], 'p_value': 1.0, 'significant': False
        }

    scores_a = np.array(scores_a)
    scores_b = np.array(scores_b)
    n = len(scores_a)

    # Observed difference
    observed_diff = float(np.mean(scores_b) - np.mean(scores_a))

    # Set random seed for reproducibility
    rng = np.random.RandomState(seed)

    # Paired bootstrap: resample indices, compute difference of means
    boot_diffs = []
    for _ in range(n_bootstrap):
        resample_idx = rng.randint(0, n, size=n)
        resample_a = scores_a[resample_idx]
        resample_b = scores_b[resample_idx]
        boot_diff = np.mean(resample_b) - np.mean(resample_a)
        boot_diffs.append(boot_diff)

    boot_diffs = np.array(boot_diffs)

    # Confidence interval for the difference
    ci_lower = float(np.percentile(boot_diffs, 2.5))
    ci_upper = float(np.percentile(boot_diffs, 97.5))

    # Two-tailed p-value: fraction of bootstrap diffs with opposite sign to observed
    # or more extreme in magnitude
    if observed_diff >= 0:
        p_value = float(np.mean(boot_diffs <= 0)) * 2
    else:
        p_value = float(np.mean(boot_diffs >= 0)) * 2

    # Cap p-value at 1.0
    p_value = min(p_value, 1.0)

    return {
        'mean_a': float(np.mean(scores_a)),
        'mean_b': float(np.mean(scores_b)),
        'mean_diff': observed_diff,
        'ci_95': [ci_lower, ci_upper],
        'p_value': p_value,
        'significant': p_value < 0.05
    }


def format_comparison_table(
    comparisons: Dict[str, Dict[str, Any]],
    method_a_name: str = "Baseline",
    method_b_name: str = "New Method"
) -> str:
    """Format method comparison results into a readable table.

    Args:
        comparisons: Dict mapping metric names to bootstrap_compare results
        method_a_name: Display name for method A
        method_b_name: Display name for method B

    Returns:
        Formatted string table

    Example:
        >>> comps = {'accuracy': bootstrap_compare(baseline, new_method)}
        >>> print(format_comparison_table(comps, "GPT-3.5", "GPT-4"))
    """
    lines = []
    lines.append("=" * 70)
    lines.append(f"METHOD COMPARISON: {method_a_name} vs {method_b_name}")
    lines.append("=" * 70)
    lines.append(f"{'Metric':<20} {method_a_name:<10} {method_b_name:<10} {'Diff':<12} {'p-value':<10} {'Sig?'}")
    lines.append("-" * 70)

    for metric_name, result in comparisons.items():
        sig_marker = "✓" if result['significant'] else ""
        diff_str = f"{result['mean_diff']:+.4f}"
        ci = result['ci_95']
        diff_with_ci = f"{diff_str} [{ci[0]:+.3f}, {ci[1]:+.3f}]"

        lines.append(
            f"{metric_name:<20} {result['mean_a']:.4f}     {result['mean_b']:.4f}     "
            f"{diff_with_ci:<24} {result['p_value']:.4f}     {sig_marker}"
        )

    lines.append("=" * 70)
    lines.append("Note: Positive diff means method B is better. Sig? = p < 0.05")

    return "\n".join(lines)


def pass_at_k(
    scores: List[float],
    k: int = 1,
    threshold: float = 0.5
) -> float:
    """Compute Pass@k - fraction of questions where score meets threshold.

    For k=1, this is equivalent to accuracy at the given threshold.
    This converts continuous judge scores into a binary pass/fail metric.

    Args:
        scores: List of continuous scores (0-1)
        k: Number of attempts (currently only k=1 supported)
        threshold: Score threshold for "pass" (default: 0.5)

    Returns:
        Pass rate as a float (0-1)

    Example:
        >>> scores = [0.6, 0.3, 0.7, 0.4, 0.8]
        >>> pass_at_k(scores, threshold=0.5)
        0.6  # 3 out of 5 pass
    """
    if not scores:
        return 0.0

    # For k=1: simple threshold check
    passes = sum(1 for s in scores if s >= threshold)
    return passes / len(scores)


def categorize_failure(row: Dict[str, Any]) -> str:
    """Categorize why a question failed to get a good answer.

    Categories:
    - 'ok': Answer is acceptable (semantic_similarity >= 0.5)
    - 'error': Processing error occurred
    - 'retrieval_empty': No documents retrieved
    - 'numeric_hallucination': Answer contains hallucinated numbers
    - 'generation_poor': Retrieved docs but generated poor answer

    Args:
        row: Dictionary or Series with result fields

    Returns:
        Category string
    """
    # Check for errors first
    if row.get('error'):
        return 'error'

    # Check for empty retrieval
    if not row.get('sources'):
        return 'retrieval_empty'

    # Check for numeric hallucination
    numeric_score = row.get('numeric_score')
    if numeric_score is not None and numeric_score < 0.5:
        return 'numeric_hallucination'

    # Check semantic similarity
    sem_sim = row.get('semantic_similarity', 0)
    if sem_sim < 0.5:
        return 'generation_poor'

    return 'ok'


def calculate_failure_breakdown(results_df: pd.DataFrame) -> Dict[str, Any]:
    """Calculate breakdown of failure categories.

    Args:
        results_df: DataFrame with evaluation results

    Returns:
        Dictionary with failure category counts and percentages
    """
    categories = results_df.apply(
        lambda row: categorize_failure(row.to_dict()),
        axis=1
    )

    counts = categories.value_counts().to_dict()
    total = len(results_df)

    breakdown = {
        'counts': counts,
        'percentages': {k: v / total for k, v in counts.items()},
        'total': total,
    }

    return breakdown


def embedding_similarity(
    predicted: str,
    gold: str,
    embeddings,
) -> float:
    """Calculate cosine similarity between predicted and gold answers using embeddings.

    Args:
        predicted: The predicted answer text
        gold: The gold/reference answer text
        embeddings: Embedding model instance (HuggingFaceEmbeddings or similar)

    Returns:
        Cosine similarity score between 0 and 1
    """
    if not predicted or not gold:
        return 0.0

    try:
        # Get embeddings for both texts
        pred_embedding = embeddings.embed_query(predicted)
        gold_embedding = embeddings.embed_query(gold)

        # Convert to numpy arrays
        pred_vec = np.array(pred_embedding)
        gold_vec = np.array(gold_embedding)

        # Calculate cosine similarity
        dot_product = np.dot(pred_vec, gold_vec)
        pred_norm = np.linalg.norm(pred_vec)
        gold_norm = np.linalg.norm(gold_vec)

        if pred_norm == 0 or gold_norm == 0:
            return 0.0

        similarity = dot_product / (pred_norm * gold_norm)

        # Clamp to [0, 1] range (cosine similarity can be negative)
        return float(max(0.0, min(1.0, similarity)))

    except Exception as e:
        print(f"Error calculating embedding similarity: {e}")
        return 0.0


def calculate_aggregate_metrics(results_df: pd.DataFrame) -> Dict[str, Any]:
    """Calculate aggregate metrics from evaluation results.

    Args:
        results_df: DataFrame with evaluation results including 'semantic_similarity',
                   optionally 'judge_score' and 'question_type' columns

    Returns:
        Dictionary with aggregate metrics
    """
    metrics = {}

    # Overall semantic similarity
    if 'semantic_similarity' in results_df.columns:
        sim_values = results_df['semantic_similarity'].dropna()
        if len(sim_values) > 0:
            sim_list = sim_values.tolist()
            mean, ci_lower, ci_upper = bootstrap_ci(sim_list)
            metrics['semantic_similarity'] = {
                'mean': mean,
                'ci_95': [ci_lower, ci_upper],
                'std': float(sim_values.std()),
                'min': float(sim_values.min()),
                'max': float(sim_values.max()),
                'count': int(len(sim_values)),
            }
        else:
            metrics['semantic_similarity'] = {
                'mean': 0.0, 'ci_95': [0.0, 0.0], 'std': 0.0,
                'min': 0.0, 'max': 0.0, 'count': 0
            }

    # LLM judge scores if available
    if 'judge_score' in results_df.columns:
        judge_values = results_df['judge_score'].dropna()
        if len(judge_values) > 0:
            judge_list = judge_values.tolist()
            mean, ci_lower, ci_upper = bootstrap_ci(judge_list)
            accuracy = float((judge_values >= 0.5).mean())
            pass_rate = pass_at_k(judge_list, threshold=0.5)
            metrics['judge_score'] = {
                'mean': mean,
                'ci_95': [ci_lower, ci_upper],
                'std': float(judge_values.std()),
                'accuracy': accuracy,
                'pass_at_1': pass_rate,
                'count': int(len(judge_values)),
            }
        else:
            metrics['judge_score'] = {
                'mean': 0.0, 'ci_95': [0.0, 0.0], 'std': 0.0,
                'accuracy': 0.0, 'pass_at_1': 0.0, 'count': 0
            }

    # Per question type breakdown with bootstrap CIs
    # Minimum samples required for meaningful CI (too few samples = unreliable CI)
    MIN_SAMPLES_FOR_CI = 5

    if 'question_type' in results_df.columns:
        metrics['by_question_type'] = {}
        for q_type in results_df['question_type'].unique():
            if pd.isna(q_type):
                continue
            type_df = results_df[results_df['question_type'] == q_type]
            type_metrics = {}

            if 'semantic_similarity' in type_df.columns:
                sim_vals = type_df['semantic_similarity'].dropna()
                n_samples = len(sim_vals)
                if n_samples >= MIN_SAMPLES_FOR_CI:
                    mean, ci_lower, ci_upper = bootstrap_ci(sim_vals.tolist())
                    type_metrics['semantic_similarity'] = {
                        'mean': mean,
                        'ci_95': [ci_lower, ci_upper],
                        'count': n_samples
                    }
                elif n_samples > 0:
                    # Not enough samples for reliable CI, just report mean
                    type_metrics['semantic_similarity'] = {
                        'mean': float(sim_vals.mean()),
                        'ci_95': None,  # Indicates insufficient samples
                        'count': n_samples
                    }
                else:
                    type_metrics['semantic_similarity'] = {
                        'mean': 0.0,
                        'ci_95': None,
                        'count': 0
                    }

            if 'judge_score' in type_df.columns:
                judge_vals = type_df['judge_score'].dropna()
                n_samples = len(judge_vals)
                accuracy = float((judge_vals >= 0.5).mean()) if n_samples > 0 else 0.0

                if n_samples >= MIN_SAMPLES_FOR_CI:
                    mean, ci_lower, ci_upper = bootstrap_ci(judge_vals.tolist())
                    # Also compute CI for accuracy (binary pass/fail)
                    binary_scores = [1.0 if s >= 0.5 else 0.0 for s in judge_vals]
                    acc_mean, acc_lower, acc_upper = bootstrap_ci(binary_scores)
                    type_metrics['judge_score'] = {
                        'mean': mean,
                        'ci_95': [ci_lower, ci_upper],
                        'accuracy': acc_mean,
                        'accuracy_ci_95': [acc_lower, acc_upper],
                        'count': n_samples
                    }
                elif n_samples > 0:
                    type_metrics['judge_score'] = {
                        'mean': float(judge_vals.mean()),
                        'ci_95': None,
                        'accuracy': accuracy,
                        'accuracy_ci_95': None,
                        'count': n_samples
                    }
                else:
                    type_metrics['judge_score'] = {
                        'mean': 0.0,
                        'ci_95': None,
                        'accuracy': 0.0,
                        'accuracy_ci_95': None,
                        'count': 0
                    }

            metrics['by_question_type'][str(q_type)] = type_metrics

    # Timing metrics
    if 'retrieval_time_ms' in results_df.columns:
        retrieval_times = results_df['retrieval_time_ms'].dropna()
        metrics['retrieval_time_ms'] = {
            'mean': float(retrieval_times.mean()) if len(retrieval_times) > 0 else 0.0,
            'p50': float(retrieval_times.median()) if len(retrieval_times) > 0 else 0.0,
            'p95': float(retrieval_times.quantile(0.95)) if len(retrieval_times) > 0 else 0.0,
        }

    if 'generation_time_ms' in results_df.columns:
        gen_times = results_df['generation_time_ms'].dropna()
        metrics['generation_time_ms'] = {
            'mean': float(gen_times.mean()) if len(gen_times) > 0 else 0.0,
            'p50': float(gen_times.median()) if len(gen_times) > 0 else 0.0,
            'p95': float(gen_times.quantile(0.95)) if len(gen_times) > 0 else 0.0,
        }

    # Error rate
    if 'error' in results_df.columns:
        error_count = results_df['error'].notna().sum()
        metrics['error_rate'] = float(error_count / len(results_df)) if len(results_df) > 0 else 0.0

    # Numeric verification metrics (hallucination check against sources)
    if 'numeric_score' in results_df.columns:
        numeric_values = results_df['numeric_score'].dropna()
        if len(numeric_values) > 0:
            metrics['numeric_verification'] = {
                'mean': float(numeric_values.mean()),
                'hallucination_rate': float((numeric_values < 1.0).mean()),
                'perfect_rate': float((numeric_values == 1.0).mean()),
                'count': int(len(numeric_values)),
            }

    # Numeric accuracy metrics (exact-match against gold answer)
    if 'numeric_accuracy' in results_df.columns:
        # Filter out None values (non-numeric questions)
        numeric_acc_values = results_df['numeric_accuracy'].dropna()
        if len(numeric_acc_values) > 0:
            acc_list = numeric_acc_values.tolist()
            mean, ci_lower, ci_upper = bootstrap_ci(acc_list)
            metrics['numeric_accuracy'] = {
                'mean': mean,
                'ci_95': [ci_lower, ci_upper],
                'exact_match_rate': float(numeric_acc_values.mean()),
                'count': int(len(numeric_acc_values)),
                'total_questions': len(results_df),
                'numeric_questions': int(len(numeric_acc_values)),
            }

    # Failure breakdown - categorize WHY questions failed
    metrics['failure_breakdown'] = calculate_failure_breakdown(results_df)

    return metrics


def format_metrics_summary(metrics: Dict[str, Any]) -> str:
    """Format metrics dictionary into a human-readable summary.

    Args:
        metrics: Dictionary of metrics from calculate_aggregate_metrics

    Returns:
        Formatted string summary
    """
    lines = []
    lines.append("=" * 60)
    lines.append("EVALUATION METRICS SUMMARY")
    lines.append("=" * 60)

    # Overall semantic similarity
    if 'semantic_similarity' in metrics:
        sim = metrics['semantic_similarity']
        lines.append(f"\nSemantic Similarity:")
        lines.append(f"  Mean:  {sim['mean']:.4f}")
        if 'ci_95' in sim:
            ci = sim['ci_95']
            lines.append(f"  95% CI: [{ci[0]:.4f}, {ci[1]:.4f}]")
        lines.append(f"  Std:   {sim['std']:.4f}")
        lines.append(f"  Range: [{sim['min']:.4f}, {sim['max']:.4f}]")
        lines.append(f"  Count: {sim['count']}")

    # LLM Judge scores
    if 'judge_score' in metrics:
        judge = metrics['judge_score']
        lines.append(f"\nLLM Judge:")
        lines.append(f"  Mean Score: {judge['mean']:.4f}")
        if 'ci_95' in judge:
            ci = judge['ci_95']
            lines.append(f"  95% CI:     [{ci[0]:.4f}, {ci[1]:.4f}]")
        lines.append(f"  Accuracy:   {judge['accuracy']:.2%}")
        if 'pass_at_1' in judge:
            lines.append(f"  Pass@1:     {judge['pass_at_1']:.2%}")
        lines.append(f"  Count:      {judge['count']}")

    # Per question type (with CIs when available)
    if 'by_question_type' in metrics and metrics['by_question_type']:
        lines.append(f"\nBy Question Type:")
        for q_type, type_metrics in metrics['by_question_type'].items():
            lines.append(f"  {q_type}:")

            # Handle new nested structure with CIs
            if 'semantic_similarity' in type_metrics:
                sim = type_metrics['semantic_similarity']
                if isinstance(sim, dict):
                    mean_str = f"{sim['mean']:.4f}"
                    if sim.get('ci_95'):
                        ci = sim['ci_95']
                        lines.append(f"    Semantic Sim: {mean_str} [{ci[0]:.3f}, {ci[1]:.3f}]")
                    else:
                        lines.append(f"    Semantic Sim: {mean_str} (n<5, no CI)")
                    lines.append(f"    Count:        {sim['count']}")
                else:
                    # Legacy format fallback
                    lines.append(f"    Semantic Sim: {sim:.4f}")

            # Legacy format fallback for semantic_similarity_mean
            elif 'semantic_similarity_mean' in type_metrics:
                lines.append(f"    Semantic Sim: {type_metrics['semantic_similarity_mean']:.4f}")
                if 'count' in type_metrics:
                    lines.append(f"    Count:        {type_metrics['count']}")

            if 'judge_score' in type_metrics:
                judge = type_metrics['judge_score']
                if isinstance(judge, dict):
                    mean_str = f"{judge['mean']:.4f}"
                    if judge.get('ci_95'):
                        ci = judge['ci_95']
                        lines.append(f"    Judge Score:  {mean_str} [{ci[0]:.3f}, {ci[1]:.3f}]")
                    else:
                        lines.append(f"    Judge Score:  {mean_str} (n<5, no CI)")
                    if judge.get('accuracy') is not None:
                        acc_str = f"{judge['accuracy']:.2%}"
                        if judge.get('accuracy_ci_95'):
                            acc_ci = judge['accuracy_ci_95']
                            lines.append(f"    Accuracy:     {acc_str} [{acc_ci[0]:.1%}, {acc_ci[1]:.1%}]")
                        else:
                            lines.append(f"    Accuracy:     {acc_str}")
                else:
                    # Legacy format fallback
                    lines.append(f"    Judge Score:  {judge:.4f}")

            # Legacy format fallback for judge_score_mean
            elif 'judge_score_mean' in type_metrics:
                lines.append(f"    Judge Score:  {type_metrics['judge_score_mean']:.4f}")
                if 'judge_accuracy' in type_metrics:
                    lines.append(f"    Accuracy:     {type_metrics['judge_accuracy']:.2%}")

    # Timing
    if 'retrieval_time_ms' in metrics:
        ret = metrics['retrieval_time_ms']
        lines.append(f"\nRetrieval Latency (ms):")
        lines.append(f"  Mean: {ret['mean']:.1f}  P50: {ret['p50']:.1f}  P95: {ret['p95']:.1f}")

    if 'generation_time_ms' in metrics:
        gen = metrics['generation_time_ms']
        lines.append(f"\nGeneration Latency (ms):")
        lines.append(f"  Mean: {gen['mean']:.1f}  P50: {gen['p50']:.1f}  P95: {gen['p95']:.1f}")

    # Error rate
    if 'error_rate' in metrics:
        lines.append(f"\nError Rate: {metrics['error_rate']:.2%}")

    # Numeric verification (hallucination check)
    if 'numeric_verification' in metrics:
        num = metrics['numeric_verification']
        lines.append(f"\nNumeric Verification (vs Sources):")
        lines.append(f"  Mean Score:        {num['mean']:.4f}")
        lines.append(f"  Hallucination Rate: {num['hallucination_rate']:.2%}")
        lines.append(f"  Perfect Rate:      {num['perfect_rate']:.2%}")
        lines.append(f"  Count:             {num['count']}")

    # Numeric accuracy (exact-match against gold)
    if 'numeric_accuracy' in metrics:
        num_acc = metrics['numeric_accuracy']
        lines.append(f"\nNumeric Accuracy (vs Gold Answer):")
        lines.append(f"  Exact Match Rate: {num_acc['exact_match_rate']:.2%}")
        if 'ci_95' in num_acc:
            ci = num_acc['ci_95']
            lines.append(f"  95% CI:           [{ci[0]:.4f}, {ci[1]:.4f}]")
        lines.append(f"  Numeric Questions: {num_acc['numeric_questions']}/{num_acc['total_questions']}")

    # Failure breakdown
    if 'failure_breakdown' in metrics:
        fb = metrics['failure_breakdown']
        lines.append(f"\nFailure Breakdown:")
        for category, pct in sorted(fb['percentages'].items(), key=lambda x: -x[1]):
            count = fb['counts'].get(category, 0)
            lines.append(f"  {category:25} {pct:6.1%} ({count})")

    lines.append("\n" + "=" * 60)

    return "\n".join(lines)
