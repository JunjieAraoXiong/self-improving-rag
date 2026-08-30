"""Evaluation metrics for RAG system."""

import json
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
import pandas as pd


def _coerce_binary_series(values: pd.Series, field_name: str) -> pd.Series:
    """Parse nullable booleans without treating the string ``False`` as true."""

    parsed = {}
    for index, raw in values.dropna().items():
        if isinstance(raw, (bool, np.bool_)):
            parsed[index] = float(bool(raw))
            continue
        if isinstance(raw, (int, float, np.integer, np.floating)) and raw in (0, 1):
            parsed[index] = float(raw)
            continue
        if isinstance(raw, str):
            normalized = raw.strip().lower()
            if normalized in {"true", "t", "yes", "y", "1"}:
                parsed[index] = 1.0
                continue
            if normalized in {"false", "f", "no", "n", "0"}:
                parsed[index] = 0.0
                continue
        raise ValueError(f"Unrecognized boolean value in {field_name}: {raw!r}")
    return pd.Series(parsed, dtype=float)


def _error_mask(values: pd.Series) -> pd.Series:
    """Return a boolean mask for non-empty terminal error values."""

    def is_error(raw: Any) -> bool:
        if raw is None:
            return False
        try:
            if pd.isna(raw):
                return False
        except (TypeError, ValueError):
            pass
        if isinstance(raw, str):
            return bool(raw.strip())
        return bool(raw)

    return values.map(is_error).astype(bool)


def _judge_correctness_thresholds(
    results_df: pd.DataFrame,
    indices: pd.Index,
) -> pd.Series:
    """Return the correctness threshold associated with each Judge score.

    New artifacts persist ``outcome_judge_threshold`` explicitly. For legacy
    rows, post-selection evaluation defaults to the current full-credit 0.99
    rule while historical oracle/policy scores retain their old 0.5 boundary.
    """

    thresholds = pd.Series(np.nan, index=indices, dtype=float)
    if 'outcome_judge_threshold' in results_df.columns:
        persisted = pd.to_numeric(
            results_df.loc[indices, 'outcome_judge_threshold'],
            errors='coerce',
        )
        invalid = persisted.dropna()[(persisted.dropna() < 0.0) | (persisted.dropna() > 1.0)]
        if len(invalid):
            raise ValueError("outcome_judge_threshold must be between 0 and 1")
        thresholds.loc[persisted.index] = persisted

    for index in indices:
        if not pd.isna(thresholds.loc[index]):
            continue
        mode = (
            str(results_df.at[index, 'evaluation_mode'])
            if 'evaluation_mode' in results_df.columns
            and not pd.isna(results_df.at[index, 'evaluation_mode'])
            else ''
        )
        thresholds.loc[index] = (
            0.99
            if mode.startswith(('post_selection_', 'terminal_', 'evaluator_'))
            else 0.5
        )
    return thresholds


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


def _has_retrieved_evidence(row: Dict[str, Any]) -> bool:
    """Recognize retrieval evidence across standard and agentic result schemas."""

    count = row.get("retrieved_document_count")
    try:
        if count is not None and float(count) > 0:
            return True
    except (TypeError, ValueError):
        pass

    for field_name in ("sources", "evidence_manifest"):
        value = row.get(field_name)
        if value is None:
            continue
        if isinstance(value, str):
            value = value.strip()
            if not value:
                continue
            try:
                value = json.loads(value)
            except json.JSONDecodeError:
                # Legacy ``sources`` may be a plain source name rather than JSON.
                if field_name == "sources":
                    return True
                continue
        if isinstance(value, (dict, list, tuple, set)):
            if value:
                return True
            continue
        if not pd.isna(value) and bool(value):
            return True

    return False


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

    # Standard results expose ``sources``; agentic results persist an evidence
    # manifest or an explicit document count instead.
    if not _has_retrieved_evidence(row):
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

    # New result files use this for a post-selection evaluator; older files may
    # contain a policy signal. Keep generic score statistics and use
    # evaluation_mode/correct to distinguish their semantics.
    if 'judge_score' in results_df.columns:
        judge_values = pd.to_numeric(
            results_df['judge_score'], errors='coerce'
        ).dropna()
        if len(judge_values) > 0:
            judge_list = judge_values.tolist()
            mean, ci_lower, ci_upper = bootstrap_ci(judge_list)
            thresholds = _judge_correctness_thresholds(
                results_df, judge_values.index
            )
            full_credit = (judge_values >= thresholds).astype(float)
            full_mean, full_lower, full_upper = bootstrap_ci(
                full_credit.tolist()
            )
            metrics['judge_score'] = {
                'mean': mean,
                'ci_95': [ci_lower, ci_upper],
                'std': float(judge_values.std()),
                'threshold_pass_rate': full_mean,
                'threshold_pass_rate_ci_95': [full_lower, full_upper],
                'correctness_thresholds': sorted(set(thresholds.tolist())),
                'partial_credit_rate_at_0_5': float(
                    (judge_values >= 0.5).mean()
                ),
                'count': int(len(judge_values)),
            }
        else:
            metrics['judge_score'] = {
                'mean': 0.0, 'ci_95': [0.0, 0.0], 'std': 0.0,
                'threshold_pass_rate': 0.0,
                'threshold_pass_rate_ci_95': [0.0, 0.0],
                'correctness_thresholds': [],
                'partial_credit_rate_at_0_5': 0.0,
                'count': 0,
            }

    if 'policy_accepted' in results_df.columns:
        accepted = _coerce_binary_series(
            results_df['policy_accepted'], 'policy_accepted'
        )
        if len(accepted) > 0:
            mean, ci_lower, ci_upper = bootstrap_ci(accepted.tolist())
            metrics['policy_acceptance'] = {
                'rate': mean,
                'ci_95': [ci_lower, ci_upper],
                'count': int(len(accepted)),
            }

    if 'correct' in results_df.columns:
        correctness = _coerce_binary_series(results_df['correct'], 'correct')
        if len(correctness) > 0:
            mean, ci_lower, ci_upper = bootstrap_ci(correctness.tolist())
            modes = set(results_df.loc[correctness.index, 'evaluation_mode'].dropna()) \
                if 'evaluation_mode' in results_df.columns else set()
            label = (
                'oracle-guided fixed-threshold correctness-label rate'
                if modes == {'oracle_guided'}
                else (
                    'post-selection correctness-label rate'
                    if modes and all(
                        str(mode).startswith(('post_selection_', 'terminal_'))
                        for mode in modes
                    )
                    else 'provided correctness-label rate'
                )
            )
            metrics['labeled_correctness'] = {
                'rate': mean,
                'ci_95': [ci_lower, ci_upper],
                'count': int(len(correctness)),
                'label': label,
            }

    if 'abstained' in results_df.columns or 'error' in results_df.columns:
        total_count = len(results_df)
        abstained = pd.Series(0.0, index=results_df.index, dtype=float)
        if 'abstained' in results_df.columns:
            abstained = _coerce_binary_series(
                results_df['abstained'], 'abstained'
            ).reindex(results_df.index).fillna(0.0)
        errors = pd.Series(False, index=results_df.index, dtype=bool)
        if 'error' in results_df.columns:
            errors = _error_mask(results_df['error']).reindex(
                results_df.index, fill_value=False
            )

        # Terminal errors take precedence if a malformed row marks both states.
        abstention_mask = (abstained == 1.0) & ~errors
        covered_mask = ~errors & ~abstention_mask
        answered_index = results_df.index[covered_mask]
        covered_count = int(covered_mask.sum())
        abstention_count = int(abstention_mask.sum())
        error_count = int(errors.sum())
        noncoverage_count = total_count - covered_count
        coverage = covered_count / total_count if total_count else 0.0
        selective = None
        selective_count = 0
        if 'correct' in results_df.columns:
            correctness = _coerce_binary_series(results_df['correct'], 'correct')
            selective_values = correctness.reindex(answered_index).dropna()
            selective_count = int(len(selective_values))
            if selective_count:
                selective = float(selective_values.mean())
        metrics['selective_prediction'] = {
            'coverage': float(coverage),
            'noncoverage_rate': (
                float(noncoverage_count / total_count) if total_count else 0.0
            ),
            'abstention_rate': (
                float(abstention_count / total_count) if total_count else 0.0
            ),
            'error_rate': (
                float(error_count / total_count) if total_count else 0.0
            ),
            'selective_accuracy': selective,
            'selective_risk': (
                float(1.0 - selective) if selective is not None else None
            ),
            'answered_evaluated_count': selective_count,
            'covered_question_count': covered_count,
            'abstention_count': abstention_count,
            'error_count': error_count,
            'noncoverage_count': noncoverage_count,
            'total_question_count': int(total_count),
            # Kept for readers of older summaries; eligibility now means every
            # requested question rather than only error-free rows.
            'eligible_question_count': int(total_count),
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
                judge_vals = pd.to_numeric(
                    type_df['judge_score'], errors='coerce'
                ).dropna()
                n_samples = len(judge_vals)
                thresholds = _judge_correctness_thresholds(
                    results_df, judge_vals.index
                )
                binary_scores = (
                    (judge_vals >= thresholds).astype(float).tolist()
                )
                threshold_pass_rate = (
                    float(np.mean(binary_scores)) if binary_scores else 0.0
                )
                partial_credit_rate = (
                    float((judge_vals >= 0.5).mean()) if n_samples else 0.0
                )

                if n_samples >= MIN_SAMPLES_FOR_CI:
                    mean, ci_lower, ci_upper = bootstrap_ci(judge_vals.tolist())
                    pass_mean, pass_lower, pass_upper = bootstrap_ci(binary_scores)
                    type_metrics['judge_score'] = {
                        'mean': mean,
                        'ci_95': [ci_lower, ci_upper],
                        'threshold_pass_rate': pass_mean,
                        'threshold_pass_rate_ci_95': [pass_lower, pass_upper],
                        'correctness_thresholds': sorted(
                            set(thresholds.tolist())
                        ),
                        'partial_credit_rate_at_0_5': partial_credit_rate,
                        'count': n_samples
                    }
                elif n_samples > 0:
                    type_metrics['judge_score'] = {
                        'mean': float(judge_vals.mean()),
                        'ci_95': None,
                        'threshold_pass_rate': threshold_pass_rate,
                        'threshold_pass_rate_ci_95': None,
                        'correctness_thresholds': sorted(
                            set(thresholds.tolist())
                        ),
                        'partial_credit_rate_at_0_5': partial_credit_rate,
                        'count': n_samples
                    }
                else:
                    type_metrics['judge_score'] = {
                        'mean': 0.0,
                        'ci_95': None,
                        'threshold_pass_rate': 0.0,
                        'threshold_pass_rate_ci_95': None,
                        'correctness_thresholds': [],
                        'partial_credit_rate_at_0_5': 0.0,
                        'count': 0
                    }

            if 'policy_accepted' in type_df.columns:
                accepted = _coerce_binary_series(
                    type_df['policy_accepted'], 'policy_accepted'
                )
                if len(accepted) > 0:
                    type_metrics['policy_acceptance_rate'] = float(accepted.mean())

            if 'correct' in type_df.columns:
                correctness = _coerce_binary_series(type_df['correct'], 'correct')
                if len(correctness) > 0:
                    type_metrics['labeled_correctness_rate'] = float(correctness.mean())

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
        error_count = int(_error_mask(results_df['error']).sum())
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
        lines.append("\nSemantic Similarity:")
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
        lines.append("\nLLM Judge:")
        lines.append(f"  Mean Score: {judge['mean']:.4f}")
        if 'ci_95' in judge:
            ci = judge['ci_95']
            lines.append(f"  95% CI:     [{ci[0]:.4f}, {ci[1]:.4f}]")
        thresholds = judge.get('correctness_thresholds', [])
        threshold_label = ", ".join(f"{value:g}" for value in thresholds)
        lines.append(
            f"  Full-credit Rate: {judge['threshold_pass_rate']:.2%} "
            f"(thresholds: {threshold_label or 'not available'})"
        )
        lines.append(
            "  Partial-credit >=0.5: "
            f"{judge.get('partial_credit_rate_at_0_5', 0.0):.2%}"
        )
        lines.append(f"  Count:      {judge['count']}")

    if 'policy_acceptance' in metrics:
        acceptance = metrics['policy_acceptance']
        lines.append("\nPolicy Acceptance:")
        lines.append(f"  Rate:  {acceptance['rate']:.2%}")
        lines.append(
            f"  95% CI: [{acceptance['ci_95'][0]:.2%}, "
            f"{acceptance['ci_95'][1]:.2%}]"
        )
        lines.append(f"  Count: {acceptance['count']}")

    if 'labeled_correctness' in metrics:
        correctness = metrics['labeled_correctness']
        lines.append(f"\nCorrectness label ({correctness['label']}):")
        lines.append(f"  Rate:  {correctness['rate']:.2%}")
        lines.append(
            f"  95% CI: [{correctness['ci_95'][0]:.2%}, "
            f"{correctness['ci_95'][1]:.2%}]"
        )
        lines.append(f"  Count: {correctness['count']}")

    if 'selective_prediction' in metrics:
        selective = metrics['selective_prediction']
        lines.append("\nSelective Prediction:")
        lines.append(f"  Coverage:        {selective['coverage']:.2%}")
        lines.append(f"  Noncoverage:     {selective['noncoverage_rate']:.2%}")
        lines.append(f"  Abstention rate: {selective['abstention_rate']:.2%}")
        lines.append(f"  Terminal errors: {selective['error_rate']:.2%}")
        if selective['selective_accuracy'] is not None:
            lines.append(
                f"  Selective accuracy: {selective['selective_accuracy']:.2%}"
            )
            lines.append(f"  Selective risk:     {selective['selective_risk']:.2%}")

    # Per question type (with CIs when available)
    if 'by_question_type' in metrics and metrics['by_question_type']:
        lines.append("\nBy Question Type:")
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
                    if judge.get('threshold_pass_rate') is not None:
                        pass_str = f"{judge['threshold_pass_rate']:.2%}"
                        if judge.get('threshold_pass_rate_ci_95'):
                            pass_ci = judge['threshold_pass_rate_ci_95']
                            lines.append(f"    Full credit:  {pass_str} [{pass_ci[0]:.1%}, {pass_ci[1]:.1%}]")
                        else:
                            lines.append(f"    Full credit:  {pass_str}")
                    if judge.get('partial_credit_rate_at_0_5') is not None:
                        lines.append(
                            "    Partial >=.5: "
                            f"{judge['partial_credit_rate_at_0_5']:.2%}"
                        )
                else:
                    # Legacy format fallback
                    lines.append(f"    Judge Score:  {judge:.4f}")

            # Legacy format fallback for judge_score_mean
            elif 'judge_score_mean' in type_metrics:
                lines.append(f"    Judge Score:  {type_metrics['judge_score_mean']:.4f}")
                if 'judge_accuracy' in type_metrics:
                    lines.append(f"    Legacy judge pass: {type_metrics['judge_accuracy']:.2%}")

            if 'policy_acceptance_rate' in type_metrics:
                lines.append(
                    f"    Policy accepted: {type_metrics['policy_acceptance_rate']:.2%}"
                )
            if 'labeled_correctness_rate' in type_metrics:
                lines.append(
                    "    Correctness label: "
                    f"{type_metrics['labeled_correctness_rate']:.2%}"
                )

    # Timing
    if 'retrieval_time_ms' in metrics:
        ret = metrics['retrieval_time_ms']
        lines.append("\nRetrieval Latency (ms):")
        lines.append(f"  Mean: {ret['mean']:.1f}  P50: {ret['p50']:.1f}  P95: {ret['p95']:.1f}")

    if 'generation_time_ms' in metrics:
        gen = metrics['generation_time_ms']
        lines.append("\nGeneration Latency (ms):")
        lines.append(f"  Mean: {gen['mean']:.1f}  P50: {gen['p50']:.1f}  P95: {gen['p95']:.1f}")

    # Error rate
    if 'error_rate' in metrics:
        lines.append(f"\nError Rate: {metrics['error_rate']:.2%}")

    # Numeric verification (hallucination check)
    if 'numeric_verification' in metrics:
        num = metrics['numeric_verification']
        lines.append("\nNumeric Verification (vs Sources):")
        lines.append(f"  Mean Score:        {num['mean']:.4f}")
        lines.append(f"  Hallucination Rate: {num['hallucination_rate']:.2%}")
        lines.append(f"  Perfect Rate:      {num['perfect_rate']:.2%}")
        lines.append(f"  Count:             {num['count']}")

    # Numeric accuracy (exact-match against gold)
    if 'numeric_accuracy' in metrics:
        num_acc = metrics['numeric_accuracy']
        lines.append("\nNumeric Accuracy (vs Gold Answer):")
        lines.append(f"  Exact Match Rate: {num_acc['exact_match_rate']:.2%}")
        if 'ci_95' in num_acc:
            ci = num_acc['ci_95']
            lines.append(f"  95% CI:           [{ci[0]:.4f}, {ci[1]:.4f}]")
        lines.append(f"  Numeric Questions: {num_acc['numeric_questions']}/{num_acc['total_questions']}")

    # Failure breakdown
    if 'failure_breakdown' in metrics:
        fb = metrics['failure_breakdown']
        lines.append("\nFailure Breakdown:")
        for category, pct in sorted(fb['percentages'].items(), key=lambda x: -x[1]):
            count = fb['counts'].get(category, 0)
            lines.append(f"  {category:25} {pct:6.1%} ({count})")

    lines.append("\n" + "=" * 60)

    return "\n".join(lines)
