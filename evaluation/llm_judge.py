"""LLM-as-a-Judge evaluation for RAG system."""

import sys
from pathlib import Path
from typing import Tuple
import math
import re

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.providers import get_provider


class JudgeEvaluationError(RuntimeError):
    """The external outcome/policy evaluator failed to produce a score."""


JUDGE_SYSTEM_PROMPT = """You are an expert evaluator for question-answering systems. Your task is to judge whether a predicted answer is correct compared to a gold/reference answer.

Guidelines:
1. Focus on factual correctness, not exact wording
2. For numerical answers, allow reasonable formatting differences (e.g., "$1,577" vs "1577 million" vs "$1.577 billion")
3. For percentage/ratio questions, ensure the numbers are equivalent
4. Partial credit is acceptable - give 0.5 for partially correct answers
5. Consider that the predicted answer may contain additional context that doesn't make it wrong

Scoring:
- 1.0: Fully correct answer
- 0.5: Partially correct (main fact correct but missing details, or close but not exact)
- 0.0: Incorrect answer

You MUST respond in the following format:
SCORE: <score>
JUSTIFICATION: <brief explanation>"""


JUDGE_USER_PROMPT_TEMPLATE = """Question: {question}

Gold Answer: {gold_answer}

Predicted Answer: {predicted_answer}

Evaluate the predicted answer against the gold answer and provide your score and justification."""


def parse_judge_response(response: str) -> Tuple[float, str]:
    """Strictly parse one score and one non-empty justification.

    Args:
        response: Raw response from the judge LLM

    Returns:
        Tuple of (score, justification)
    """
    if not isinstance(response, str):
        raise JudgeEvaluationError("Judge response must be text")
    score_matches = re.findall(
        r"(?im)^\s*SCORE:\s*([^\r\n]+?)\s*$",
        response,
    )
    if len(score_matches) != 1:
        raise JudgeEvaluationError(
            f"Expected exactly one SCORE field, found {len(score_matches)}"
        )
    raw_score = score_matches[0].strip()
    if not re.fullmatch(r"(?:0(?:\.\d+)?|1(?:\.0+)?)", raw_score):
        raise JudgeEvaluationError("SCORE must be a finite number in [0, 1]")
    score = float(raw_score)
    if not math.isfinite(score) or not 0.0 <= score <= 1.0:
        raise JudgeEvaluationError("SCORE must be a finite number in [0, 1]")

    justification_markers = list(
        re.finditer(r"(?im)^\s*JUSTIFICATION:\s*", response)
    )
    if len(justification_markers) != 1:
        raise JudgeEvaluationError(
            "Expected exactly one JUSTIFICATION field"
        )
    justification = response[justification_markers[0].end() :].strip()
    if not justification:
        raise JudgeEvaluationError("JUSTIFICATION must be non-empty")
    if re.search(r"(?im)^\s*(?:SCORE|JUSTIFICATION):", justification):
        raise JudgeEvaluationError("Judge response contains duplicate fields")
    return score, justification


def llm_as_judge(
    question: str,
    gold_answer: str,
    predicted_answer: str,
    judge_model: str = "gpt-4o-mini",
) -> Tuple[float, str]:
    """Use an LLM to judge whether the predicted answer is correct.

    Args:
        question: The original question
        gold_answer: The gold/reference answer
        predicted_answer: The predicted answer to evaluate
        judge_model: Model to use for judging (default: Claude Sonnet 4.5)

    Returns:
        Tuple of (score between 0-1, justification string)
    """
    if not predicted_answer or not gold_answer:
        return 0.0, "Empty answer"

    try:
        # Get provider for the judge model
        provider = get_provider(judge_model)

        # Format the prompt
        user_prompt = JUDGE_USER_PROMPT_TEMPLATE.format(
            question=question,
            gold_answer=gold_answer,
            predicted_answer=predicted_answer,
        )

        # Generate judge response
        response = provider.generate(
            system_prompt=JUDGE_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            max_tokens=256,
            temperature=0.0,
        )

        # Parse the response
        score, justification = parse_judge_response(response.content)

        return score, justification

    except Exception as e:
        raise JudgeEvaluationError(f"Judge evaluation failed: {e}") from e


def llm_as_judge_batch(
    questions: list,
    gold_answers: list,
    predicted_answers: list,
    judge_model: str = "gpt-4o-mini",
) -> list:
    """Evaluate multiple predictions at once (sequential).

    Args:
        questions: List of questions
        gold_answers: List of gold answers
        predicted_answers: List of predicted answers
        judge_model: Model to use for judging

    Returns:
        List of (score, justification) tuples
    """
    results = []
    for q, gold, pred in zip(questions, gold_answers, predicted_answers):
        score, justification = llm_as_judge(q, gold, pred, judge_model)
        results.append((score, justification))
    return results


def llm_as_judge_parallel(
    questions: list,
    gold_answers: list,
    predicted_answers: list,
    judge_model: str = "gpt-4o-mini",
    max_workers: int = 10,
) -> list:
    """Evaluate multiple predictions in parallel for ~10x speedup.

    Uses ThreadPoolExecutor to run LLM judge calls concurrently.
    This is safe because LLM API calls are I/O-bound, not CPU-bound.

    Args:
        questions: List of questions
        gold_answers: List of gold answers
        predicted_answers: List of predicted answers
        judge_model: Model to use for judging (default: gpt-4o-mini for speed)
        max_workers: Maximum number of concurrent API calls (default: 10)

    Returns:
        List of (score, justification) tuples in the same order as inputs
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    n = len(questions)
    if n == 0:
        return []

    # Create list to store results in order
    results = [None] * n

    def evaluate_single(idx: int) -> Tuple[int, Tuple[float, str]]:
        """Evaluate a single question and return (index, result)."""
        score, justification = llm_as_judge(
            questions[idx],
            gold_answers[idx],
            predicted_answers[idx],
            judge_model,
        )
        return idx, (score, justification)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        futures = {
            executor.submit(evaluate_single, i): i
            for i in range(n)
        }

        # Collect results as they complete
        for future in as_completed(futures):
            try:
                idx, result = future.result()
                results[idx] = result
            except Exception as e:
                idx = futures[future]
                results[idx] = (0.0, f"Parallel eval error: {str(e)}")

    return results


def llm_as_judge_with_numeric_check(
    question: str,
    gold_answer: str,
    predicted_answer: str,
    judge_model: str = "gpt-4o-mini",
    use_numeric_augmentation: bool = True,
) -> Tuple[float, str]:
    """LLM-as-Judge with tool-augmented numeric verification.

    This implements the hybrid evaluation approach recommended for financial QA:
    1. Run standard LLM-as-Judge for semantic evaluation
    2. Run deterministic numeric check to catch magnitude errors
    3. Override LLM if numeric check finds clear mismatch/match

    Per GPT-5.2 research: "66% of FinanceBench questions involve numerical
    calculation. A vanilla LLM-as-Judge may not consistently catch magnitude
    errors or subtle numeric discrepancies."

    Args:
        question: The original question
        gold_answer: The gold/reference answer
        predicted_answer: The predicted answer to evaluate
        judge_model: Model to use for judging
        use_numeric_augmentation: Whether to use numeric verification (default True)

    Returns:
        Tuple of (score between 0-1, justification string)
    """
    # First, get LLM judge score
    llm_score, llm_justification = llm_as_judge(
        question=question,
        gold_answer=gold_answer,
        predicted_answer=predicted_answer,
        judge_model=judge_model,
    )

    # If numeric augmentation disabled, return LLM score directly
    if not use_numeric_augmentation:
        return llm_score, llm_justification

    # Augment with numeric verification
    try:
        from evaluation.numeric_check import augmented_judge
        final_score, final_justification = augmented_judge(
            question=question,
            gold_answer=gold_answer,
            predicted_answer=predicted_answer,
            llm_score=llm_score,
            llm_justification=llm_justification,
        )
        return final_score, final_justification
    except ImportError:
        # Fall back to LLM-only if numeric_check not available
        return llm_score, llm_justification
    except Exception as e:
        # On any error, fall back to LLM score
        print(f"Warning: Numeric check failed: {e}")
        return llm_score, f"{llm_justification} [numeric check failed: {e}]"
