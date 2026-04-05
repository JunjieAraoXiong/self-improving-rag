"""Evaluate on rLLM's FinQA benchmark using our TableAgent.

Runs the TableAgent (with any LLM) on the FinQA test set and reports
Pass@1 accuracy, directly comparable to rLLM's published numbers.

Usage:
    # With GPT-4o-mini (default):
    python scripts/eval_finqa.py --model gpt-4o-mini --n 50

    # With rLLM-FinQA-4B via vLLM:
    python scripts/eval_finqa.py --vllm-base-url http://localhost:30000/v1 --n 558

    # Full test set:
    python scripts/eval_finqa.py --model gpt-4o-mini --n 558
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

import pandas as pd
from tqdm import tqdm

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "rllm"))

from dotenv import load_dotenv
load_dotenv()

from src.agents.table_agent import TableAgent


def load_finqa_test(n: int = None) -> pd.DataFrame:
    """Load FinQA test set from rLLM's data directory."""
    test_path = Path(__file__).parent.parent.parent / "rllm" / "projects" / "finqa" / "data" / "test_finqa.csv"
    if not test_path.exists():
        print("FinQA test data not found. Run: python -m projects.finqa.prepare_finqa_data")
        sys.exit(1)
    df = pd.read_csv(test_path)
    if n and n < len(df):
        df = df.head(n)
    return df


def judge_correctness(predicted: str, gold: str, tolerance: float = 0.05) -> bool:
    """Simple numeric correctness check (matches rLLM's binary reward)."""
    if not predicted or not gold:
        return False

    # Clean both
    pred_clean = re.sub(r'[,$%\s]', '', str(predicted).strip())
    gold_clean = re.sub(r'[,$%\s]', '', str(gold).strip())

    # Try exact string match first
    if pred_clean == gold_clean:
        return True

    # Try numeric comparison
    try:
        pred_val = float(pred_clean)
        gold_val = float(gold_clean)
        if gold_val == 0:
            return abs(pred_val) < 0.01
        return abs(pred_val - gold_val) / abs(gold_val) <= tolerance
    except ValueError:
        # Fall back to string containment
        return gold_clean.lower() in pred_clean.lower()


def main():
    parser = argparse.ArgumentParser(description="Evaluate on FinQA benchmark")
    parser.add_argument("--model", default="gpt-4o-mini", help="LLM model for ReAct agent")
    parser.add_argument("--vllm-base-url", default=None, help="vLLM server URL for rLLM-FinQA-4B")
    parser.add_argument("--n", type=int, default=50, help="Number of questions (default: 50, full: 558)")
    parser.add_argument("--max-steps", type=int, default=20, help="Max ReAct steps per question")
    parser.add_argument("--output", default=None, help="Output CSV path")
    args = parser.parse_args()

    # Load test set
    df = load_finqa_test(args.n)
    print(f"FinQA test set: {len(df)} questions")
    print(f"Model: {args.vllm_base_url or args.model}")

    # Create agent
    agent = TableAgent(
        model_name=args.model,
        vllm_base_url=args.vllm_base_url,
        max_steps=args.max_steps,
    )

    # Run evaluation
    results = []
    correct = 0
    total = 0
    errors = 0

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating"):
        question = row["question"]
        company = row["company"]
        gold_answer = str(row["answer"])
        q_type = row.get("question_type", "")

        start = time.time()
        try:
            decision = agent.decide({
                "question": question,
                "company": company,
            })
            predicted = decision.decision_value.get("answer")
            error = decision.decision_value.get("error")
            trace = decision.decision_value.get("computation_trace", [])
            latency = (time.time() - start) * 1000
        except Exception as e:
            predicted = None
            error = str(e)
            trace = []
            latency = (time.time() - start) * 1000

        # Judge
        is_correct = judge_correctness(predicted, gold_answer) if predicted else False
        if is_correct:
            correct += 1
        if error:
            errors += 1
        total += 1

        results.append({
            "idx": idx,
            "company": company,
            "question": question[:100],
            "question_type": q_type,
            "gold_answer": gold_answer,
            "predicted": predicted,
            "correct": is_correct,
            "error": error,
            "n_steps": len(trace),
            "latency_ms": latency,
        })

        # Progress update every 10 questions
        if total % 10 == 0:
            acc = correct / total * 100
            print(f"  [{total}/{len(df)}] Accuracy: {acc:.1f}% | Errors: {errors}")

        # Reset agent for next question
        agent.reset()

    # Summary
    accuracy = correct / total * 100 if total > 0 else 0
    print(f"\n{'='*50}")
    print(f"FinQA Results ({args.vllm_base_url or args.model})")
    print(f"{'='*50}")
    print(f"Pass@1:    {accuracy:.1f}% ({correct}/{total})")
    print(f"Errors:    {errors}/{total}")
    print(f"Avg steps: {sum(r['n_steps'] for r in results) / len(results):.1f}")
    print(f"Avg latency: {sum(r['latency_ms'] for r in results) / len(results):.0f}ms")

    # Breakdown by question type
    results_df = pd.DataFrame(results)
    if "question_type" in results_df.columns:
        print(f"\nBy question type:")
        for qt, group in results_df.groupby("question_type"):
            qt_acc = group["correct"].mean() * 100
            print(f"  {qt}: {qt_acc:.1f}% ({group['correct'].sum()}/{len(group)})")

    # Save results
    if args.output:
        output_path = args.output
    else:
        model_tag = "rllm-4b" if args.vllm_base_url else args.model.split("/")[-1]
        output_path = f"bulk_runs/finqa_{model_tag}_n{total}.csv"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path, index=False)
    print(f"\nResults saved: {output_path}")

    # Reference comparison
    print(f"\n--- Reference (rLLM blog) ---")
    print(f"Qwen3-4B base:     27.9%")
    print(f"rLLM-FinQA-4B:     59.7%")
    print(f"GPT-4.1:           62.7%")
    print(f"Ours ({model_tag}): {accuracy:.1f}%")


if __name__ == "__main__":
    main()
