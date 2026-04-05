"""Oracle label generation for meta-learning router.

This script runs all retrieval pipelines on each question to determine
the optimal pipeline per question. Used to:
1. Generate training labels for meta-learning router
2. Calculate upper-bound (oracle) accuracy
3. Determine if routing has enough signal to be worthwhile
"""

import sys
import json
import argparse
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain_chroma import Chroma

from dataset_adapters import FinanceBenchAdapter
from evaluation.metrics import embedding_similarity
from src.retrieval_tools.tool_registry import (
    build_pipeline,
    build_retriever_for_pipeline,
)
from src.config import DEFAULTS, get_embedding_model

load_dotenv()

# Pipelines to compare (excluding 'routed' which is the router itself)
PIPELINES = ["semantic", "hybrid", "hybrid_filter", "hybrid_filter_rerank"]


@dataclass
class OracleConfig:
    """Configuration for oracle label generation."""
    dataset_name: str = "financebench"
    chroma_path: str = "chroma_docling"
    embedding_model: str = DEFAULTS.embedding_model
    reranker_model: str = DEFAULTS.reranker_model
    top_k: int = 5
    initial_k_factor: float = 3.0
    output_dir: str = "oracle_labels"
    timestamp: str = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        # Resolve paths relative to project root
        project_root = Path(__file__).parent.parent
        if not Path(self.chroma_path).is_absolute():
            self.chroma_path = str(project_root / self.chroma_path)
        if not Path(self.output_dir).is_absolute():
            self.output_dir = str(project_root / self.output_dir)


class OracleLabelGenerator:
    """Generates oracle labels by running all pipelines on each question."""

    def __init__(self, config: OracleConfig):
        self.config = config
        self.pipelines: Dict[str, Any] = {}
        self.db = None
        self.embedding_fn = None

    def initialize(self):
        """Initialize ChromaDB and all pipelines."""
        print(f"Loading embedding model: {self.config.embedding_model}")
        self.embedding_fn = get_embedding_model(self.config.embedding_model)

        print(f"Loading ChromaDB from: {self.config.chroma_path}")
        self.db = Chroma(
            persist_directory=self.config.chroma_path,
            embedding_function=self.embedding_fn,
        )
        print(f"  Loaded {self.db._collection.count()} chunks")

        # Build all pipelines
        print("Building retrieval pipelines...")
        for pipeline_id in PIPELINES:
            retriever, set_k_fn, take_top_k_fn, use_hybrid = build_retriever_for_pipeline(
                pipeline_id, self.db, top_k=self.config.top_k
            )
            pipeline = build_pipeline(
                pipeline_id=pipeline_id,
                retriever=retriever,
                top_k=self.config.top_k,
                initial_k_factor=self.config.initial_k_factor,
                set_k_fn=set_k_fn,
                take_top_k_fn=take_top_k_fn,
                reranker_model=self.config.reranker_model,
                db=self.db,
                use_hybrid=use_hybrid,
            )
            self.pipelines[pipeline_id] = pipeline
            print(f"  Built: {pipeline_id}")

    def evaluate_question(
        self,
        question: str,
        gold_answer: str,
    ) -> Dict[str, Any]:
        """Run all pipelines on a question and score retrieval quality.

        Uses embedding similarity between retrieved context and gold answer
        as the scoring metric (same as bulk_testing.py).
        """
        scores = {}

        for pipeline_id, pipeline in self.pipelines.items():
            try:
                docs = pipeline.retrieve(question)
                context = "\n\n".join([d.page_content for d in docs])

                # Score by similarity to gold answer
                score = embedding_similarity(
                    context, gold_answer, self.embedding_fn
                )
                scores[pipeline_id] = round(score, 4)
            except Exception as e:
                print(f"  Error with {pipeline_id}: {e}")
                scores[pipeline_id] = 0.0

        # Determine best pipeline
        best_pipeline = max(scores, key=scores.get)
        best_score = scores[best_pipeline]

        return {
            "scores": scores,
            "best_pipeline": best_pipeline,
            "best_score": best_score,
            "score_spread": max(scores.values()) - min(scores.values()),
        }

    def run(self, dataset: pd.DataFrame, question_col: str, answer_col: str) -> pd.DataFrame:
        """Run oracle evaluation on entire dataset."""
        results = []

        for idx, row in tqdm(dataset.iterrows(), total=len(dataset), desc="Oracle eval"):
            question = row[question_col]
            gold_answer = row[answer_col]

            eval_result = self.evaluate_question(question, gold_answer)

            result = {
                "idx": idx,
                "question": question[:100] + "..." if len(question) > 100 else question,
                "best_pipeline": eval_result["best_pipeline"],
                "best_score": eval_result["best_score"],
                "score_spread": eval_result["score_spread"],
            }

            # Add individual pipeline scores
            for pipeline_id, score in eval_result["scores"].items():
                result[f"score_{pipeline_id}"] = score

            # Add question type if available
            if "question_type" in row:
                result["question_type"] = row["question_type"]

            results.append(result)

        return pd.DataFrame(results)

    def analyze_results(self, results_df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze oracle results to determine if routing is worthwhile."""
        analysis = {}

        # Pipeline distribution
        pipeline_counts = results_df["best_pipeline"].value_counts()
        analysis["pipeline_distribution"] = pipeline_counts.to_dict()
        analysis["pipeline_percentages"] = (pipeline_counts / len(results_df) * 100).round(1).to_dict()

        # Score statistics
        score_cols = [c for c in results_df.columns if c.startswith("score_")]
        for col in score_cols:
            pipeline = col.replace("score_", "")
            analysis[f"{pipeline}_mean"] = round(results_df[col].mean(), 4)

        # Oracle vs best fixed pipeline
        analysis["oracle_mean"] = round(results_df["best_score"].mean(), 4)

        best_fixed_scores = {
            pipeline: results_df[f"score_{pipeline}"].mean()
            for pipeline in PIPELINES
        }
        best_fixed_pipeline = max(best_fixed_scores, key=best_fixed_scores.get)
        analysis["best_fixed_pipeline"] = best_fixed_pipeline
        analysis["best_fixed_mean"] = round(best_fixed_scores[best_fixed_pipeline], 4)

        # Key metric: potential gain from routing
        analysis["potential_gain"] = round(
            analysis["oracle_mean"] - analysis["best_fixed_mean"], 4
        )

        # Is routing worthwhile? (threshold: 0.05 gain, no single pipeline >85%)
        dominant_pct = max(analysis["pipeline_percentages"].values())
        analysis["dominant_pipeline_pct"] = float(dominant_pct)
        analysis["routing_worthwhile"] = bool(
            analysis["potential_gain"] > 0.05 and dominant_pct < 85
        )

        # By question type (if available)
        if "question_type" in results_df.columns:
            by_type = {}
            for qtype in results_df["question_type"].unique():
                type_df = results_df[results_df["question_type"] == qtype]
                type_dist = type_df["best_pipeline"].value_counts()
                by_type[qtype] = {
                    "count": len(type_df),
                    "distribution": type_dist.to_dict(),
                    "oracle_mean": round(type_df["best_score"].mean(), 4),
                }
            analysis["by_question_type"] = by_type

        return analysis


def load_dataset(dataset_name: str) -> tuple[pd.DataFrame, str, str]:
    """Load dataset and return (df, question_col, answer_col)."""
    if dataset_name == "financebench":
        adapter = FinanceBenchAdapter()
        df = adapter.load_dataset()
        return df, adapter.get_question_column(), adapter.get_answer_column()
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def main():
    parser = argparse.ArgumentParser(description="Generate oracle labels for meta-learning")
    parser.add_argument("--dataset", default="financebench", choices=["financebench"])
    parser.add_argument("--chroma-path", default="chroma_docling")
    parser.add_argument("--embedding", default=DEFAULTS.embedding_model)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--output-dir", default="oracle_labels")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of questions (for testing)")
    args = parser.parse_args()

    config = OracleConfig(
        dataset_name=args.dataset,
        chroma_path=args.chroma_path,
        embedding_model=args.embedding,
        top_k=args.top_k,
        output_dir=args.output_dir,
    )

    # Create output directory
    output_path = Path(config.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load dataset
    print(f"\nLoading dataset: {config.dataset_name}")
    df, question_col, answer_col = load_dataset(config.dataset_name)
    print(f"  Loaded {len(df)} questions")

    if args.limit:
        df = df.head(args.limit)
        print(f"  Limited to {len(df)} questions")

    # Initialize generator
    generator = OracleLabelGenerator(config)
    generator.initialize()

    # Run evaluation
    print("\nRunning oracle evaluation...")
    results_df = generator.run(df, question_col, answer_col)

    # Analyze results
    print("\nAnalyzing results...")
    analysis = generator.analyze_results(results_df)

    # Save outputs
    csv_path = output_path / f"{config.dataset_name}_{config.timestamp}.csv"
    json_path = output_path / f"{config.dataset_name}_{config.timestamp}_analysis.json"

    results_df.to_csv(csv_path, index=False)
    with open(json_path, "w") as f:
        json.dump(analysis, f, indent=2)

    print(f"\nResults saved to:")
    print(f"  CSV: {csv_path}")
    print(f"  Analysis: {json_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("ORACLE ANALYSIS SUMMARY")
    print("=" * 60)
    print(f"\nPipeline Distribution:")
    for pipeline, pct in analysis["pipeline_percentages"].items():
        print(f"  {pipeline}: {pct}%")

    print(f"\nMean Scores:")
    for pipeline in PIPELINES:
        print(f"  {pipeline}: {analysis[f'{pipeline}_mean']}")

    print(f"\nKey Metrics:")
    print(f"  Oracle (upper bound): {analysis['oracle_mean']}")
    print(f"  Best fixed ({analysis['best_fixed_pipeline']}): {analysis['best_fixed_mean']}")
    print(f"  Potential gain: {analysis['potential_gain']}")
    print(f"  Dominant pipeline: {analysis['dominant_pipeline_pct']}%")

    print(f"\n{'✓' if analysis['routing_worthwhile'] else '✗'} Routing worthwhile: {analysis['routing_worthwhile']}")

    if "by_question_type" in analysis:
        print(f"\nBy Question Type:")
        for qtype, stats in analysis["by_question_type"].items():
            print(f"  {qtype} (n={stats['count']}): oracle={stats['oracle_mean']}")


if __name__ == "__main__":
    main()
