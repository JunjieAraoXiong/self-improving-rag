#!/usr/bin/env python3
"""Generate silver labels for Adaptive-RAG router training.

Following the Adaptive-RAG methodology (NAACL 2024):
https://github.com/starsuzi/Adaptive-RAG

This script runs all pipeline configurations for each question and records
which configuration produces the best answer. These "silver labels" are used
to train the learned pipeline router.

Usage:
    python scripts/generate_oracle_labels.py [--resume] [--output PATH]

Output:
    data/oracle_labels.json with format:
    {
        "question_id": {
            "question": "...",
            "question_type": "metrics-generated",
            "best_config": "hybrid_filter_rerank_k10_rse",
            "best_score": 0.87,
            "all_scores": {"semantic_k5": 0.72, "hybrid_k10": 0.82, ...}
        }
    }
"""

import sys
import json
import time
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Tuple

from tqdm import tqdm
from dotenv import load_dotenv

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain_chroma import Chroma
from dataset_adapters import FinanceBenchAdapter
from evaluation.metrics import embedding_similarity
from src.retrieval_tools.tool_registry import (
    build_pipeline,
    build_retriever_for_pipeline,
)
from src.config import (
    DEFAULTS,
    get_embedding_model,
    get_provider_for_model,
)
from src.providers import get_provider

# Load environment variables
load_dotenv()


# =============================================================================
# Configuration (Adaptive-RAG Style)
# =============================================================================

# Pipeline configurations to test
# Simplified to match the 4 pipelines used by the rule-based router
# This reduces API calls: 4 configs × N questions instead of 22 × N
PIPELINE_CONFIGS = [
    # =========================================================================
    # Core pipelines (matching rule-based router options)
    # =========================================================================
    {"pipeline_id": "semantic", "top_k": 5},
    {"pipeline_id": "hybrid", "top_k": 5},
    {"pipeline_id": "hybrid_filter", "top_k": 5},
    {"pipeline_id": "hybrid_filter_rerank", "top_k": 5, "reranker": "bge"},
]

# Extended configs for comprehensive analysis (commented out for speed)
# PIPELINE_CONFIGS_EXTENDED = [
#     {"pipeline_id": "semantic", "top_k": 10},
#     {"pipeline_id": "hybrid", "top_k": 10},
#     {"pipeline_id": "hybrid_filter", "top_k": 10},
#     {"pipeline_id": "hybrid_filter_rerank", "top_k": 10, "reranker": "bge"},
#     {"pipeline_id": "hybrid_filter_rerank", "top_k": 5, "reranker": "cohere"},
#     {"pipeline_id": "hybrid_filter_rerank", "top_k": 10, "reranker": "cohere"},
#     {"pipeline_id": "hybrid_filter_rerank", "top_k": 10, "reranker": "bge", "use_rse": True},
#     {"pipeline_id": "hybrid_filter_rerank", "top_k": 10, "reranker": "bge", "use_hyde": True},
# ]

# Default model for generation (using a reliable, fast option)
DEFAULT_MODEL = "gpt-4o-mini"

# Default paths
DEFAULT_OUTPUT = "data/oracle_labels.json"
DEFAULT_CHROMA_PATH = "chroma_docling"  # Use the Docling-ingested database


def config_key(config: Dict) -> str:
    """Generate a unique key for a pipeline configuration.

    Examples:
        semantic_k5
        hybrid_filter_rerank_k10_cohere
        hybrid_filter_rerank_k10_cohere_rse
        hybrid_filter_rerank_k10_bge_rse_hyde
    """
    key = f"{config['pipeline_id']}_k{config['top_k']}"

    # Add reranker type if specified
    if config.get("reranker"):
        key += f"_{config['reranker']}"

    # Add technique flags
    if config.get("use_rse"):
        key += "_rse"
    if config.get("use_hyde"):
        key += "_hyde"

    return key


# =============================================================================
# Oracle Label Generator
# =============================================================================

class OracleLabelGenerator:
    """Generates silver labels by testing all pipeline configurations.

    Follows Adaptive-RAG methodology for creating training data.
    """

    def __init__(
        self,
        chroma_path: str,
        model_name: str = DEFAULT_MODEL,
        embedding_model: str = "bge-large",  # Use free local embeddings
    ):
        self.chroma_path = chroma_path
        self.model_name = model_name
        self.embedding_model_name = embedding_model
        self.embeddings = None
        self.db = None
        self.llm_provider = None
        self.hyde_generator = None
        self.pipelines: Dict[str, Any] = {}
        self.pipeline_configs: Dict[str, Dict] = {}  # Store config for each pipeline

    def initialize(self):
        """Initialize embeddings, database, LLM provider, and pipelines."""
        print("\nInitializing components...")

        # Initialize embeddings
        print(f"  Loading embeddings: {self.embedding_model_name}")
        self.embeddings = get_embedding_model(self.embedding_model_name)

        # Load ChromaDB
        print(f"  Loading ChromaDB from: {self.chroma_path}")
        self.db = Chroma(
            persist_directory=self.chroma_path,
            embedding_function=self.embeddings
        )
        print(f"    Loaded {self.db._collection.count()} chunks")

        # Initialize LLM provider
        print(f"  Initializing LLM: {self.model_name}")
        self.llm_provider = get_provider(self.model_name)

        # Initialize HyDE generator (lazy, only if needed)
        self.hyde_generator = None

        # Build all pipeline configurations
        print(f"  Building {len(PIPELINE_CONFIGS)} pipeline configurations...")
        for config in PIPELINE_CONFIGS:
            key = config_key(config)

            # Determine reranker model
            reranker = config.get("reranker", "bge")
            if reranker == "cohere":
                reranker_model = "cohere"
            else:
                reranker_model = DEFAULTS.reranker_model  # BGE

            # Build retriever
            retriever, set_k_fn, take_top_k_fn, use_hybrid = build_retriever_for_pipeline(
                config["pipeline_id"],
                self.db,
                top_k=config["top_k"]
            )

            # Build pipeline with RSE support
            pipeline = build_pipeline(
                pipeline_id=config["pipeline_id"],
                retriever=retriever,
                top_k=config["top_k"],
                initial_k_factor=DEFAULTS.initial_k_factor,
                set_k_fn=set_k_fn,
                take_top_k_fn=take_top_k_fn,
                reranker_model=reranker_model,
                use_rse=config.get("use_rse", False),
                db=self.db,
                use_hybrid=use_hybrid,
            )

            self.pipelines[key] = pipeline
            self.pipeline_configs[key] = config
            print(f"    Built: {key}")

        print("  Initialization complete!\n")

    def _get_hyde_generator(self):
        """Lazy-load HyDE generator."""
        if self.hyde_generator is None:
            from src.retrieval_tools.hyde import HyDE
            # Create HyDE with semantic retriever
            base_retriever = self.db.as_retriever(search_kwargs={"k": 20})
            self.hyde_generator = HyDE(
                retriever_fn=base_retriever.invoke,
                model_name="gpt-4o-mini",  # Fast model for HyDE
            )
        return self.hyde_generator

    def generate_answer(
        self,
        question: str,
        pipeline,
        config: Dict,
    ) -> Tuple[str, float]:
        """Generate answer using a specific pipeline configuration.

        Args:
            question: The question to answer
            pipeline: The retrieval pipeline
            config: Pipeline configuration dict (for HyDE, RSE flags)

        Returns:
            Tuple of (answer_text, retrieval_time_ms)
        """
        retrieval_start = time.time()

        # Check if HyDE is enabled - use hypothetical document for retrieval
        use_hyde = config.get("use_hyde", False)
        use_rse = config.get("use_rse", False)

        if use_hyde:
            # Generate hypothetical answer first
            hyde_gen = self._get_hyde_generator()
            hypothetical = hyde_gen.generate_hypothetical(question)
            # Use hypothetical for retrieval
            docs = pipeline.retrieve(hypothetical)
        else:
            docs = pipeline.retrieve(question)

        retrieval_time = (time.time() - retrieval_start) * 1000

        if not docs:
            return "", retrieval_time

        # Build context - use RSE segments if enabled
        if use_rse and hasattr(pipeline, 'retrieve_segments'):
            # RSE merges chunks into coherent segments
            if use_hyde:
                segments = pipeline.retrieve_segments(
                    self._get_hyde_generator().generate_hypothetical(question)
                )
            else:
                segments = pipeline.retrieve_segments(question)
            context = "\n\n---\n\n".join(segments) if segments else "\n\n".join(d.page_content for d in docs)
        else:
            context = "\n\n".join(d.page_content for d in docs)

        # Generate answer
        system_prompt = (
            "You are a precise financial analysis assistant. "
            "Answer questions accurately using the provided context. "
            "Be concise and precise with numbers."
        )

        user_prompt = f"""Answer the following question using the information from the provided context.

Context:
{context}

Question: {question}

Answer:"""

        response = self.llm_provider.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tokens=500,
            temperature=0.0,  # Deterministic for fair comparison
        )

        return response.content if response.content else "", retrieval_time

    def score_answer(self, predicted: str, gold: str) -> float:
        """Calculate semantic similarity between predicted and gold answer."""
        if not predicted:
            return 0.0
        return embedding_similarity(predicted, gold, self.embeddings)

    def process_question(
        self,
        question: str,
        gold_answer: str,
        question_id: str,
    ) -> Dict[str, Any]:
        """Process a single question with all pipeline configurations.

        This is the core of silver label generation - we run every config
        and record which one produces the best answer.

        Returns:
            Dictionary with all scores and best configuration
        """
        all_scores = {}
        all_answers = {}

        for config in PIPELINE_CONFIGS:
            key = config_key(config)
            pipeline = self.pipelines[key]

            try:
                answer, _ = self.generate_answer(question, pipeline, config)
                score = self.score_answer(answer, gold_answer)
                all_scores[key] = score
                all_answers[key] = answer
            except Exception as e:
                print(f"    Error with {key}: {str(e)[:50]}")
                all_scores[key] = 0.0
                all_answers[key] = f"ERROR: {str(e)}"

        # Find best configuration (silver label)
        best_key = max(all_scores, key=all_scores.get)
        best_config = self.pipeline_configs[best_key]

        return {
            # Silver label - the best configuration for this question
            "best_config": best_key,
            "best_config_details": {
                "pipeline_id": best_config["pipeline_id"],
                "top_k": best_config["top_k"],
                "reranker": best_config.get("reranker", "bge"),
                "use_rse": best_config.get("use_rse", False),
                "use_hyde": best_config.get("use_hyde", False),
            },
            # Top-level fields for training script compatibility
            "best_pipeline": best_config["pipeline_id"],
            "best_top_k": best_config["top_k"],
            "best_score": all_scores[best_key],
            "all_scores": all_scores,
            "best_answer": all_answers[best_key],
        }

    def generate_labels(
        self,
        output_path: str,
        resume: bool = False,
        limit: int = None,
    ) -> Dict[str, Any]:
        """Generate oracle labels for all questions.

        Args:
            output_path: Path to save results
            resume: If True, load existing results and continue
            limit: Maximum number of questions to process (None = all)

        Returns:
            Dictionary of all oracle labels
        """
        # Load dataset
        adapter = FinanceBenchAdapter()
        df = adapter.load_dataset()

        # Apply limit if specified
        if limit is not None:
            df = df.head(limit)

        print(f"Loaded {len(df)} questions from FinanceBench")

        # Load existing results if resuming
        labels = {}
        if resume and Path(output_path).exists():
            with open(output_path, "r") as f:
                data = json.load(f)
                labels = data.get("labels", {})
            print(f"Resuming from {len(labels)} existing labels")

        # Process each question
        print("\n" + "=" * 60)
        print("GENERATING ORACLE LABELS")
        print("=" * 60)
        print(f"Total questions: {len(df)}")
        print(f"Pipeline configs: {len(PIPELINE_CONFIGS)}")
        print(f"Total API calls: {len(df) * len(PIPELINE_CONFIGS)}")
        print("=" * 60 + "\n")

        start_time = time.time()
        question_col = adapter.get_question_column()
        answer_col = adapter.get_answer_column()
        type_col = adapter.get_question_type_column()

        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Questions"):
            question_id = str(row.get("financebench_id", idx))

            # Skip if already processed
            if question_id in labels:
                continue

            question = row[question_col]
            gold_answer = row[answer_col]
            question_type = row.get(type_col, "unknown")

            # Process question
            result = self.process_question(question, gold_answer, question_id)

            # Store result
            labels[question_id] = {
                "question": question,
                "question_type": question_type,
                "gold_answer": gold_answer,
                **result,
            }

            # Save intermediate results every 10 questions
            if len(labels) % 10 == 0:
                self._save_results(output_path, labels, start_time)

        # Final save
        self._save_results(output_path, labels, start_time)

        return labels

    def _save_results(
        self,
        output_path: str,
        labels: Dict[str, Any],
        start_time: float,
    ):
        """Save results to JSON file."""
        elapsed = time.time() - start_time

        # Calculate statistics
        pipeline_counts = {}
        for label in labels.values():
            key = f"{label['best_pipeline']}_k{label['best_top_k']}"
            pipeline_counts[key] = pipeline_counts.get(key, 0) + 1

        avg_score = (
            sum(l["best_score"] for l in labels.values()) / len(labels)
            if labels else 0
        )

        output = {
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "model": self.model_name,
                "chroma_path": self.chroma_path,
                "num_questions": len(labels),
                "num_configs": len(PIPELINE_CONFIGS),
                "elapsed_seconds": elapsed,
                "avg_best_score": avg_score,
            },
            "pipeline_distribution": pipeline_counts,
            "labels": labels,
        }

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate oracle labels for meta-learning router"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=DEFAULT_OUTPUT,
        help=f"Output path (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--chroma-path",
        type=str,
        default=DEFAULT_CHROMA_PATH,
        help=f"ChromaDB path (default: {DEFAULT_CHROMA_PATH})",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"LLM model (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--embedding",
        type=str,
        default="bge-large",
        help="Embedding model (default: bge-large, FREE local)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing output file",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of questions to process (for testing)",
    )
    args = parser.parse_args()

    # Resolve paths relative to project root
    project_root = Path(__file__).parent.parent
    output_path = str(project_root / args.output)
    chroma_path = str(project_root / args.chroma_path)

    print("=" * 60)
    print("ORACLE LABEL GENERATION")
    print("=" * 60)
    print(f"Output: {output_path}")
    print(f"ChromaDB: {chroma_path}")
    print(f"Model: {args.model}")
    print(f"Resume: {args.resume}")
    print(f"Limit: {args.limit if args.limit else 'all'}")
    print("=" * 60)

    # Validate ChromaDB path exists
    if not Path(chroma_path).exists():
        print(f"\nERROR: ChromaDB not found at {chroma_path}")
        print("Available directories:")
        for d in project_root.iterdir():
            if d.is_dir() and d.name.startswith("chroma"):
                print(f"  - {d.name}")
        sys.exit(1)

    # Generate labels
    generator = OracleLabelGenerator(
        chroma_path=chroma_path,
        model_name=args.model,
        embedding_model=args.embedding,
    )
    generator.initialize()
    labels = generator.generate_labels(output_path, resume=args.resume, limit=args.limit)

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total questions: {len(labels)}")

    # Distribution by pipeline
    pipeline_counts = {}
    type_scores = {}
    for label in labels.values():
        key = f"{label['best_pipeline']}_k{label['best_top_k']}"
        pipeline_counts[key] = pipeline_counts.get(key, 0) + 1

        qtype = label.get("question_type", "unknown")
        if qtype not in type_scores:
            type_scores[qtype] = []
        type_scores[qtype].append(label["best_score"])

    print("\nBest pipeline distribution:")
    for key, count in sorted(pipeline_counts.items(), key=lambda x: -x[1]):
        pct = count / len(labels) * 100
        print(f"  {key}: {count} ({pct:.1f}%)")

    print("\nAverage best score by question type:")
    for qtype, scores in type_scores.items():
        avg = sum(scores) / len(scores) if scores else 0
        print(f"  {qtype}: {avg:.3f} (n={len(scores)})")

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
