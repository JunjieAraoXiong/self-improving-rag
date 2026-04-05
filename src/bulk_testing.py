"""Bulk testing framework for RAG system evaluation."""

import sys
import time
import json
import argparse
import random
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm


def set_global_seed(seed: int) -> None:
    """Set global random seeds for reproducibility.

    Initializes random state for Python's random module and NumPy.
    This ensures reproducible results across runs when the same seed is used.

    Args:
        seed: Integer seed value for reproducibility
    """
    random.seed(seed)
    np.random.seed(seed)
    # Note: If using torch in the future, add torch.manual_seed(seed)
    print(f"  Global seed set to: {seed}")

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import retrieval components
from langchain_chroma import Chroma

# Import custom modules
from dataset_adapters import BaseDatasetAdapter, FinanceBenchAdapter
from evaluation.metrics import (
    embedding_similarity,
    calculate_aggregate_metrics,
    format_metrics_summary,
    bootstrap_ci,
    bootstrap_compare,
)
from evaluation.latex_tables import (
    generate_results_table,
    significance_marker,
)
from evaluation.llm_judge import llm_as_judge
from evaluation.numeric_check import numeric_match
from src.postprocessing.numeric_verify import verify_numeric_answer
from src.retrieval_tools.tool_registry import (
    build_pipeline,
    build_retriever_for_pipeline,
)
from src.retrieval_tools.router import build_routed_pipeline
from src.config import (
    DEFAULTS,
    PIPELINES,
    EMBEDDINGS,
    get_model_abbrev,
    get_provider_for_model,
    get_embedding_model,
)
from src.providers import get_provider

# Load environment variables
load_dotenv()


@dataclass
class BulkTestConfig:
    """Configuration for bulk testing runs."""

    # Dataset settings
    dataset_name: str

    # Retrieval policy
    pipeline_id: str = DEFAULTS.pipeline_id

    # Model settings
    model_name: str = DEFAULTS.llm_model
    embedding_model: str = DEFAULTS.embedding_model

    # Retrieval settings
    top_k_retrieval: int = DEFAULTS.top_k
    initial_k_factor: float = DEFAULTS.initial_k_factor

    # Reranker settings
    reranker_model: str = DEFAULTS.reranker_model

    # Generation settings
    temperature: float = DEFAULTS.temperature
    max_tokens: int = DEFAULTS.max_tokens

    # Evaluation settings
    use_llm_judge: bool = False
    judge_model: str = DEFAULTS.judge_model
    use_numeric_verify: bool = False

    # Paths
    chroma_path: str = DEFAULTS.chroma_path
    output_dir: str = DEFAULTS.output_dir

    # Router settings (for pipeline_id="routed")
    router_classifier_model: str = DEFAULTS.router_classifier_model
    router_hyde_model: str = DEFAULTS.router_hyde_model
    use_rule_router: bool = False  # Use free rule-based router instead of LLM
    domain: str = None  # Domain for route selection (finance, legal, medical)

    # RSE settings
    use_rse: bool = False  # Enable Relevant Segment Extraction

    # Agentic RAG settings
    use_agentic_retry: bool = False  # Enable multi-agent retry loop
    max_retries: int = 1  # Max retries when agentic mode is enabled
    retry_threshold: float = 0.5  # Score below which to trigger retry
    agent_log_dir: str = "agent_logs"  # Directory for agent decision logs
    blind_judge: bool = False  # If True, Judge uses self-evaluation (no gold answer)

    # Table reasoning settings (rLLM-FinQA integration)
    use_table_agent: bool = False  # Route numeric computation to TableAgent
    vllm_base_url: str = None  # vLLM server URL for rLLM-FinQA-4B model

    # Ablation study settings
    ablation: str = None  # Ablation mode to run
    ablation_no_retrieval_escalation: bool = False
    ablation_no_prompt_escalation: bool = False
    ablation_no_hyde: bool = False
    ablation_no_deterministic_verify: bool = False

    # Reproducibility settings
    seed: int = 42  # Global random seed for reproducibility
    run_id: int = 0  # Current run index (0-indexed) for multi-run experiments

    # Runtime metadata
    timestamp: str = None

    def __post_init__(self):
        """Generate timestamp if not provided and resolve paths relative to project root."""
        if self.timestamp is None:
            self.timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        if not self.pipeline_id:
            self.pipeline_id = DEFAULTS.pipeline_id

        # Resolve paths relative to project root
        base_dir = Path(__file__).parent.parent
        self.chroma_path = str(base_dir / self.chroma_path)
        self.output_dir = str(base_dir / self.output_dir)

    def get_model_abbrev(self) -> str:
        """Get abbreviated model name for filename."""
        return get_model_abbrev(self.model_name)

    def generate_filename(self, dataset_abbrev: str, include_run: bool = False) -> str:
        """Generate output filename from configuration.

        Args:
            dataset_abbrev: Short name for the dataset
            include_run: If True, include run_id in filename for multi-run experiments
        """
        model_abbrev = self.get_model_abbrev()
        temp_str = f"t{self.temperature}".replace(".", "")
        base = f"{self.timestamp}_{dataset_abbrev}_{model_abbrev}_k{self.top_k_retrieval}_{temp_str}"
        if include_run:
            return f"{base}_seed{self.seed}_run{self.run_id}.csv"
        return f"{base}.csv"


class BulkTestRunner:
    """Main bulk testing runner."""

    def __init__(self, config: BulkTestConfig):
        self.config = config
        self.retriever = None
        self.embeddings = None
        self.llm_provider = None
        self.pipeline = None

    def initialize_framework(self):
        """Initialize RAG framework components."""
        try:
            print("\nInitializing RAG framework...")

            # Initialize embeddings (uses FREE local model by default)
            emb_config = EMBEDDINGS.get(self.config.embedding_model)
            if emb_config:
                print(f"  Loading embeddings: {emb_config.model_id} ({emb_config.provider})")
            else:
                print(f"  Loading embeddings: {self.config.embedding_model}")
            self.embeddings = get_embedding_model(self.config.embedding_model)

            # Load ChromaDB
            print(f"  Loading ChromaDB from: {self.config.chroma_path}")
            db = Chroma(
                persist_directory=self.config.chroma_path,
                embedding_function=self.embeddings
            )

            print(f"  Creating retriever for pipeline: {self.config.pipeline_id}")

            if self.config.pipeline_id == "routed":
                # Use the question-type router for adaptive retrieval
                self.pipeline = build_routed_pipeline(
                    db=db,
                    embedding_fn=self.embeddings,
                    classifier_model=self.config.router_classifier_model,
                    hyde_model=self.config.router_hyde_model,
                    reranker_model=self.config.reranker_model,
                    use_rule_router=self.config.use_rule_router,
                    domain=self.config.domain,
                )
            else:
                # Standard pipeline
                retriever, set_k_fn, take_top_k_fn, use_hybrid = build_retriever_for_pipeline(
                    self.config.pipeline_id, db, top_k=self.config.top_k_retrieval
                )
                self.retriever = retriever
                self.pipeline = build_pipeline(
                    pipeline_id=self.config.pipeline_id,
                    retriever=retriever,
                    top_k=self.config.top_k_retrieval,
                    initial_k_factor=self.config.initial_k_factor,
                    set_k_fn=set_k_fn,
                    take_top_k_fn=take_top_k_fn,
                    reranker_model=self.config.reranker_model,
                    db=db,  # Pass db for pre-filtering
                    use_hybrid=use_hybrid,  # Pass flag for pre-filtering
                )

            # Initialize LLM provider
            provider_name = get_provider_for_model(self.config.model_name)
            print(f"  Initializing LLM: {self.config.model_name} (provider: {provider_name})")
            self.llm_provider = get_provider(self.config.model_name)

            print("Framework initialization complete!\n")
            return True

        except Exception as e:
            print(f"ERROR: Framework initialization failed: {str(e)}")
            return False

    def process_single_question(self, question: str, question_id: Any) -> Dict[str, Any]:
        """Process a single question through the RAG pipeline."""
        result = {
            'predicted_answer': None,
            'sources': None,
            'retrieval_time_ms': 0,
            'generation_time_ms': 0,
            'error': None
        }

        try:
            # Retrieval phase
            retrieval_start = time.time()

            # Use RSE for segment-level retrieval if enabled
            if self.config.use_rse and self.pipeline and hasattr(self.pipeline, 'retrieve_segments'):
                segments = self.pipeline.retrieve_segments(question)
                result['retrieval_time_ms'] = (time.time() - retrieval_start) * 1000

                if not segments:
                    result['error'] = "No relevant segments found (RSE)"
                    return result

                # RSE returns merged text segments, not Document objects
                context = "\n\n---\n\n".join(segments)
                result['sources'] = [{"source": "RSE segment", "segment_count": len(segments)}]
            else:
                # Standard document-level retrieval
                docs = self.pipeline.retrieve(question) if self.pipeline else []
                result['retrieval_time_ms'] = (time.time() - retrieval_start) * 1000

                if not docs:
                    result['error'] = "No relevant documents found"
                    return result

                # Extract context and sources
                context = "\n\n".join(d.page_content for d in docs)
                sources = [doc.metadata for doc in docs]
                result['sources'] = sources

            # Generation phase
            generation_start = time.time()

            system_prompt = (
                "You are a precise financial analysis assistant who approaches every question methodically. "
                "ALWAYS enter PLAN MODE before answering: first analyze what information is needed, "
                "identify relevant data points in the context, then formulate your answer. "
                "Be accurate with numbers, dates, and company names. "
                "ALWAYS provide your best answer based on the available context - "
                "never refuse to answer or say you cannot find the information."
            )

            user_prompt = f"""Answer the following question using the information from the provided context.

PLAN MODE REQUIRED - Before answering, you MUST:
1. IDENTIFY: What specific information does this question ask for? (number, explanation, comparison, etc.)
2. LOCATE: Find the relevant data points, figures, or facts in the context
3. VERIFY: Check that the data matches the correct company, time period, and fiscal year
4. CALCULATE: If math is needed, show your work step-by-step
5. ANSWER: Only then provide your final answer

IMPORTANT INSTRUCTIONS:
- ALWAYS provide an answer - even if the context seems incomplete, give your best hypothesis based on available information
- Use precise numbers, dates, and company names from the context when available
- Do NOT use information from other companies or fiscal years unless explicitly asked
- Pay close attention to fiscal years and time periods mentioned in both the question and context
- For numerical questions requiring a specific number, percentage, or ratio as the answer:
  * After your planning steps, provide ONLY the numerical value with appropriate units
  * Format examples: "$1,577 million" or "65.4%" or "24.26"
  * Do NOT add explanatory sentences like "The answer is..." or "According to the context..."
- For non-numerical or explanatory questions, provide full context and reasoning
- NEVER say "The provided context does not contain sufficient information" - always attempt an answer

Context:
{context}

Question: {question}

Plan and Answer:"""

            # Use the provider abstraction
            response = self.llm_provider.generate(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
            )

            result['generation_time_ms'] = (time.time() - generation_start) * 1000

            if response.content:
                result['predicted_answer'] = response.content

                # Numeric verification - check if numbers in answer exist in sources
                verification = verify_numeric_answer(response.content, docs)
                result['numeric_score'] = verification.score
                result['flagged_numbers'] = verification.flagged_numbers
            else:
                result['error'] = "Empty response from LLM"

        except Exception as e:
            result['error'] = f"Error: {str(e)}"

        return result

    def run_bulk_test(self, adapter: BaseDatasetAdapter) -> pd.DataFrame:
        """Run bulk test on a dataset."""
        print("\n" + "=" * 60)
        print("STARTING BULK TEST")
        print("=" * 60)

        # Load dataset
        try:
            df = adapter.load_dataset()
        except Exception as e:
            print(f"ERROR: Failed to load dataset: {str(e)}")
            sys.exit(1)

        # Get column names
        question_col = adapter.get_question_column()
        answer_col = adapter.get_answer_column()
        question_type_col = adapter.get_question_type_column()
        metadata_cols = adapter.get_metadata_columns()

        print(f"Dataset loaded: {len(df)} questions")
        print(f"Question column: {question_col}")
        print(f"Answer column: {answer_col}")

        # Initialize framework
        if not self.initialize_framework():
            print("ERROR: Framework initialization failed. Exiting.")
            sys.exit(1)

        # Prepare results storage
        results = []

        # Process questions with progress bar
        print("\nProcessing questions...")
        start_time = time.time()

        try:
            for idx, row in tqdm(df.iterrows(), total=len(df), desc="Questions"):
                question = row[question_col]
                gold_answer = row[answer_col]

                # Process question
                result = self.process_single_question(question, idx)

                # Calculate semantic similarity
                if result['predicted_answer'] is not None:
                    sem_sim = embedding_similarity(
                        result['predicted_answer'],
                        gold_answer,
                        self.embeddings
                    )
                else:
                    sem_sim = 0.0

                # LLM judge if enabled
                judge_score = None
                judge_justification = None
                if self.config.use_llm_judge and result['predicted_answer'] is not None:
                    judge_score, judge_justification = llm_as_judge(
                        question=question,
                        gold_answer=gold_answer,
                        predicted_answer=result['predicted_answer'],
                        judge_model=self.config.judge_model
                    )

                # Format sources
                sources_str = None
                if result['sources']:
                    source_names = [s.get('source', 'unknown') for s in result['sources']]
                    sources_str = "; ".join(source_names)

                # Build result row
                result_row = {
                    'question_id': idx,
                    'question': question,
                    'gold_answer': gold_answer,
                    'predicted_answer': result['predicted_answer'],
                    'semantic_similarity': sem_sim,
                    'numeric_score': result.get('numeric_score'),
                    'flagged_numbers': str(result.get('flagged_numbers', [])),
                    'retrieval_time_ms': result['retrieval_time_ms'],
                    'generation_time_ms': result['generation_time_ms'],
                    'sources': sources_str,
                    'error': result['error']
                }

                if self.config.use_llm_judge:
                    result_row['judge_score'] = judge_score
                    result_row['judge_justification'] = judge_justification

                if question_type_col and question_type_col in row:
                    result_row['question_type'] = row[question_type_col]

                for col in metadata_cols:
                    if col in row:
                        result_row[col] = row[col]

                results.append(result_row)

        except KeyboardInterrupt:
            print(f"\n\nInterrupted! Saving {len(results)} partial results...")
            if results:
                results_df = pd.DataFrame(results)
                self._save_results(results_df, adapter, partial=True)
            sys.exit(0)

        print(f"\nProcessing complete! Total time: {time.time() - start_time:.2f}s")
        return pd.DataFrame(results)

    def _save_results(
        self,
        results_df: pd.DataFrame,
        adapter: BaseDatasetAdapter,
        partial: bool = False,
        include_run: bool = False,
    ) -> Optional[Path]:
        """Save results to CSV and summary to JSON.

        Args:
            results_df: DataFrame with evaluation results
            adapter: Dataset adapter for naming
            partial: If True, mark as partial results (interrupted run)
            include_run: If True, include run_id in filename (for multi-run experiments)

        Returns:
            Path to the saved CSV file, or None on error
        """
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(exist_ok=True)

        filename = self.config.generate_filename(adapter.name, include_run=include_run)
        if partial:
            filename = filename.replace('.csv', '_PARTIAL.csv')

        output_path = output_dir / filename

        # Save CSV
        results_df.to_csv(output_path, index=False)
        print(f"\nResults saved to: {output_path}")

        # Calculate and save metrics
        metrics = calculate_aggregate_metrics(results_df)
        metrics['config'] = asdict(self.config)
        # Add seed info for reproducibility
        metrics['seed'] = self.config.seed
        metrics['run_id'] = self.config.run_id

        summary_path = output_path.with_suffix('.json')
        with open(summary_path, 'w') as f:
            json.dump(metrics, f, indent=2)

        print(f"Summary saved to: {summary_path}")
        print(format_metrics_summary(metrics))

        return output_path

    def save_results(self, results_df: pd.DataFrame, adapter: BaseDatasetAdapter):
        """Public method to save results."""
        return self._save_results(results_df, adapter, partial=False)

    def run_agentic_test(self, adapter: BaseDatasetAdapter) -> pd.DataFrame:
        """Run agentic RAG test with self-correcting retry loop.

        This mode uses multi-agent orchestration:
        1. RetrievalAgent: Decides retrieval strategy
        2. ReasoningAgent: Generates answers
        3. JudgeAgent: Evaluates and triggers retries
        """
        print("\n" + "=" * 60)
        print("STARTING AGENTIC RAG TEST")
        print(f"Max retries: {self.config.max_retries}")
        print(f"Retry threshold: {self.config.retry_threshold}")
        print("=" * 60)

        # Load dataset
        try:
            df = adapter.load_dataset()
        except Exception as e:
            print(f"ERROR: Failed to load dataset: {str(e)}")
            sys.exit(1)

        # Get column names
        question_col = adapter.get_question_column()
        answer_col = adapter.get_answer_column()
        question_type_col = adapter.get_question_type_column()
        metadata_cols = adapter.get_metadata_columns()

        print(f"Dataset loaded: {len(df)} questions")

        # Initialize embeddings and ChromaDB
        print("\nInitializing RAG framework...")
        emb_config = EMBEDDINGS.get(self.config.embedding_model)
        if emb_config:
            print(f"  Loading embeddings: {emb_config.model_id} ({emb_config.provider})")
        self.embeddings = get_embedding_model(self.config.embedding_model)

        print(f"  Loading ChromaDB from: {self.config.chroma_path}")
        from langchain_chroma import Chroma
        db = Chroma(
            persist_directory=self.config.chroma_path,
            embedding_function=self.embeddings
        )

        # Build agentic orchestrator
        from src.agents import AgenticRAGOrchestrator
        from src.agents.orchestrator import AgenticRAGConfig

        agentic_config = AgenticRAGConfig(
            max_retries=self.config.max_retries,
            retry_threshold=self.config.retry_threshold,
            blind_judge=self.config.blind_judge,
            llm_model=self.config.model_name,
            judge_model=self.config.judge_model,
            reranker_model=self.config.reranker_model,
            use_rule_router=self.config.use_rule_router,
            use_rse=self.config.use_rse,
            log_dir=self.config.agent_log_dir,
            enable_logging=True,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            # Table reasoning settings
            use_table_agent=self.config.use_table_agent,
            vllm_base_url=self.config.vllm_base_url,
            # Ablation study settings
            ablation_no_retrieval_escalation=self.config.ablation_no_retrieval_escalation,
            ablation_no_prompt_escalation=self.config.ablation_no_prompt_escalation,
            ablation_no_hyde=self.config.ablation_no_hyde,
            ablation_no_deterministic_verify=self.config.ablation_no_deterministic_verify,
        )

        orchestrator = AgenticRAGOrchestrator(
            config=agentic_config,
            db=db,
            embedding_fn=self.embeddings,
        )

        print("Agentic orchestrator initialized!\n")

        # Process questions
        results = []
        start_time = time.time()

        try:
            for idx, row in tqdm(df.iterrows(), total=len(df), desc="Agentic Processing"):
                question = row[question_col]
                gold_answer = row[answer_col]

                # Process through agentic pipeline
                agentic_result = orchestrator.process_question(
                    question=question,
                    gold_answer=gold_answer,
                    question_id=str(idx),
                )

                # Calculate semantic similarity
                if agentic_result.final_answer:
                    sem_sim = embedding_similarity(
                        agentic_result.final_answer,
                        gold_answer,
                        self.embeddings
                    )
                else:
                    sem_sim = 0.0

                # Calculate numeric accuracy (predicted vs gold)
                # This measures whether the predicted numbers match the gold answer
                # Different from verify_numeric_answer which checks hallucination
                numeric_result, numeric_explanation = numeric_match(
                    gold=gold_answer,
                    predicted=agentic_result.final_answer or ""
                )
                # Convert: True=1.0, False=0.0, None=skip (non-numeric question)
                if numeric_result is True:
                    numeric_accuracy = 1.0
                elif numeric_result is False:
                    numeric_accuracy = 0.0
                else:
                    numeric_accuracy = None  # Can't determine (no numbers in question)

                # Build result row
                result_row = {
                    'question_id': idx,
                    'question': question,
                    'gold_answer': gold_answer,
                    'predicted_answer': agentic_result.final_answer,
                    'semantic_similarity': sem_sim,
                    'judge_score': agentic_result.final_score,
                    'correct': agentic_result.correct,
                    'attempts': agentic_result.attempts,
                    'improvement_from_retry': agentic_result.improvement_from_retry,
                    'numeric_accuracy': numeric_accuracy,
                    'numeric_explanation': numeric_explanation,
                    'retrieval_time_ms': agentic_result.retrieval_time_ms,
                    'generation_time_ms': agentic_result.generation_time_ms,
                    'total_time_ms': agentic_result.total_time_ms,
                    'error': agentic_result.error,
                }

                # Add question type if available
                if question_type_col and question_type_col in row:
                    result_row['question_type'] = row[question_type_col]

                # Add metadata columns
                for col in metadata_cols:
                    if col in row:
                        result_row[col] = row[col]

                results.append(result_row)

        except KeyboardInterrupt:
            print(f"\n\nInterrupted! Saving {len(results)} partial results...")
            if results:
                results_df = pd.DataFrame(results)
                self._save_agentic_results(results_df, adapter, orchestrator, partial=True)
            sys.exit(0)

        print(f"\nProcessing complete! Total time: {time.time() - start_time:.2f}s")

        # Print summary
        orchestrator.print_summary()

        # Export decision logs
        log_paths = orchestrator.export_decisions()
        if log_paths:
            print(f"Decision logs saved to: {log_paths.get('json', 'N/A')}")

        return pd.DataFrame(results)

    def _save_agentic_results(
        self,
        results_df: pd.DataFrame,
        adapter: BaseDatasetAdapter,
        orchestrator,
        partial: bool = False
    ):
        """Save agentic results with additional statistics."""
        # Standard save
        self._save_results(results_df, adapter, partial)

        # Save additional agentic statistics
        output_dir = Path(self.config.output_dir)
        stats = orchestrator.get_statistics()

        stats_path = output_dir / f"{self.config.timestamp}_{adapter.name}_agentic_stats.json"
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"Agentic stats saved to: {stats_path}")

    def save_agentic_results(self, results_df: pd.DataFrame, adapter: BaseDatasetAdapter, orchestrator):
        """Public method to save agentic results."""
        self._save_agentic_results(results_df, adapter, orchestrator, partial=False)


def main():
    """Main entry point for bulk testing."""
    parser = argparse.ArgumentParser(description="Run bulk testing on RAG framework")

    parser.add_argument(
        '--dataset', type=str, default='financebench',
        help='Dataset to test on (default: financebench)'
    )
    parser.add_argument(
        '--pipeline', type=str, default=DEFAULTS.pipeline_id, choices=PIPELINES,
        help=f'Retrieval pipeline (default: {DEFAULTS.pipeline_id})'
    )
    parser.add_argument(
        '--model', type=str, default=DEFAULTS.llm_model,
        help=f'LLM model (default: {DEFAULTS.llm_model})'
    )
    parser.add_argument(
        '--top-k', type=int, default=DEFAULTS.top_k,
        help=f'Documents to retrieve (default: {DEFAULTS.top_k})'
    )
    parser.add_argument(
        '--initial-k-factor', type=float, default=DEFAULTS.initial_k_factor,
        help=f'Initial retrieval multiplier (default: {DEFAULTS.initial_k_factor})'
    )
    parser.add_argument(
        '--reranker', type=str, default=DEFAULTS.reranker_model,
        help=f'Reranker model (default: {DEFAULTS.reranker_model})'
    )
    parser.add_argument(
        '--temperature', type=float, default=DEFAULTS.temperature,
        help=f'Generation temperature (default: {DEFAULTS.temperature})'
    )
    parser.add_argument(
        '--max-tokens', type=int, default=DEFAULTS.max_tokens,
        help=f'Max tokens (default: {DEFAULTS.max_tokens})'
    )
    parser.add_argument(
        '--subset', type=str, default=None,
        help='Path to subset questions CSV'
    )
    parser.add_argument(
        '--split', type=str, default='dev',
        choices=['train', 'dev', 'test', 'validation'],
        help='Dataset split to use (for FinQA and similar HuggingFace datasets). Default: dev'
    )
    parser.add_argument(
        '--use-llm-judge', action='store_true',
        help='Enable LLM-as-a-Judge evaluation'
    )
    parser.add_argument(
        '--use-numeric-verify', action='store_true',
        help='Enable numeric verification to detect hallucinated numbers'
    )
    parser.add_argument(
        '--judge-model', type=str, default=DEFAULTS.judge_model,
        help=f'Judge model (default: {DEFAULTS.judge_model})'
    )
    parser.add_argument(
        '--embedding', type=str, default=DEFAULTS.embedding_model,
        help=f'Embedding model (default: {DEFAULTS.embedding_model}). Use "openai-large" for ChromaDB built with OpenAI embeddings.'
    )
    parser.add_argument(
        '--chroma-path', type=str, default=None,
        help='Path to ChromaDB directory (overrides default dataset-based path)'
    )
    parser.add_argument(
        '--router-classifier-model', type=str, default=DEFAULTS.router_classifier_model,
        help=f'Model for question classification in routed pipeline (default: {DEFAULTS.router_classifier_model})'
    )
    parser.add_argument(
        '--router-hyde-model', type=str, default=DEFAULTS.router_hyde_model,
        help=f'Model for HyDE generation in routed pipeline (default: {DEFAULTS.router_hyde_model})'
    )
    parser.add_argument(
        '--use-rule-router', action='store_true',
        help='Use free rule-based router instead of LLM classifier (instant, no API cost)'
    )
    parser.add_argument(
        '--domain', type=str, default=None,
        choices=['finance', 'legal', 'medical'],
        help='Domain for route selection. Affects reranking strategy (e.g., legal skips reranking per LegalBench-RAG findings)'
    )

    # RSE argument
    parser.add_argument(
        '--use-rse', action='store_true',
        help='Enable Relevant Segment Extraction (RSE) for table-heavy queries. Merges adjacent chunks into coherent segments.'
    )

    # Agentic RAG arguments
    parser.add_argument(
        '--use-agentic-retry', action='store_true',
        help='Enable agentic RAG with self-correcting retry loop'
    )
    parser.add_argument(
        '--max-retries', type=int, default=1,
        help='Maximum retry attempts when agentic mode is enabled (default: 1)'
    )
    parser.add_argument(
        '--retry-threshold', type=float, default=0.5,
        help='Judge score below which to trigger retry (default: 0.5)'
    )
    parser.add_argument(
        '--agent-log-dir', type=str, default='agent_logs',
        help='Directory for agent decision logs (default: agent_logs)'
    )
    parser.add_argument(
        '--blind-judge', action='store_true',
        help='Use blind judge mode: Judge evaluates without seeing gold answer (for realistic TPR/FPR)'
    )

    # Table reasoning (rLLM-FinQA) arguments
    parser.add_argument(
        '--use-table-agent', action='store_true',
        help='Enable TableAgent for numeric computation questions (requires rLLM install)'
    )
    parser.add_argument(
        '--vllm-base-url', type=str, default=None,
        help='vLLM server URL for rLLM-FinQA-4B model (e.g. http://localhost:30000/v1)'
    )

    # Ablation study arguments
    parser.add_argument(
        '--ablation', type=str, default=None,
        choices=[
            'no_retrieval_escalation',
            'no_prompt_escalation',
            'no_hyde',
            'no_deterministic_verify',
            'no_judge',  # Equivalent to max_retries=0
        ],
        help='Run ablation study by disabling specific components'
    )

    # Reproducibility arguments (ICLR requirements)
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Global random seed for reproducibility (default: 42)'
    )
    parser.add_argument(
        '--num-runs', type=int, default=1,
        help='Number of independent runs with different seeds (default: 1). Seeds will be base_seed, base_seed+1, ...'
    )

    args = parser.parse_args()

    # Create configuration
    config = BulkTestConfig(
        dataset_name=args.dataset,
        pipeline_id=args.pipeline,
        model_name=args.model,
        embedding_model=args.embedding,
        top_k_retrieval=args.top_k,
        initial_k_factor=args.initial_k_factor,
        reranker_model=args.reranker,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        use_llm_judge=args.use_llm_judge,
        judge_model=args.judge_model,
        router_classifier_model=args.router_classifier_model,
        router_hyde_model=args.router_hyde_model,
        use_rule_router=args.use_rule_router,
        domain=args.domain,
        # RSE settings
        use_rse=args.use_rse,
        # Agentic settings
        use_agentic_retry=args.use_agentic_retry,
        max_retries=args.max_retries,
        retry_threshold=args.retry_threshold,
        agent_log_dir=args.agent_log_dir,
        blind_judge=args.blind_judge,
        # Table reasoning settings
        use_table_agent=args.use_table_agent,
        vllm_base_url=args.vllm_base_url,
        # Ablation settings
        ablation=args.ablation,
        # Reproducibility settings
        seed=args.seed,
    )

    # Handle ablation mode - set appropriate flags
    if args.ablation:
        print(f"\n*** ABLATION MODE: {args.ablation} ***")
        config.use_agentic_retry = True  # Force agentic mode for ablations

        if args.ablation == 'no_retrieval_escalation':
            config.ablation_no_retrieval_escalation = True
        elif args.ablation == 'no_prompt_escalation':
            config.ablation_no_prompt_escalation = True
        elif args.ablation == 'no_hyde':
            config.ablation_no_hyde = True
        elif args.ablation == 'no_deterministic_verify':
            config.ablation_no_deterministic_verify = True
        elif args.ablation == 'no_judge':
            # No judge = no retries (single-pass equivalent)
            config.max_retries = 0

    # Handle chroma path - explicit override takes precedence
    if args.chroma_path:
        config.chroma_path = str(Path(__file__).parent.parent / args.chroma_path)
    else:
        # Auto-adjust chroma path for known datasets
        ds = args.dataset.lower()
        if config.chroma_path.endswith(DEFAULTS.chroma_path):
            if ds == 'financebench':
                config.chroma_path = str(Path(__file__).parent.parent / "chroma_docling")

    # Auto-detect domain from dataset if not explicitly specified
    if config.domain is None:
        ds = args.dataset.lower()
        dataset_to_domain = {
            'financebench': 'finance',
        }
        config.domain = dataset_to_domain.get(ds)
        if config.domain:
            print(f"Auto-detected domain '{config.domain}' from dataset '{ds}'")

    # Select dataset adapter
    ds = args.dataset.lower()
    if ds == 'financebench':
        adapter = FinanceBenchAdapter(subset_csv=args.subset)
    else:
        print(f"ERROR: Unknown dataset '{args.dataset}'")
        print("Available datasets: financebench")
        sys.exit(1)

    # Multi-run support for ICLR-class reproducibility
    num_runs = args.num_runs
    base_seed = args.seed
    all_run_results = []  # Store DataFrames from each run
    all_run_paths = []  # Store paths to individual run files

    print("\n" + "=" * 60)
    print(f"EXPERIMENT CONFIGURATION")
    print("=" * 60)
    print(f"  Dataset:    {args.dataset}")
    print(f"  Pipeline:   {config.pipeline_id}")
    print(f"  Model:      {config.model_name}")
    print(f"  Base seed:  {base_seed}")
    print(f"  Num runs:   {num_runs}")
    if config.use_agentic_retry:
        print(f"  Mode:       Agentic RAG (max_retries={config.max_retries})")
        if config.use_table_agent:
            print(f"  TableAgent: ON (vllm={config.vllm_base_url or 'LLM fallback'})")
    else:
        print(f"  Mode:       Standard RAG")
    print("=" * 60)

    # Write experiment manifest for reproducibility
    manifest_dir = Path(config.output_dir)
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "timestamp": config.timestamp,
        "dataset": args.dataset,
        "pipeline": config.pipeline_id,
        "model": config.model_name,
        "judge_model": config.judge_model,
        "seed": base_seed,
        "num_runs": num_runs,
        "use_agentic_retry": config.use_agentic_retry,
        "max_retries": config.max_retries,
        "use_table_agent": config.use_table_agent,
        "vllm_base_url": config.vllm_base_url,
        "blind_judge": config.blind_judge,
        "top_k": config.top_k_retrieval,
        "temperature": config.temperature,
        "embedding_model": config.embedding_model,
        "reranker_model": config.reranker_model,
        "ablation": config.ablation,
    }
    manifest_path = manifest_dir / f"manifest_{config.timestamp}.json"
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    print(f"  Manifest:   {manifest_path}")

    for run_idx in range(num_runs):
        current_seed = base_seed + run_idx
        config.seed = current_seed
        config.run_id = run_idx

        print(f"\n{'='*60}")
        print(f"RUN {run_idx + 1}/{num_runs} (seed={current_seed})")
        print(f"{'='*60}")

        # Set global seed for this run
        set_global_seed(current_seed)

        # Create runner for this run
        runner = BulkTestRunner(config)

        try:
            if config.use_agentic_retry:
                # Run agentic RAG with retry loop
                results_df = runner.run_agentic_test(adapter)
            else:
                # Standard RAG (no retry)
                results_df = runner.run_bulk_test(adapter)

            # Add run metadata
            results_df['run_id'] = run_idx
            results_df['seed'] = current_seed

            all_run_results.append(results_df)

            # Save individual run results
            if num_runs > 1:
                run_path = runner._save_results(results_df, adapter, partial=False, include_run=True)
                if run_path:
                    all_run_paths.append(run_path)
            else:
                # Single run: save normally
                runner.save_results(results_df, adapter)

        except Exception as e:
            print(f"\nERROR in run {run_idx + 1}: {str(e)}")
            import traceback
            traceback.print_exc()
            if not all_run_results:
                sys.exit(1)
            print(f"Continuing with {len(all_run_results)} completed runs...")

    # Aggregate results across runs if multiple runs
    if num_runs > 1 and len(all_run_results) > 1:
        print("\n" + "=" * 60)
        print("AGGREGATING RESULTS ACROSS RUNS")
        print("=" * 60)

        aggregate_and_save_results(
            all_run_results=all_run_results,
            config=config,
            adapter=adapter,
            all_run_paths=all_run_paths,
        )


def aggregate_and_save_results(
    all_run_results: List[pd.DataFrame],
    config: BulkTestConfig,
    adapter,
    all_run_paths: List[Path],
) -> None:
    """Aggregate results across multiple runs and save summary.

    Computes mean ± std across runs for all metrics, and saves:
    1. Combined CSV with all runs
    2. Aggregated JSON summary with cross-run statistics
    3. LaTeX-ready table snippet
    """
    # Combine all runs
    combined_df = pd.concat(all_run_results, ignore_index=True)

    # Save combined CSV
    output_dir = Path(config.output_dir)
    output_dir.mkdir(exist_ok=True)
    combined_path = output_dir / f"{config.timestamp}_{adapter.name}_combined_{len(all_run_results)}runs.csv"
    combined_df.to_csv(combined_path, index=False)
    print(f"\nCombined results saved to: {combined_path}")

    # Compute per-question aggregates (mean across runs for same question_id)
    question_col = 'question_id'
    if question_col not in combined_df.columns:
        question_col = combined_df.columns[0]  # Fallback to first column

    # Metrics to aggregate
    metric_cols = ['semantic_similarity', 'judge_score', 'numeric_accuracy']
    metric_cols = [c for c in metric_cols if c in combined_df.columns]

    # Compute cross-run statistics
    cross_run_stats = {
        'num_runs': len(all_run_results),
        'seeds': [config.seed - config.run_id + i for i in range(len(all_run_results))],
        'run_files': [str(p) for p in all_run_paths],
    }

    # Per-run means (for computing std across runs)
    run_means = {}
    for col in metric_cols:
        run_means[col] = []
        for run_df in all_run_results:
            if col in run_df.columns:
                run_means[col].append(float(run_df[col].dropna().mean()))

    # Aggregate statistics
    for col in metric_cols:
        if run_means.get(col):
            means = run_means[col]
            cross_run_stats[col] = {
                'mean_across_runs': float(np.mean(means)),
                'std_across_runs': float(np.std(means)),
                'per_run_means': means,
            }
            # Bootstrap CI on run means
            if len(means) >= 2:
                mean, ci_lower, ci_upper = bootstrap_ci(means, n_bootstrap=1000)
                cross_run_stats[col]['ci_95'] = [ci_lower, ci_upper]

    # Save aggregated summary
    summary_path = output_dir / f"{config.timestamp}_{adapter.name}_aggregated_{len(all_run_results)}runs.json"
    summary = {
        'config': asdict(config),
        'cross_run_stats': cross_run_stats,
    }
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Aggregated summary saved to: {summary_path}")

    # Print cross-run summary
    print("\n" + "=" * 60)
    print("CROSS-RUN SUMMARY")
    print("=" * 60)
    for col in metric_cols:
        if col in cross_run_stats:
            stats = cross_run_stats[col]
            mean = stats['mean_across_runs']
            std = stats['std_across_runs']
            print(f"\n{col}:")
            print(f"  Mean ± Std: {mean:.4f} ± {std:.4f}")
            if 'ci_95' in stats:
                ci = stats['ci_95']
                print(f"  95% CI:     [{ci[0]:.4f}, {ci[1]:.4f}]")
            print(f"  Per-run:    {[f'{m:.4f}' for m in stats['per_run_means']]}")
    print("\n" + "=" * 60)

    # Generate LaTeX table row
    print("\n" + "=" * 60)
    print("LATEX TABLE ROW (copy to paper)")
    print("=" * 60)

    pipeline_name = config.pipeline_id
    if config.use_agentic_retry:
        pipeline_name = f"SC-RAG (B={config.max_retries})"

    row_parts = [pipeline_name]
    for col in metric_cols:
        if col in cross_run_stats:
            stats = cross_run_stats[col]
            mean = stats['mean_across_runs']
            std = stats['std_across_runs']
            row_parts.append(f"{mean:.2f} $\\pm$ {std:.2f}")
        else:
            row_parts.append("--")

    latex_row = " & ".join(row_parts) + r" \\"
    print(latex_row)
    print("=" * 60)

    # Save LaTeX snippet
    latex_path = output_dir / f"{config.timestamp}_{adapter.name}_latex_row.tex"
    with open(latex_path, 'w') as f:
        f.write(f"% Pipeline: {pipeline_name}\n")
        f.write(f"% Runs: {len(all_run_results)}, Seeds: {cross_run_stats['seeds']}\n")
        f.write(latex_row + "\n")
    print(f"LaTeX row saved to: {latex_path}")

    # Update manifest with results
    manifest["results"] = {
        "num_runs_completed": len(all_run_results),
        "cross_run_mean": cross_run_stats.get("mean"),
        "cross_run_std": cross_run_stats.get("std"),
    }
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    print(f"Manifest updated: {manifest_path}")


if __name__ == "__main__":
    main()
