"""Bulk testing framework for RAG system evaluation."""

import sys
import time
import json
import argparse
import random
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm


def set_global_seed(seed: int) -> None:
    """Seed only the local Python and NumPy random number generators.

    Hosted model sampling is controlled separately, when the selected provider
    supports a request seed. Identical local seeds do not make remote responses
    deterministic or independent.

    Args:
        seed: Integer seed for local Python and NumPy operations
    """
    random.seed(seed)
    np.random.seed(seed)
    # Note: If using torch in the future, add torch.manual_seed(seed)
    print(f"  Local Python/NumPy seed set to: {seed}")

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import retrieval components
from langchain_chroma import Chroma

# Import custom modules
from dataset_adapters import BaseDatasetAdapter, FinanceBenchAdapter
from evaluation.metrics import (
    _coerce_binary_series,
    _error_mask,
    embedding_similarity,
    calculate_aggregate_metrics,
    format_metrics_summary,
    bootstrap_ci,
)
from evaluation.outcome_evaluator import (
    DEFAULT_JUDGE_CORRECTNESS_THRESHOLD,
    evaluate_post_selection,
)
from src.postprocessing.numeric_verify import verify_numeric_answer
from src.agents.orchestrator import _summarize_usage_records
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
from src.providers.base import get_usage_tracker

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
    outcome_judge_model: str = DEFAULTS.judge_model
    outcome_judge_threshold: float = DEFAULT_JUDGE_CORRECTNESS_THRESHOLD
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
    policy_mode: str = "gap_driven_v2"  # or paper_fixed for paper reproduction

    # Ablation study settings
    ablation: str = None  # Ablation mode to run
    ablation_no_retrieval_escalation: bool = False
    ablation_no_prompt_escalation: bool = False
    ablation_no_untyped_citation_gate: bool = False

    # Reproducibility settings. `seed` is retained for config compatibility,
    # but controls only Python/NumPy; it is not a hosted-generation guarantee.
    seed: int = 42
    generation_seed: Optional[int] = None  # Best-effort provider request seed
    run_id: int = 0  # Current run index (0-indexed) for multi-run experiments

    # Runtime metadata
    timestamp: str = None

    def __post_init__(self):
        """Generate timestamp if not provided and resolve paths relative to project root."""
        if self.timestamp is None:
            self.timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        if not self.pipeline_id:
            self.pipeline_id = DEFAULTS.pipeline_id
        if not 0.0 <= self.outcome_judge_threshold <= 1.0:
            raise ValueError("outcome_judge_threshold must be between 0 and 1")

        # Resolve paths relative to project root
        base_dir = Path(__file__).parent.parent
        self.chroma_path = str(base_dir / self.chroma_path)
        self.output_dir = str(base_dir / self.output_dir)

    def get_model_abbrev(self) -> str:
        """Get abbreviated model name for filename."""
        return get_model_abbrev(self.model_name)

    def reproducibility_metadata(self) -> Dict[str, Any]:
        """Describe exactly which randomness controls were requested."""

        return {
            "local_seed": self.seed,
            "local_seed_scope": "python_random_and_numpy_only",
            "generation_seed_requested": self.generation_seed,
            "generation_seed_scope": "provider_best_effort",
            "hosted_generation_deterministic": False,
            "requested_model": self.model_name,
            "provider": get_provider_for_model(self.model_name),
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

    def generate_filename(self, dataset_abbrev: str, include_run: bool = False) -> str:
        """Generate output filename from configuration.

        Args:
            dataset_abbrev: Short name for the dataset
            include_run: If True, include run_id in filename for multi-run experiments
        """
        model_abbrev = self.get_model_abbrev()
        temp_str = f"t{self.temperature}".replace(".", "")
        base = f"{self.timestamp}_{dataset_abbrev}_{model_abbrev}_k{self.top_k_retrieval}_{temp_str}"
        if self.use_agentic_retry:
            mode = self.policy_mode.replace("_", "-")
            base += (
                f"_agentic-{mode}_retries{self.max_retries}"
                f"_seed{self.seed}_run{self.run_id}"
            )
        if include_run and not self.use_agentic_retry:
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
        self.agentic_orchestrator = None

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
            'error': None,
            'prompt_tokens': 0,
            'completion_tokens': 0,
            'llm_calls': 0,
            'usage_by_model': {},
            'estimated_cost_usd': 0.0,
        }
        tracker = get_usage_tracker()
        usage_cursor = tracker.cursor()

        try:
            # Retrieval phase
            retrieval_start = time.time()
            docs = []  # Initialize so verify_numeric_answer has a fallback

            # Use RSE for segment-level retrieval if enabled
            if (
                self.config.use_rse
                and self.pipeline
                and hasattr(self.pipeline, 'retrieve_segment_documents')
            ):
                docs = self.pipeline.retrieve_segment_documents(question)
                result['retrieval_time_ms'] = (time.time() - retrieval_start) * 1000

                if not docs:
                    result['error'] = "No relevant segments found (RSE)"
                    return result

                context = "\n\n---\n\n".join(doc.page_content for doc in docs)
                result['sources'] = [doc.metadata for doc in docs]
            elif self.config.use_rse and self.pipeline and hasattr(self.pipeline, 'retrieve_segments'):
                segments = self.pipeline.retrieve_segments(question)
                result['retrieval_time_ms'] = (time.time() - retrieval_start) * 1000
                if not segments:
                    result['error'] = "No relevant segments found (RSE)"
                    return result
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
                seed=self.config.generation_seed,
            )

            result['generation_time_ms'] = (time.time() - generation_start) * 1000

            if response.content:
                result['predicted_answer'] = response.content

                # Numeric verification - check if numbers in answer exist in sources
                if docs and self.config.use_numeric_verify:
                    verification = verify_numeric_answer(response.content, docs)
                    result['numeric_score'] = verification.score
                    result['flagged_numbers'] = verification.flagged_numbers
            else:
                result['error'] = "Empty response from LLM"

        except Exception as e:
            result['error'] = f"Error: {str(e)}"
        finally:
            usage = _summarize_usage_records(
                tracker.records_since(usage_cursor),
                fallback_model=self.config.model_name,
            )
            result['prompt_tokens'] = usage['prompt_tokens']
            result['completion_tokens'] = usage['completion_tokens']
            result['llm_calls'] = usage['calls']
            result['usage_by_model'] = usage['by_model']
            result['estimated_cost_usd'] = usage['estimated_cost_usd']

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

                # The same evaluator is used for single-pass and agentic
                # methods, and it runs only after the final answer is frozen.
                evaluator_tracker = get_usage_tracker()
                evaluator_cursor = evaluator_tracker.cursor()
                outcome = evaluate_post_selection(
                    question=question,
                    gold_answer=gold_answer,
                    predicted_answer=result['predicted_answer'],
                    use_llm_judge=self.config.use_llm_judge,
                    judge_model=self.config.outcome_judge_model,
                    judge_threshold=self.config.outcome_judge_threshold,
                    terminal_error=result['error'],
                )
                evaluator_usage = _summarize_usage_records(
                    evaluator_tracker.records_since(evaluator_cursor),
                    fallback_model=self.config.outcome_judge_model,
                )
                sem_sim = embedding_similarity(
                    result['predicted_answer'] or "",
                    gold_answer,
                    self.embeddings,
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
                    'judge_score': outcome.judge_score,
                    'judge_justification': outcome.judge_justification,
                    'outcome_judge_score': outcome.judge_score,
                    'outcome_judge_threshold': (
                        outcome.judge_correctness_threshold
                    ),
                    'outcome_evaluated': outcome.evaluated,
                    'outcome_evaluation_error': (
                        outcome.judge_justification
                        if outcome.mode == 'evaluator_error'
                        else None
                    ),
                    'evaluation_prompt_tokens': evaluator_usage['prompt_tokens'],
                    'evaluation_completion_tokens': (
                        evaluator_usage['completion_tokens']
                    ),
                    'evaluation_llm_calls': evaluator_usage['calls'],
                    'evaluation_estimated_cost_usd': (
                        evaluator_usage['estimated_cost_usd']
                    ),
                    'evaluation_usage_by_model': json.dumps(
                        evaluator_usage['by_model'], sort_keys=True
                    ),
                    'correct': outcome.correct,
                    'evaluation_mode': outcome.mode,
                    'exact_match': outcome.exact_match,
                    'numeric_accuracy': (
                        float(outcome.numeric_correct)
                        if outcome.numeric_correct is not None
                        else None
                    ),
                    'numeric_explanation': outcome.numeric_explanation,
                    'policy_score': None,
                    'policy_accepted': None,
                    'abstained': False,
                    'numeric_score': result.get('numeric_score'),
                    'flagged_numbers': str(result.get('flagged_numbers', [])),
                    'retrieval_time_ms': result['retrieval_time_ms'],
                    'generation_time_ms': result['generation_time_ms'],
                    'prompt_tokens': result['prompt_tokens'],
                    'completion_tokens': result['completion_tokens'],
                    'llm_calls': result['llm_calls'],
                    'usage_by_model': json.dumps(
                        result['usage_by_model'], sort_keys=True
                    ),
                    'cost_usd': result['estimated_cost_usd'],
                    'estimated_cost_usd': result['estimated_cost_usd'],
                    'sources': sources_str,
                    'error': result['error']
                }

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

        selection_metadata = adapter.get_selection_metadata()
        for key, value in selection_metadata.items():
            if value is None or isinstance(value, (str, int, float, bool)):
                results_df[f'dataset_{key}'] = value

        # Save CSV
        results_df.to_csv(output_path, index=False)
        print(f"\nResults saved to: {output_path}")

        # Calculate and save metrics
        metrics = calculate_aggregate_metrics(results_df)
        metrics['config'] = asdict(self.config)
        # `seed` remains for backward-compatible consumers; its scope is made
        # explicit in the structured reproducibility metadata.
        metrics['seed'] = self.config.seed
        metrics['local_seed'] = self.config.seed
        metrics['generation_seed'] = self.config.generation_seed
        metrics['run_id'] = self.config.run_id
        metrics['reproducibility'] = self.config.reproducibility_metadata()
        metrics['dataset_selection'] = selection_metadata

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
            policy_mode=self.config.policy_mode,
            llm_model=self.config.model_name,
            judge_model=self.config.judge_model,
            reranker_model=self.config.reranker_model,
            pipeline_id=self.config.pipeline_id,
            top_k=self.config.top_k_retrieval,
            initial_k_factor=self.config.initial_k_factor,
            use_rule_router=self.config.use_rule_router,
            use_rse=self.config.use_rse,
            log_dir=self.config.agent_log_dir,
            enable_logging=True,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            generation_seed=self.config.generation_seed,
            # Ablation study settings
            ablation_no_retrieval_escalation=self.config.ablation_no_retrieval_escalation,
            ablation_no_prompt_escalation=self.config.ablation_no_prompt_escalation,
            ablation_no_untyped_citation_gate=(
                self.config.ablation_no_untyped_citation_gate
            ),
        )

        orchestrator = AgenticRAGOrchestrator(
            config=agentic_config,
            db=db,
            embedding_fn=self.embeddings,
        )
        self.agentic_orchestrator = orchestrator

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

                evaluator_tracker = get_usage_tracker()
                evaluator_cursor = evaluator_tracker.cursor()
                outcome = evaluate_post_selection(
                    question=question,
                    gold_answer=gold_answer,
                    predicted_answer=agentic_result.final_answer,
                    use_llm_judge=self.config.use_llm_judge,
                    judge_model=self.config.outcome_judge_model,
                    judge_threshold=self.config.outcome_judge_threshold,
                    terminal_error=agentic_result.error,
                    abstained=agentic_result.abstained,
                )
                evaluator_usage = _summarize_usage_records(
                    evaluator_tracker.records_since(evaluator_cursor),
                    fallback_model=self.config.outcome_judge_model,
                )
                sem_sim = embedding_similarity(
                    agentic_result.final_answer or "",
                    gold_answer,
                    self.embeddings,
                )

                # Build result row
                result_row = {
                    'question_id': idx,
                    'question': question,
                    'gold_answer': gold_answer,
                    'predicted_answer': agentic_result.final_answer,
                    'semantic_similarity': sem_sim,
                    # `judge_score` is the shared post-selection evaluator for
                    # compatibility; controller utility remains policy_score.
                    'judge_score': outcome.judge_score,
                    'judge_justification': outcome.judge_justification,
                    'outcome_judge_score': outcome.judge_score,
                    'outcome_judge_threshold': (
                        outcome.judge_correctness_threshold
                    ),
                    'outcome_evaluated': outcome.evaluated,
                    'outcome_evaluation_error': (
                        outcome.judge_justification
                        if outcome.mode == 'evaluator_error'
                        else None
                    ),
                    'evaluation_prompt_tokens': evaluator_usage['prompt_tokens'],
                    'evaluation_completion_tokens': (
                        evaluator_usage['completion_tokens']
                    ),
                    'evaluation_llm_calls': evaluator_usage['calls'],
                    'evaluation_estimated_cost_usd': (
                        evaluator_usage['estimated_cost_usd']
                    ),
                    'evaluation_usage_by_model': json.dumps(
                        evaluator_usage['by_model'], sort_keys=True
                    ),
                    'policy_score': agentic_result.final_score,
                    'policy_accepted': agentic_result.policy_accepted,
                    'evaluation_mode': outcome.mode,
                    'policy_evaluation_mode': agentic_result.evaluation_mode,
                    'abstained': agentic_result.abstained,
                    'policy_mode': self.config.policy_mode,
                    'query_plan': json.dumps(
                        agentic_result.query_plan, sort_keys=True
                    ),
                    'correction_history': json.dumps(
                        agentic_result.correction_history, sort_keys=True
                    ),
                    'evidence_manifest': json.dumps(
                        agentic_result.evidence_manifest, sort_keys=True
                    ),
                    'retrieved_document_count': len(
                        agentic_result.evidence_manifest
                    ),
                    'sources': json.dumps(
                        list(agentic_result.evidence_manifest.values()),
                        sort_keys=True,
                    ),
                    'finance_question_spec': json.dumps(
                        agentic_result.finance_question_spec, sort_keys=True
                    ),
                    'finance_program': json.dumps(
                        agentic_result.finance_program, sort_keys=True
                    ),
                    'finance_verification': json.dumps(
                        agentic_result.finance_verification, sort_keys=True
                    ),
                    'correct': outcome.correct,
                    'policy_oracle_correct': agentic_result.correct,
                    'exact_match': outcome.exact_match,
                    'attempts': agentic_result.attempts,
                    'improvement_from_retry': agentic_result.improvement_from_retry,
                    'policy_score_improved': agentic_result.improvement_from_retry,
                    'numeric_accuracy': (
                        float(outcome.numeric_correct)
                        if outcome.numeric_correct is not None
                        else None
                    ),
                    'numeric_explanation': outcome.numeric_explanation,
                    'retrieval_time_ms': agentic_result.retrieval_time_ms,
                    'generation_time_ms': agentic_result.generation_time_ms,
                    'total_time_ms': agentic_result.total_time_ms,
                    'cost_usd': agentic_result.cost_usd,
                    'estimated_cost_usd': agentic_result.cost_usd,
                    'prompt_tokens': agentic_result.prompt_tokens,
                    'completion_tokens': agentic_result.completion_tokens,
                    'llm_calls': agentic_result.llm_calls,
                    'usage_by_model': json.dumps(
                        agentic_result.usage_by_model, sort_keys=True
                    ),
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
        output_path = self._save_results(results_df, adapter, partial)

        # Save additional agentic statistics
        output_dir = Path(self.config.output_dir)
        stats = orchestrator.get_statistics()

        mode = self.config.policy_mode.replace("_", "-")
        stats_path = output_dir / (
            f"{self.config.timestamp}_{adapter.name}_agentic-{mode}"
            f"_retries{self.config.max_retries}_seed{self.config.seed}"
            f"_run{self.config.run_id}_stats.json"
        )
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"Agentic stats saved to: {stats_path}")
        return output_path

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
        '--use-llm-judge', action='store_true',
        help='Enable a gold-based evaluator only after final answer selection'
    )
    parser.add_argument(
        '--use-numeric-verify', action='store_true',
        help=(
            'Enable source-grounding numeric verification for the standard '
            'single-pass path; agentic v2 verification is mandatory'
        )
    )
    parser.add_argument(
        '--judge-model', type=str, default=DEFAULTS.judge_model,
        help=f'Agentic policy Judge model (default: {DEFAULTS.judge_model})'
    )
    parser.add_argument(
        '--outcome-judge-model', type=str, default=DEFAULTS.judge_model,
        help=(
            'Post-selection gold-based evaluator model, separate from the '
            f'policy Judge (default: {DEFAULTS.judge_model})'
        )
    )
    parser.add_argument(
        '--outcome-judge-threshold',
        type=float,
        default=DEFAULT_JUDGE_CORRECTNESS_THRESHOLD,
        help=(
            'Minimum post-selection evaluator score labeled fully correct '
            f'(default: {DEFAULT_JUDGE_CORRECTNESS_THRESHOLD})'
        ),
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
        help=(
            'Hide gold answers from the policy Judge; correctness still requires '
            'independent post-selection evaluation'
        )
    )
    parser.add_argument(
        '--policy-mode', type=str, default='gap_driven_v2',
        choices=['gap_driven_v2', 'paper_fixed'],
        help=(
            'Correction policy: evidence-gap-directed v2 (default) or the '
            'historical fixed 10/20/30 retry schedule'
        ),
    )

    # Ablation study arguments
    parser.add_argument(
        '--ablation', type=str, default=None,
        choices=[
            'no_retrieval_escalation',
            'no_prompt_escalation',
            'no_untyped_citation_gate',
        ],
        help=(
            'Run a supported component ablation. Typed calculation verification '
            'is never disabled; use non-agentic mode for a single-pass baseline.'
        )
    )

    # Reproducibility arguments (ICLR requirements)
    parser.add_argument(
        '--seed', type=int, default=42,
        help=(
            'Local Python/NumPy seed (default: 42). This does not control '
            'hosted model sampling.'
        )
    )
    parser.add_argument(
        '--generation-seed', type=int, default=None,
        help=(
            'Best-effort hosted-generation request seed. Applied only by '
            'providers that support it; support and response metadata are logged.'
        )
    )
    parser.add_argument(
        '--num-runs', type=int, default=1,
        help=(
            'Number of repeated runs (default: 1). Local seeds are incremented; '
            'this does not guarantee independent hosted generations.'
        )
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
        outcome_judge_model=args.outcome_judge_model,
        outcome_judge_threshold=args.outcome_judge_threshold,
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
        policy_mode=args.policy_mode,
        use_numeric_verify=args.use_numeric_verify,
        # Ablation settings
        ablation=args.ablation,
        # Reproducibility settings
        seed=args.seed,
        generation_seed=args.generation_seed,
    )

    # Handle ablation mode - set appropriate flags
    if args.ablation:
        print(f"\n*** ABLATION MODE: {args.ablation} ***")
        config.use_agentic_retry = True  # Force agentic mode for ablations

        if args.ablation == 'no_retrieval_escalation':
            if args.policy_mode != 'paper_fixed':
                parser.error(
                    '--ablation no_retrieval_escalation requires '
                    '--policy-mode paper_fixed'
                )
            config.ablation_no_retrieval_escalation = True
        elif args.ablation == 'no_prompt_escalation':
            config.ablation_no_prompt_escalation = True
        elif args.ablation == 'no_untyped_citation_gate':
            config.ablation_no_untyped_citation_gate = True

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

    # Repeated-run support. Different local seeds do not imply independent
    # hosted generations; provider request seeds are tracked separately.
    num_runs = args.num_runs
    base_seed = args.seed
    base_generation_seed = args.generation_seed
    all_run_results = []  # Store DataFrames from each run
    all_run_paths = []  # Store paths to individual run files

    print("\n" + "=" * 60)
    print("EXPERIMENT CONFIGURATION")
    print("=" * 60)
    print(f"  Dataset:    {args.dataset}")
    print(f"  Pipeline:   {config.pipeline_id}")
    print(f"  Model:      {config.model_name}")
    print(f"  Local seed: {base_seed}")
    print(f"  Generation request seed: {base_generation_seed}")
    print(f"  Num runs:   {num_runs}")
    if config.use_agentic_retry:
        print(f"  Mode:       Agentic RAG (max_retries={config.max_retries})")
    else:
        print("  Mode:       Standard RAG")
    print("=" * 60)

    for run_idx in range(num_runs):
        current_seed = base_seed + run_idx
        current_generation_seed = (
            base_generation_seed + run_idx
            if base_generation_seed is not None
            else None
        )
        config.seed = current_seed
        config.generation_seed = current_generation_seed
        config.run_id = run_idx

        print(f"\n{'='*60}")
        print(
            f"RUN {run_idx + 1}/{num_runs} "
            f"(local_seed={current_seed}, generation_seed={current_generation_seed})"
        )
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
            # Preserve `seed` for existing analysis scripts while naming its
            # actual scope explicitly in new output columns.
            results_df['seed'] = current_seed
            results_df['local_seed'] = current_seed
            results_df['generation_seed'] = current_generation_seed

            all_run_results.append(results_df)

            # Save individual run results
            if num_runs > 1:
                if config.use_agentic_retry:
                    run_path = runner._save_agentic_results(
                        results_df,
                        adapter,
                        runner.agentic_orchestrator,
                        partial=False,
                    )
                else:
                    run_path = runner._save_results(
                        results_df, adapter, partial=False, include_run=True
                    )
                if run_path:
                    all_run_paths.append(run_path)
            else:
                # Single run: save normally
                if config.use_agentic_retry:
                    runner.save_agentic_results(
                        results_df, adapter, runner.agentic_orchestrator
                    )
                else:
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
) -> Dict[str, Path]:
    """Aggregate results across multiple runs and save summary.

    Aligns runs by question ID, averages repeated observations for each
    question, then bootstraps those question clusters. Saves:

    1. Combined CSV with all runs
    2. Per-question CSV with means and observation counts across runs
    3. JSON summary with per-run metrics and question-cluster statistics

    Raises:
        ValueError: If question IDs are absent, null, duplicated within a run,
            or do not form the same set in every run.
    """
    if not all_run_results:
        raise ValueError("At least one completed run is required for aggregation")

    reference_ids = None
    reference_id_set = None
    prepared_runs = []
    combined_runs = []
    per_run_metrics = []
    observed_local_seeds = []
    observed_generation_seeds = []

    def one_or_many(frame: pd.DataFrame, column: str):
        if column not in frame.columns:
            return None
        values = frame[column].dropna().unique().tolist()
        if len(values) == 1:
            return values[0]
        return values or None

    def numeric_column(frame: pd.DataFrame, column: str) -> pd.Series:
        if column not in frame.columns:
            return pd.Series(np.nan, index=frame.index, dtype=float)
        return pd.to_numeric(frame[column], errors="coerce").astype(float)

    def sum_available(*series: pd.Series) -> pd.Series:
        return pd.concat(series, axis=1).sum(axis=1, min_count=1)

    for run_index, run_df in enumerate(all_run_results):
        if 'question_id' not in run_df.columns:
            raise ValueError(f"Run {run_index} is missing required question_id")
        if run_df['question_id'].isna().any():
            raise ValueError(f"Run {run_index} contains null question_id values")
        duplicated = run_df.loc[
            run_df['question_id'].duplicated(keep=False), 'question_id'
        ].tolist()
        if duplicated:
            duplicate_labels = list(dict.fromkeys(map(str, duplicated)))
            raise ValueError(
                f"Run {run_index} contains duplicate question_id values: "
                f"{duplicate_labels}"
            )

        run_ids = run_df['question_id'].tolist()
        try:
            run_id_set = set(run_ids)
        except TypeError as exc:
            raise ValueError(
                f"Run {run_index} contains unhashable question_id values"
            ) from exc
        if reference_ids is None:
            reference_ids = run_ids
            reference_id_set = run_id_set
        elif run_id_set != reference_id_set:
            missing = sorted(map(str, reference_id_set - run_id_set))
            extra = sorted(map(str, run_id_set - reference_id_set))
            raise ValueError(
                f"Run {run_index} question_id set differs from run 0; "
                f"missing={missing}, extra={extra}"
            )

        prepared = pd.DataFrame({'question_id': run_df['question_id']})
        for column in ('correct', 'policy_accepted'):
            if column in run_df.columns:
                prepared[column] = _coerce_binary_series(
                    run_df[column], column
                ).reindex(run_df.index)
            else:
                prepared[column] = np.nan

        if 'abstained' in run_df.columns:
            abstained = _coerce_binary_series(
                run_df['abstained'], 'abstained'
            ).reindex(run_df.index).fillna(0.0)
        else:
            abstained = pd.Series(0.0, index=run_df.index, dtype=float)
        errors = (
            _error_mask(run_df['error']).astype(float)
            if 'error' in run_df.columns
            else pd.Series(0.0, index=run_df.index, dtype=float)
        )
        # Keep terminal outcomes mutually exclusive, matching metrics.py.
        prepared['error'] = errors
        prepared['abstained'] = abstained.where(errors == 0.0, 0.0)
        prepared['coverage'] = (
            (prepared['error'] == 0.0) & (prepared['abstained'] == 0.0)
        ).astype(float)

        for column in (
            'semantic_similarity',
            'judge_score',
            'numeric_accuracy',
            'prompt_tokens',
            'completion_tokens',
            'evaluation_prompt_tokens',
            'evaluation_completion_tokens',
            'llm_calls',
            'evaluation_llm_calls',
            'retrieval_time_ms',
            'generation_time_ms',
        ):
            prepared[column] = numeric_column(run_df, column)

        prepared['generation_tokens'] = sum_available(
            prepared['prompt_tokens'], prepared['completion_tokens']
        )
        prepared['evaluation_tokens'] = sum_available(
            prepared['evaluation_prompt_tokens'],
            prepared['evaluation_completion_tokens'],
        )
        prepared['total_tokens'] = sum_available(
            prepared['generation_tokens'], prepared['evaluation_tokens']
        )
        prepared['total_llm_calls'] = sum_available(
            prepared['llm_calls'], prepared['evaluation_llm_calls']
        )

        base_cost = numeric_column(run_df, 'cost_usd')
        if 'estimated_cost_usd' in run_df.columns:
            base_cost = base_cost.combine_first(
                numeric_column(run_df, 'estimated_cost_usd')
            )
        evaluation_cost = numeric_column(
            run_df, 'evaluation_estimated_cost_usd'
        )
        prepared['cost_usd'] = base_cost
        prepared['evaluation_cost_usd'] = evaluation_cost
        prepared['total_cost_usd'] = sum_available(base_cost, evaluation_cost)

        derived_latency = sum_available(
            prepared['retrieval_time_ms'], prepared['generation_time_ms']
        )
        prepared['total_time_ms'] = numeric_column(
            run_df, 'total_time_ms'
        ).combine_first(derived_latency)

        prepared_runs.append(prepared)
        combined_copy = run_df.copy()
        combined_copy['_aggregation_run_index'] = run_index
        combined_runs.append(combined_copy)

        local_seed_column = (
            'local_seed' if 'local_seed' in run_df.columns else 'seed'
        )
        local_seed = one_or_many(run_df, local_seed_column)
        generation_seed = one_or_many(run_df, 'generation_seed')
        observed_local_seeds.append(local_seed)
        observed_generation_seeds.append(generation_seed)
        per_run_metrics.append(
            {
                'run_index': run_index,
                'run_id': one_or_many(run_df, 'run_id'),
                'local_seed': local_seed,
                'local_seed_source': local_seed_column,
                'generation_seed_requested': generation_seed,
                'question_count': int(len(run_df)),
                'metrics': calculate_aggregate_metrics(run_df),
            }
        )

    combined_df = pd.concat(combined_runs, ignore_index=True)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    combined_path = output_dir / f"{config.timestamp}_{adapter.name}_combined_{len(all_run_results)}runs.csv"
    combined_df.to_csv(combined_path, index=False)
    print(f"\nCombined results saved to: {combined_path}")

    prepared_df = pd.concat(prepared_runs, ignore_index=True)
    metric_cols = [
        column for column in prepared_runs[0].columns if column != 'question_id'
    ]
    grouped = prepared_df.groupby('question_id', sort=False, dropna=False)
    per_question_df = pd.DataFrame({'question_id': reference_ids})
    per_question_df['run_observation_count'] = per_question_df[
        'question_id'
    ].map(grouped.size())

    for metadata_column in ('question', 'question_type'):
        if metadata_column in all_run_results[0].columns:
            metadata = all_run_results[0].set_index('question_id')[metadata_column]
            per_question_df[metadata_column] = per_question_df['question_id'].map(
                metadata
            )

    for column in metric_cols:
        per_question_df[f'{column}_mean'] = per_question_df['question_id'].map(
            grouped[column].mean()
        )
        per_question_df[f'{column}_run_count'] = per_question_df[
            'question_id'
        ].map(grouped[column].count())

    per_question_path = output_dir / (
        f"{config.timestamp}_{adapter.name}_per_question_"
        f"{len(all_run_results)}runs.csv"
    )
    per_question_df.to_csv(per_question_path, index=False)
    print(f"Per-question means saved to: {per_question_path}")

    question_cluster_stats = {}
    for column in metric_cols:
        values = per_question_df[f'{column}_mean'].dropna().astype(float)
        per_run_means = []
        for prepared in prepared_runs:
            run_values = prepared[column].dropna().astype(float)
            per_run_means.append(
                float(run_values.mean()) if len(run_values) else None
            )
        if len(values):
            mean, ci_lower, ci_upper = bootstrap_ci(
                values.tolist(), n_bootstrap=1000, seed=config.seed
            )
            question_cluster_stats[column] = {
                'mean': mean,
                'ci_95': [ci_lower, ci_upper],
                'std_across_questions': float(np.std(values.to_numpy())),
                'question_count': int(len(values)),
                'total_question_count': int(len(per_question_df)),
                'per_run_means': per_run_means,
                'bootstrap_unit': 'question_id',
                'run_reduction': 'mean_within_question',
            }
        else:
            question_cluster_stats[column] = {
                'mean': None,
                'ci_95': None,
                'std_across_questions': None,
                'question_count': 0,
                'total_question_count': int(len(per_question_df)),
                'per_run_means': per_run_means,
                'bootstrap_unit': 'question_id',
                'run_reduction': 'mean_within_question',
            }

    summary_path = output_dir / f"{config.timestamp}_{adapter.name}_aggregated_{len(all_run_results)}runs.json"
    selection_metadata = (
        adapter.get_selection_metadata()
        if callable(getattr(adapter, 'get_selection_metadata', None))
        else {}
    )
    summary = {
        'config': asdict(config),
        'dataset_selection': selection_metadata,
        'aggregation': {
            'num_runs': len(all_run_results),
            'question_count': len(per_question_df),
            'question_id_validation': 'unique_per_run_and_identical_sets',
            'bootstrap_unit': 'question_id',
            'run_reduction': 'mean_within_question',
            'local_seeds': observed_local_seeds,
            'generation_seeds_requested': observed_generation_seeds,
            'generation_reproducibility': (
                'provider_best_effort; repeated hosted generations are not '
                'guaranteed deterministic or independent'
            ),
            'run_files': [str(path) for path in all_run_paths],
        },
        'per_run_metrics': per_run_metrics,
        'question_cluster_stats': question_cluster_stats,
        'artifacts': {
            'combined_csv': str(combined_path),
            'per_question_csv': str(per_question_path),
        },
    }

    def json_safe(value):
        if isinstance(value, dict):
            return {str(key): json_safe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [json_safe(item) for item in value]
        if isinstance(value, np.generic):
            return json_safe(value.item())
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, float) and not np.isfinite(value):
            return None
        return value

    with open(summary_path, 'w') as f:
        json.dump(json_safe(summary), f, indent=2, allow_nan=False)
    print(f"Aggregated summary saved to: {summary_path}")

    print("\n" + "=" * 60)
    print("QUESTION-CLUSTER SUMMARY")
    print("=" * 60)
    for column in metric_cols:
        stats = question_cluster_stats[column]
        if stats['mean'] is None:
            continue
        ci = stats['ci_95']
        print(
            f"{column}: {stats['mean']:.4f} "
            f"[{ci[0]:.4f}, {ci[1]:.4f}] "
            f"(questions={stats['question_count']})"
        )
    print("\n" + "=" * 60)
    return {
        'combined_csv': combined_path,
        'per_question_csv': per_question_path,
        'summary_json': summary_path,
    }


if __name__ == "__main__":
    main()
