"""Evidence-gap-driven financial QA orchestration."""

import hashlib
import json
import time
from dataclasses import dataclass, field, fields, replace
from typing import Any, Dict, List, Optional

from .base import AgentDecision
from .retrieval_agent import RetrievalAgent
from .reasoning_agent import ReasoningAgent
from .judge_agent import JudgeAgent
from .logger import AgentLogger
from .correction_policy import CorrectionAction, CorrectionPlan, CorrectionPolicy
from src.providers.base import get_usage_tracker
from src.config import calculate_cost
from src.finance_contract import build_finance_question_spec
from src.query_understanding import FinanceQueryPlan, compile_finance_query


@dataclass
class AgenticRAGConfig:
    """Configuration for the agentic RAG orchestrator."""
    # Agent settings
    max_retries: int = 1  # Maximum retry attempts (0 = no retries)
    retry_threshold: float = 0.5  # Score below which to retry
    blind_judge: bool = False  # If True, Judge uses self-evaluation without gold answer
    policy_mode: str = "gap_driven_v2"  # or paper_fixed for historical reproduction

    # Model settings
    llm_model: str = None  # Will use config default
    judge_model: str = None  # Will use config default
    reranker_model: str = None  # Will use config default

    # Retrieval settings
    pipeline_id: str = "hybrid_filter_rerank"
    top_k: int = 10
    initial_k_factor: float = 3.0
    use_rule_router: bool = True  # Use free rule-based routing
    use_rse: bool = False  # Use Relevant Segment Extraction

    # Logging settings
    log_dir: str = "agent_logs"
    enable_logging: bool = True

    # Generation settings
    temperature: float = 0.0
    max_tokens: int = 4096
    generation_seed: Optional[int] = None

    # Ablation study settings
    # These flags disable specific components to measure their contribution
    ablation_no_retrieval_escalation: bool = False  # Fix top_k=10 on all attempts
    ablation_no_prompt_escalation: bool = False     # Fix "standard" prompt on all attempts
    ablation_no_untyped_citation_gate: bool = False  # Calculation verifier remains required

    def __post_init__(self) -> None:
        if (
            not isinstance(self.max_retries, int)
            or isinstance(self.max_retries, bool)
            or self.max_retries < 0
        ):
            raise ValueError("max_retries must be a non-negative integer")
        if not 0.0 <= self.retry_threshold <= 1.0:
            raise ValueError("retry_threshold must be between 0 and 1")
        if (
            not isinstance(self.max_tokens, int)
            or isinstance(self.max_tokens, bool)
            or self.max_tokens < 1
        ):
            raise ValueError("max_tokens must be a positive integer")
        if (
            not isinstance(self.top_k, int)
            or isinstance(self.top_k, bool)
            or self.top_k < 1
        ):
            raise ValueError("top_k must be a positive integer")
        if self.initial_k_factor < 1.0:
            raise ValueError("initial_k_factor must be at least 1.0")
        if self.policy_mode not in {"gap_driven_v2", "paper_fixed"}:
            raise ValueError(
                "policy_mode must be 'gap_driven_v2' or 'paper_fixed'; "
                f"got {self.policy_mode!r}"
            )
        if self.policy_mode == "gap_driven_v2" and self.max_tokens < 2048:
            raise ValueError(
                "gap_driven_v2 requires max_tokens >= 2048 so the strict "
                "FinanceProgram is not truncated"
            )
        if (
            self.policy_mode == "gap_driven_v2"
            and self.ablation_no_retrieval_escalation
        ):
            raise ValueError(
                "no_retrieval_escalation is a paper_fixed-only ablation; "
                "gap_driven_v2 uses targeted retrieval rather than depth escalation"
            )
        supported = {
            "semantic",
            "hybrid",
            "hybrid_filter",
            "hybrid_filter_rerank",
        }
        if self.pipeline_id not in supported:
            raise ValueError(
                "Agentic retry requires a fixed pipeline so its retrieval-depth "
                f"schedule is reproducible; got {self.pipeline_id!r}. "
                f"Choose one of {sorted(supported)}."
            )


def _summarize_usage_records(
    records: List[Dict[str, Any]],
    *,
    fallback_model: str,
) -> Dict[str, Any]:
    """Aggregate completed provider calls without losing model attribution."""

    by_model: Dict[str, Dict[str, Any]] = {}
    total_prompt = 0
    total_completion = 0
    total_cost = 0.0
    for record in records:
        model = str(record.get("model") or fallback_model)
        provider = record.get("provider")
        prompt_tokens = int(record.get("prompt_tokens", 0) or 0)
        completion_tokens = int(record.get("completion_tokens", 0) or 0)
        cost = calculate_cost(
            model,
            {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
            },
        )
        entry = by_model.setdefault(
            model,
            {
                "provider": provider,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "calls": 0,
                "estimated_cost_usd": 0.0,
            },
        )
        entry["prompt_tokens"] += prompt_tokens
        entry["completion_tokens"] += completion_tokens
        entry["calls"] += 1
        entry["estimated_cost_usd"] += cost
        request_metadata = record.get("request_metadata")
        if isinstance(request_metadata, dict):
            entry.setdefault("requests", []).append(request_metadata.copy())
        total_prompt += prompt_tokens
        total_completion += completion_tokens
        total_cost += cost
    return {
        "prompt_tokens": total_prompt,
        "completion_tokens": total_completion,
        "calls": len(records),
        "estimated_cost_usd": total_cost,
        "by_model": by_model,
    }


def _json_safe_manifest_value(value: Any) -> Any:
    """Normalize one metadata value for stable JSON/CSV persistence."""

    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, tuple)):
        return [_json_safe_manifest_value(item) for item in value]
    if hasattr(value, "item"):
        try:
            return _json_safe_manifest_value(value.item())
        except (TypeError, ValueError):
            pass
    return str(value)


def _build_evidence_manifest(docs: List[Any]) -> Dict[str, Dict[str, Any]]:
    """Map positional Doc labels to verifiable persisted source provenance."""

    manifest: Dict[str, Dict[str, Any]] = {}
    for index, doc in enumerate(docs, 1):
        metadata = getattr(doc, "metadata", {}) or {}
        content = getattr(doc, "page_content", "") or ""
        entry: Dict[str, Any] = {
            "source": _json_safe_manifest_value(
                metadata.get("source") or "Unknown"
            ),
            "content_sha256": hashlib.sha256(
                content.encode("utf-8")
            ).hexdigest(),
        }
        for key in (
            "page",
            "positions",
            "chunk_id",
            "child_chunk_ids",
            "corpus_version",
            "index_version",
            "parser_version",
            "embedding_model",
            "source_id",
            "doc_name",
            "form",
            "filing_date",
            "accession_number",
            "rank",
            "score",
            "retrieval_score",
            "rerank_score",
            "reranker_score",
        ):
            value = metadata.get(key)
            if value is not None:
                entry[key] = _json_safe_manifest_value(value)
        manifest[f"Doc{index}"] = entry
    return manifest


def _document_identity(doc: Any) -> str:
    """Return a stable identity for evidence deduplication and fingerprints."""

    metadata = getattr(doc, "metadata", {}) or {}
    content = getattr(doc, "page_content", "") or ""
    content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
    for key in ("chunk_id", "child_chunk_ids"):
        value = metadata.get(key)
        if value:
            identity = {
                "kind": key,
                "id": _json_safe_manifest_value(value),
                "corpus_version": metadata.get("corpus_version")
                or metadata.get("index_version"),
                "content_sha256": content_hash,
            }
            return json.dumps(
                _json_safe_manifest_value(identity),
                sort_keys=True,
                separators=(",", ":"),
            )
    fallback = {
        "source": metadata.get("source"),
        "page": metadata.get("page"),
        "positions": metadata.get("positions"),
        "content_sha256": content_hash,
    }
    encoded = json.dumps(
        _json_safe_manifest_value(fallback),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def _evidence_fingerprint(docs: List[Any]) -> str:
    """Hash the ordered evidence snapshot used by one attempt."""

    encoded = json.dumps(
        [_document_identity(doc) for doc in docs],
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:20]


def _merge_evidence(existing: List[Any], retrieved: List[Any]) -> List[Any]:
    """Append newly retrieved evidence without renumbering existing Doc labels."""

    merged = list(existing)
    seen = {_document_identity(doc) for doc in merged}
    for doc in retrieved:
        identity = _document_identity(doc)
        if identity not in seen:
            merged.append(doc)
            seen.add(identity)
    return merged


def _targeted_query(
    question: str,
    query_plan: FinanceQueryPlan,
    correction: Optional[CorrectionPlan],
) -> str:
    """Build one focused retrieval query for the affected evidence needs."""

    if correction is None or not correction.affected_need_ids:
        return question
    affected = set(correction.affected_need_ids)
    queries = [
        need.query
        for need in query_plan.evidence_needs
        if need.need_id in affected
    ]
    queries = list(dict.fromkeys(query for query in queries if query))
    if not queries:
        return question
    return f"{question}\nTarget missing evidence: {'; '.join(queries)}"


def _verification_report_from_decision(decision: AgentDecision) -> Dict[str, Any]:
    """Translate a Judge decision into the correction-policy wire format."""

    values = decision.decision_value
    policy_pass = values.get("pass")
    if policy_pass is None:
        # Backward-compatible decisions and test doubles predate the explicit
        # `pass` field. A non-retry, non-abstain decision meant acceptance.
        policy_pass = not values.get("retry", False) and not values.get(
            "abstain", False
        )
    passed = bool(
        policy_pass
        and values.get("verification_passed", True)
        and not values.get("abstain", False)
    )
    structured_issues = values.get("verification_issues") or []
    reason_codes = (
        values.get("verification_reason_codes")
        or decision.metadata.get("verification_reason_codes")
        or []
    )
    if not passed and not reason_codes:
        reason_codes = ["low_policy_score"]
    return {
        "passed": passed,
        "issues": structured_issues or list(reason_codes),
        "reason_codes": list(reason_codes),
    }


@dataclass
class AgenticRAGResult:
    """Result from processing a question through the agentic system."""
    question_id: str
    question: str
    final_answer: Optional[str]
    final_score: float
    correct: Optional[bool]
    policy_accepted: bool
    evaluation_mode: str
    abstained: bool
    attempts: int
    improvement_from_retry: bool  # Legacy: policy-score improvement, not correctness
    decision_log: List[Dict[str, Any]]
    retrieval_time_ms: float
    generation_time_ms: float
    total_time_ms: float
    error: Optional[str] = None
    cost_usd: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    llm_calls: int = 0
    usage_by_model: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    evidence_manifest: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    query_plan: Dict[str, Any] = field(default_factory=dict)
    correction_history: List[Dict[str, Any]] = field(default_factory=list)
    finance_question_spec: Dict[str, Any] = field(default_factory=dict)
    finance_program: Optional[Dict[str, Any]] = None
    finance_verification: Optional[Dict[str, Any]] = None


class AgenticRAGOrchestrator:
    """Coordinates all agents to process questions with self-correction.

    The orchestrator implements the core agentic RAG loop:

    ```
    while attempt <= max_retries:
        1. RetrievalAgent decides strategy and retrieves documents
        2. ReasoningAgent generates answer from documents
        3. JudgeAgent evaluates and decides: accept or retry?

        if accepted:
            break
        else:
            escalate all agent strategies
            attempt += 1
    ```

    All decisions are logged for analysis and interpretability.
    """

    def __init__(self, config: AgenticRAGConfig = None, db=None, embedding_fn=None):
        """Initialize the orchestrator.

        Args:
            config: Configuration for the agentic system
            db: ChromaDB instance
            embedding_fn: Embedding function for retrieval
        """
        self.config = config or AgenticRAGConfig()
        self.db = db
        self.embedding_fn = embedding_fn

        # Initialize agents with ablation settings
        self.retrieval_agent = RetrievalAgent(
            db=db,
            embedding_fn=embedding_fn,
            reranker_model=self.config.reranker_model,
            pipeline_id=self.config.pipeline_id,
            base_top_k=self.config.top_k,
            base_initial_k_factor=self.config.initial_k_factor,
            use_rule_router=self.config.use_rule_router,
            use_rse=self.config.use_rse,
            disable_escalation=self.config.ablation_no_retrieval_escalation,
        )

        self.reasoning_agent = ReasoningAgent(
            model_name=self.config.llm_model,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            disable_escalation=self.config.ablation_no_prompt_escalation,
            generation_seed=self.config.generation_seed,
        )

        self.judge_agent = JudgeAgent(
            judge_model=self.config.judge_model,
            retry_threshold=self.config.retry_threshold,
            threshold_decay=(
                0.1 if self.config.policy_mode == "paper_fixed" else 0.0
            ),
            enable_deterministic_gate=(
                not self.config.ablation_no_untyped_citation_gate
            ),
        )

        # Initialize logger
        if self.config.enable_logging:
            self.logger = AgentLogger(output_dir=self.config.log_dir)
        else:
            self.logger = None

        # Track overall statistics
        self.total_questions = 0
        self.total_retries = 0
        self.successful_retries = 0

        # Cost tracking
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_llm_calls = 0
        self.total_usage_by_model: Dict[str, Dict[str, Any]] = {}
        self.total_cost_usd = 0.0

    def reset_agents(self) -> None:
        """Reset all agents for a new question."""
        self.retrieval_agent.reset()
        self.reasoning_agent.reset()
        self.judge_agent.reset()

    def process_question(
        self,
        question: str,
        gold_answer: str = None,
        question_id: str = None,
    ) -> AgenticRAGResult:
        """Process a single question through the agentic pipeline.

        Args:
            question: The question to answer
            gold_answer: Reference answer for evaluation (optional)
            question_id: Unique identifier for logging

        Returns:
            AgenticRAGResult with answer and decision log
        """
        start_time = time.time()

        # Snapshot completed provider calls before this question. Aggregate
        # totals are global, so subtracting them across orchestrators can leak
        # unrelated work; a record cursor preserves per-question attribution.
        tracker = get_usage_tracker()
        usage_cursor = tracker.cursor()

        # Generate question ID if not provided
        if question_id is None:
            question_id = f"Q{self.total_questions:04d}"

        self.total_questions += 1

        # Reset agents for new question
        self.reset_agents()

        # Compile the question once. The plan is a deterministic contract for
        # retrieval, calculation, verification, and post-hoc evaluation.
        query_plan = compile_finance_query(question)
        correction_policy = CorrectionPolicy(self.config.policy_mode)
        require_finance_program = bool(
            self.config.policy_mode == "gap_driven_v2"
            and query_plan.requires_calculation
        )
        finance_question_spec = (
            build_finance_question_spec(query_plan)
            if require_finance_program
            else None
        )
        finance_question_spec_payload = (
            finance_question_spec.model_dump(mode="json")
            if finance_question_spec is not None
            else {}
        )

        # Start logging
        if self.logger:
            self.logger.start_question(question_id, question, gold_answer)
            self.logger.log_query_plan(query_plan.to_dict())
            self.logger.log_finance_question_spec(finance_question_spec_payload)

        # Initialize tracking variables
        decision_log = []
        attempt = 0
        final_answer = None
        final_score = 0.0
        best_answer = None
        best_score = float("-inf")
        best_answer_manifest: Dict[str, Dict[str, Any]] = {}
        best_answer_program: Optional[Dict[str, Any]] = None
        best_answer_finance_verification: Optional[Dict[str, Any]] = None
        best_verified_answer = None
        best_verified_score = float("-inf")
        best_verified_manifest: Dict[str, Dict[str, Any]] = {}
        best_verified_program: Optional[Dict[str, Any]] = None
        best_verified_finance_verification: Optional[Dict[str, Any]] = None
        final_evidence_manifest: Dict[str, Dict[str, Any]] = {}
        final_finance_program: Optional[Dict[str, Any]] = None
        final_finance_verification: Optional[Dict[str, Any]] = None
        previous_answer = None
        retry_feedback = ""
        policy_accepted = False
        abstained = False
        retrieval_time_ms = 0.0
        generation_time_ms = 0.0
        error = None
        current_docs: List[Any] = []
        current_correction: Optional[CorrectionPlan] = None
        correction_history: List[Dict[str, Any]] = []
        gap_fingerprints: List[str] = []

        # Relative periods need a corpus filing calendar, and calculations need
        # an exact trusted AST.  Neither may be guessed by the answer model.
        preflight_issues = [
            {
                "code": "unresolved_constraint",
                "message": constraint,
            }
            for constraint in query_plan.unresolved_constraints
        ]
        if require_finance_program and finance_question_spec is None:
            preflight_issues.append(
                {
                    "code": "unsupported_claim",
                    "message": "No full trusted finance-program contract could be compiled",
                }
            )
        preflight_abstain = bool(
            self.config.policy_mode == "gap_driven_v2" and preflight_issues
        )
        if preflight_abstain:
            if self.logger:
                self.logger.start_attempt(0)
            preflight_plan = correction_policy.plan(
                {"passed": False, "issues": preflight_issues},
                attempt=0,
                max_attempts=1,
                evidence_fingerprint="preflight",
                evidence_needs=query_plan.evidence_needs,
            )
            correction_payload = preflight_plan.to_dict()
            correction_history.append(correction_payload)
            if self.logger:
                self.logger.log_correction_plan(correction_payload)
            decision_log.append(
                {
                    "attempt": 0,
                    "state": "preflight",
                    "issues": preflight_issues,
                    "correction": correction_payload,
                    "attempt_time_ms": 0.0,
                }
            )
            abstained = True

        # Main retry loop
        while not preflight_abstain and attempt <= self.config.max_retries:
            attempt_start = time.time()

            if self.logger:
                self.logger.start_attempt(attempt)

            try:
                # === Agent A: Retrieval ===
                retrieval_start = time.time()
                should_retrieve = (
                    attempt == 0
                    or current_correction is None
                    or current_correction.requires_retrieval
                )
                if should_retrieve:
                    retrieval_query = (
                        question
                        if self.config.policy_mode == "paper_fixed"
                        else _targeted_query(
                            question,
                            query_plan,
                            current_correction,
                        )
                    )
                    strategy_attempt = (
                        attempt if self.config.policy_mode == "paper_fixed" else 0
                    )
                    retrieval_decision = self.retrieval_agent.decide({
                        "question": retrieval_query,
                        "attempt": strategy_attempt,
                        "workflow_attempt": attempt,
                        "query_plan": query_plan.to_dict(),
                        "correction_plan": (
                            current_correction.to_dict()
                            if current_correction is not None
                            else None
                        ),
                    })
                    retrieved_docs = self.retrieval_agent.retrieve(
                        retrieval_query,
                        retrieval_decision,
                    )
                    if (
                        self.config.policy_mode == "gap_driven_v2"
                        and current_docs
                        and current_correction is not None
                    ):
                        docs = _merge_evidence(current_docs, retrieved_docs)
                    else:
                        docs = retrieved_docs
                    current_docs = docs
                else:
                    docs = current_docs
                    retrieval_decision = AgentDecision(
                        agent_name="RetrievalAgent",
                        decision_type="evidence_reuse",
                        decision_value={
                            "reuse_evidence": True,
                            "num_documents": len(docs),
                            "correction_action": current_correction.action.value,
                        },
                        confidence=1.0,
                        reasoning=(
                            "Reusing the verified evidence snapshot because the "
                            "diagnosed gap does not require another retrieval call."
                        ),
                        metadata={
                            "attempt": attempt,
                            "evidence_fingerprint": _evidence_fingerprint(docs),
                        },
                    )

                retrieval_time_ms += (time.time() - retrieval_start) * 1000

                if self.logger:
                    self.logger.log_decision("retrieval_agent", retrieval_decision)

                if not docs:
                    error = "No documents retrieved"
                    break

                evidence_manifest = _build_evidence_manifest(docs)
                if self.logger:
                    self.logger.log_evidence_manifest(evidence_manifest)

                # === Agent B: Reasoning ===
                generation_start = time.time()

                reasoning_decision = self.reasoning_agent.decide({
                    "question": question,
                    "documents": docs,
                    "attempt": attempt,
                    "previous_answer": previous_answer,
                    "retry_feedback": retry_feedback,
                    "query_plan": query_plan.to_dict(),
                    "correction_plan": (
                        current_correction.to_dict()
                        if current_correction is not None
                        else None
                    ),
                    "require_finance_program": require_finance_program,
                    "finance_question_spec": finance_question_spec_payload,
                })

                generation_time_ms += (time.time() - generation_start) * 1000

                if self.logger:
                    self.logger.log_decision("reasoning_agent", reasoning_decision)

                answer = reasoning_decision.decision_value.get("answer")
                finance_program = reasoning_decision.decision_value.get(
                    "finance_program"
                )
                finance_program_issues = reasoning_decision.decision_value.get(
                    "finance_program_issues", []
                )

                generation_error = reasoning_decision.decision_value.get("error")
                if generation_error:
                    error = f"Reasoning provider failed: {generation_error}"
                    decision_log.append({
                        "attempt": attempt,
                        "retrieval": retrieval_decision.to_dict(),
                        "evidence_manifest": evidence_manifest,
                        "reasoning": reasoning_decision.to_dict(),
                        "judge": None,
                        "correction": None,
                        "state": "error",
                        "attempt_time_ms": (time.time() - attempt_start) * 1000,
                    })
                    break

                if not answer and not require_finance_program:
                    error = "No answer generated"
                    decision_log.append({
                        "attempt": attempt,
                        "retrieval": retrieval_decision.to_dict(),
                        "evidence_manifest": evidence_manifest,
                        "reasoning": reasoning_decision.to_dict(),
                        "judge": None,
                        "correction": None,
                        "state": "error",
                        "attempt_time_ms": (time.time() - attempt_start) * 1000,
                    })
                    break

                # === Agent C: Judge ===
                # In blind mode, Judge uses self-evaluation without gold answer.
                # Documents are passed for blind numeric grounding verification.
                judge_gold = (
                    gold_answer
                    if self.config.policy_mode == "paper_fixed"
                    and not self.config.blind_judge
                    else None
                )
                judge_decision = self.judge_agent.decide({
                    "question": question,
                    "predicted_answer": answer,
                    "gold_answer": judge_gold,
                    "attempt": attempt,
                    "max_retries": self.config.max_retries,
                    "documents": docs,
                    "query_plan": query_plan.to_dict(),
                    "require_finance_program": require_finance_program,
                    "finance_program": finance_program,
                    "finance_program_issues": finance_program_issues,
                    "finance_question_spec": finance_question_spec_payload or None,
                })

                if self.logger:
                    self.logger.log_decision("judge_agent", judge_decision)

                score = judge_decision.decision_value.get("score", 0.0)
                should_retry = judge_decision.decision_value.get("retry", False)
                should_abstain = judge_decision.decision_value.get("abstain", False)
                canonical_answer = judge_decision.decision_value.get(
                    "canonical_answer"
                )
                finance_verification = judge_decision.decision_value.get(
                    "finance_program_verification"
                )
                if canonical_answer and judge_decision.decision_value.get(
                    "verification_passed", False
                ):
                    answer = canonical_answer

                # Track best answer
                if score > best_score:
                    best_score = score
                    best_answer = answer
                    best_answer_manifest = evidence_manifest
                    best_answer_program = finance_program
                    best_answer_finance_verification = finance_verification
                if (
                    judge_decision.decision_value.get("verification_passed", True)
                    and score > best_verified_score
                ):
                    best_verified_score = score
                    best_verified_answer = answer
                    best_verified_manifest = evidence_manifest
                    best_verified_program = finance_program
                    best_verified_finance_verification = finance_verification

                # Preserve gold-free feedback for the next attempt. In oracle
                # mode, an LLM justification may disclose the reference answer,
                # so only deterministic feedback is forwarded verbatim.
                # Doc numbers are positional and may change after retrieval
                # escalation. Do not invite the reasoner to copy stale labels.
                from evaluation.numeric_check import strip_evidence_citations
                previous_answer = strip_evidence_citations(answer or "").strip()
                judge_values = judge_decision.decision_value
                deterministic_failure = judge_decision.metadata.get(
                    "deterministic_gate_triggered", False
                )
                if self.config.blind_judge or deterministic_failure:
                    retry_feedback = judge_values.get(
                        "verification_feedback"
                    ) or judge_values.get("justification", "")
                else:
                    retry_feedback = (
                        "The policy judge rejected the previous answer. Re-check "
                        "the retrieved evidence, calculation, units, entity, and period."
                    )

                # Convert verifier/Judge output into the smallest justified next
                # action. Unlike the historical policy, a citation or rendering
                # problem reuses evidence; only evidence gaps trigger retrieval.
                correction_plan = correction_policy.plan(
                    _verification_report_from_decision(judge_decision),
                    attempt=attempt,
                    max_attempts=self.config.max_retries + 1,
                    evidence_fingerprint=_evidence_fingerprint(docs),
                    previous_gap_fingerprints=gap_fingerprints,
                    evidence_needs=query_plan.evidence_needs,
                )
                correction_payload = correction_plan.to_dict()
                local_repair_applied = False

                # The verifier already executed the trusted AST.  When only the
                # model-supplied result or visible rendering is wrong, use that
                # canonical result directly instead of spending another LLM call.
                if correction_plan.action in {
                    CorrectionAction.LOCAL_RECOMPUTE,
                    CorrectionAction.RERENDER,
                }:
                    repaired_program = judge_decision.decision_value.get(
                        "repaired_finance_program"
                    )
                    repaired_verification = judge_decision.decision_value.get(
                        "repaired_finance_program_verification"
                    )
                    repair_reverified = bool(
                        repaired_program
                        and repaired_verification
                        and repaired_verification.get("fully_verified", False)
                    )
                    if canonical_answer and repair_reverified:
                        local_repair_applied = True
                        should_retry = False
                        should_abstain = False
                        final_answer = canonical_answer
                        final_score = score
                        final_evidence_manifest = evidence_manifest
                        final_finance_program = repaired_program
                        final_finance_verification = dict(repaired_verification)
                        final_finance_verification.update(
                            {
                                "locally_repaired": True,
                                "repaired_and_reverified": True,
                                "repair_action": correction_plan.action.value,
                                "final_rendered_answer": canonical_answer,
                            }
                        )
                        policy_accepted = True
                        correction_payload["local_repair_applied"] = True
                        correction_payload["repaired_and_reverified"] = True
                        correction_payload["canonical_answer"] = canonical_answer
                    else:
                        # A local action without an executable canonical result is
                        # an internal contract failure; never expose the draft.
                        should_retry = False
                        should_abstain = True
                        correction_payload["local_repair_applied"] = False
                        correction_payload["local_repair_error"] = (
                            "Verifier supplied no fully reverified repair artifact"
                        )
                correction_history.append(correction_payload)
                if self.logger:
                    self.logger.log_correction_plan(correction_payload)

                if local_repair_applied:
                    pass
                elif correction_plan.action is CorrectionAction.ACCEPT:
                    should_retry = False
                elif correction_plan.terminal:
                    should_retry = False
                    should_abstain = True
                else:
                    should_retry = True
                    gap_fingerprints.append(correction_plan.gap_fingerprint)

                # Log this attempt
                decision_log.append({
                    "attempt": attempt,
                    "retrieval": retrieval_decision.to_dict(),
                    "evidence_manifest": evidence_manifest,
                    "reasoning": reasoning_decision.to_dict(),
                    "judge": judge_decision.to_dict(),
                    "correction": correction_payload,
                    "attempt_time_ms": (time.time() - attempt_start) * 1000,
                })

                # Check if we should retry
                if not should_retry:
                    if local_repair_applied:
                        pass
                    elif should_abstain:
                        # A failed final grounding gate must not leak the
                        # unsupported draft as the system's final answer. The
                        # historical policy may retain its earlier behavior, but
                        # v2 always makes abstention explicit.
                        if (
                            self.config.policy_mode == "paper_fixed"
                            and best_verified_answer is not None
                        ):
                            final_answer = best_verified_answer
                            final_score = best_verified_score
                            final_evidence_manifest = best_verified_manifest
                            final_finance_program = best_verified_program
                            final_finance_verification = (
                                best_verified_finance_verification
                            )
                            abstained = False
                        else:
                            abstained = True
                            final_answer = None
                            final_score = 0.0
                            final_evidence_manifest = {}
                            final_finance_program = None
                            final_finance_verification = finance_verification
                        policy_accepted = False
                    else:
                        # Return the strongest verified candidate, not merely
                        # the last candidate generated before the loop stopped.
                        if self.config.policy_mode == "gap_driven_v2":
                            final_answer = canonical_answer or answer
                            final_score = score
                            final_evidence_manifest = evidence_manifest
                            final_finance_program = finance_program
                            final_finance_verification = finance_verification
                        elif best_verified_answer is not None:
                            final_answer = best_verified_answer
                            final_score = best_verified_score
                            final_evidence_manifest = best_verified_manifest
                            final_finance_program = best_verified_program
                            final_finance_verification = (
                                best_verified_finance_verification
                            )
                        else:
                            final_answer = best_answer
                            final_score = best_score
                            final_evidence_manifest = best_answer_manifest
                            final_finance_program = best_answer_program
                            final_finance_verification = (
                                best_answer_finance_verification
                            )
                        threshold = judge_decision.decision_value.get(
                            "acceptance_threshold", self.config.retry_threshold
                        )
                        policy_accepted = bool(
                            judge_decision.decision_value.get("pass", False)
                            or final_score >= threshold
                        )
                    break

                # Escalate strategies for retry
                self.total_retries += 1
                if self.config.policy_mode == "paper_fixed":
                    self.retrieval_agent.escalate_strategy()
                    self.reasoning_agent.escalate_strategy()

                current_correction = correction_plan

                attempt += 1

            except Exception as e:
                error = f"Error in attempt {attempt}: {str(e)}"
                import traceback
                traceback.print_exc()
                break

        # Historical reproduction retains the original best-candidate fallback.
        # In v2, reaching this point means no candidate passed the controller.
        if final_answer is None and not abstained:
            if error is not None:
                # Infrastructure/provider failures are not model abstentions.
                # Keep them out of selective-risk coverage and expose the
                # failure explicitly in per-question artifacts.
                policy_accepted = False
                final_score = 0.0
                final_evidence_manifest = {}
                final_finance_program = None
                final_finance_verification = None
            elif self.config.policy_mode == "gap_driven_v2":
                abstained = True
                policy_accepted = False
                final_score = 0.0
                final_evidence_manifest = {}
                final_finance_program = None
                final_finance_verification = None
            else:
                final_answer = best_verified_answer or best_answer
                if best_verified_answer is not None:
                    final_score = best_verified_score
                    final_evidence_manifest = best_verified_manifest
                    final_finance_program = best_verified_program
                    final_finance_verification = best_verified_finance_verification
                else:
                    final_score = best_score if best_score != float("-inf") else 0.0
                    final_evidence_manifest = best_answer_manifest
                    final_finance_program = best_answer_program
                    final_finance_verification = best_answer_finance_verification

        # A policy threshold is a stopping rule, not a correctness label.  The
        # historical controller relaxes its threshold across attempts, so an
        # accepted answer may still fall below the fixed evaluation threshold.
        # Blind outputs have no correctness label here and must be evaluated
        # independently after selection.
        policy_accepted = bool(
            policy_accepted and final_answer is not None and not abstained
        )
        gold_used_by_policy = bool(
            self.config.policy_mode == "paper_fixed"
            and not self.config.blind_judge
            and gold_answer is not None
        )
        correct = (
            bool(final_answer is not None and final_score >= self.config.retry_threshold)
            if gold_used_by_policy
            else None
        )
        evaluation_mode = "oracle_guided" if gold_used_by_policy else "blind_policy"

        # Check if retry improved the result
        improvement_from_retry = False
        judged_attempts = [entry for entry in decision_log if entry.get("judge")]
        if len(judged_attempts) > 1:
            first_score = judged_attempts[0]["judge"]["decision_value"].get(
                "score", 0.0
            )
            if policy_accepted and final_score > first_score:
                improvement_from_retry = True
                self.successful_retries += 1

        total_time_ms = (time.time() - start_time) * 1000

        usage = _summarize_usage_records(
            tracker.records_since(usage_cursor),
            fallback_model=self.config.llm_model or "gpt-4o-mini",
        )
        question_prompt_tokens = usage["prompt_tokens"]
        question_completion_tokens = usage["completion_tokens"]
        question_cost = usage["estimated_cost_usd"]
        self.total_cost_usd += question_cost
        self.total_prompt_tokens += question_prompt_tokens
        self.total_completion_tokens += question_completion_tokens
        self.total_llm_calls += usage["calls"]
        for model, model_usage in usage["by_model"].items():
            total = self.total_usage_by_model.setdefault(
                model,
                {
                    "provider": model_usage.get("provider"),
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "calls": 0,
                    "estimated_cost_usd": 0.0,
                },
            )
            for key in (
                "prompt_tokens",
                "completion_tokens",
                "calls",
                "estimated_cost_usd",
            ):
                total[key] += model_usage[key]

        # Finish logging
        if self.logger:
            self.logger.finish_question(
                final_answer,
                correct,
                improvement_from_retry,
                policy_accepted=policy_accepted,
                evaluation_mode=evaluation_mode,
                evidence_manifest=final_evidence_manifest,
                finance_question_spec=finance_question_spec_payload,
                finance_program=final_finance_program,
                finance_verification=final_finance_verification,
                error=error,
                abstained=abstained,
                prompt_tokens=question_prompt_tokens,
                completion_tokens=question_completion_tokens,
                llm_calls=usage["calls"],
                usage_by_model=usage["by_model"],
                estimated_cost_usd=question_cost,
            )

        return AgenticRAGResult(
            question_id=question_id,
            question=question,
            final_answer=final_answer,
            final_score=final_score,
            correct=correct,
            policy_accepted=policy_accepted,
            evaluation_mode=evaluation_mode,
            abstained=abstained,
            attempts=min(attempt + 1, self.config.max_retries + 1),
            improvement_from_retry=improvement_from_retry,
            decision_log=decision_log,
            retrieval_time_ms=retrieval_time_ms,
            generation_time_ms=generation_time_ms,
            total_time_ms=total_time_ms,
            error=error,
            cost_usd=question_cost,
            prompt_tokens=question_prompt_tokens,
            completion_tokens=question_completion_tokens,
            llm_calls=usage["calls"],
            usage_by_model=usage["by_model"],
            evidence_manifest=final_evidence_manifest,
            query_plan=query_plan.to_dict(),
            correction_history=correction_history,
            finance_question_spec=finance_question_spec_payload,
            finance_program=final_finance_program,
            finance_verification=final_finance_verification,
        )

    def process_batch(
        self,
        questions: List[str],
        gold_answers: List[str] = None,
        question_ids: List[str] = None,
        show_progress: bool = True,
    ) -> List[AgenticRAGResult]:
        """Process multiple questions.

        Args:
            questions: List of questions
            gold_answers: List of reference answers (optional)
            question_ids: List of question IDs (optional)
            show_progress: Whether to show progress bar

        Returns:
            List of AgenticRAGResult objects
        """
        results = []

        if gold_answers is None:
            gold_answers = [None] * len(questions)
        if question_ids is None:
            question_ids = [f"Q{i:04d}" for i in range(len(questions))]

        iterator = zip(questions, gold_answers, question_ids)
        if show_progress:
            from tqdm import tqdm
            iterator = tqdm(list(iterator), desc="Processing questions")

        for question, gold, qid in iterator:
            result = self.process_question(question, gold, qid)
            results.append(result)

        return results

    def export_decisions(self, filename: str = None) -> Dict[str, str]:
        """Export all logged decisions.

        Args:
            filename: Base filename (without extension)

        Returns:
            Dictionary with paths to exported files
        """
        if self.logger is None:
            return {}

        paths = {}
        paths["json"] = self.logger.export_json(filename)
        paths["csv"] = self.logger.export_csv(filename)

        return paths

    def get_statistics(self) -> Dict[str, Any]:
        """Get overall statistics for the session.

        Returns:
            Dictionary with session statistics
        """
        stats = {
            "total_questions": self.total_questions,
            "total_retries": self.total_retries,
            "successful_retries": self.successful_retries,
            "retry_rate": self.total_retries / max(1, self.total_questions),
            "retry_success_rate": self.successful_retries / max(1, self.total_retries),
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "total_llm_calls": self.total_llm_calls,
            "usage_by_model": self.total_usage_by_model,
            "total_cost_usd": self.total_cost_usd,
            "avg_cost_per_question": self.total_cost_usd / max(1, self.total_questions),
        }

        if self.logger:
            stats.update(self.logger.summary())

        return stats

    def print_summary(self) -> None:
        """Print session summary to console."""
        if self.logger:
            self.logger.print_summary()
        else:
            stats = self.get_statistics()
            print(f"\nProcessed {stats['total_questions']} questions")
            print(f"Retries: {stats['total_retries']} ({stats['retry_rate']:.1%})")
            print(f"Successful retries: {stats['successful_retries']}")


def build_agentic_orchestrator(
    db,
    embedding_fn,
    config: AgenticRAGConfig = None,
    **kwargs
) -> AgenticRAGOrchestrator:
    """Factory function to build an orchestrator with default settings.

    Args:
        db: ChromaDB instance
        embedding_fn: Embedding function
        config: Optional configuration
        **kwargs: Override config values

    Returns:
        Configured AgenticRAGOrchestrator
    """
    if config is None:
        config = AgenticRAGConfig(**kwargs)
    elif kwargs:
        valid_fields = {item.name for item in fields(AgenticRAGConfig)}
        unknown = sorted(set(kwargs) - valid_fields)
        if unknown:
            raise TypeError(f"Unknown AgenticRAGConfig fields: {unknown}")
        # ``replace`` creates a fresh dataclass and reruns ``__post_init__``;
        # mutating an existing instance would bypass validation.
        config = replace(config, **kwargs)

    return AgenticRAGOrchestrator(config=config, db=db, embedding_fn=embedding_fn)
