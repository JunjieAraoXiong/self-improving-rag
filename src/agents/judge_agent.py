"""Judge Agent: Evaluates answers and decides whether to retry.

Includes a deterministic verification gate that runs BEFORE LLM-as-judge
to catch answers lacking proper evidence citations. In blind mode (no gold
answer), uses numeric grounding verification to check that all numbers in
the answer are supported by the retrieved documents.
"""

from typing import Any, Dict, List, Optional, Tuple

from langchain_core.documents import Document

from .base import AgentDecision, BaseAgent
from evaluation.deterministic_verify import (
    VerificationResult,
    deterministic_verify,
    format_verification_feedback,
)
from evaluation.numeric_check import (
    apply_context_scale,
    extract_numbers,
    find_numeric_match,
    infer_scale,
    is_likely_year,
    strip_evidence_citations,
)


class JudgeAgent(BaseAgent):
    """Agent C: Evaluates answer quality and decides whether to retry.

    This agent:
    1. Compares the generated answer to the gold answer (when available)
    2. Evaluates answer quality using LLM-as-judge
    3. Decides whether to trigger a retry based on score threshold
    4. Can adjust threshold on different attempts

    The agent provides interpretable decisions about:
    - Whether the answer is correct
    - Why the answer succeeded or failed
    - Whether a retry is warranted
    """

    def __init__(
        self,
        judge_model: str = None,
        retry_threshold: float = 0.5,
        min_threshold: float = 0.3,
        threshold_decay: float = 0.1,
        enable_deterministic_gate: bool = True,
        require_all_numbers_cited: bool = True,
    ):
        """Initialize the judge agent.

        Args:
            judge_model: Model to use for LLM-as-judge evaluation
            retry_threshold: Score below which to trigger retry (default 0.5)
            min_threshold: Minimum threshold even after escalation (default 0.3)
            threshold_decay: Per-attempt threshold reduction. Use zero for the
                v2 calibrated policy; ``0.1`` reproduces the paper policy.
            enable_deterministic_gate: Run deterministic verification before LLM judge
            require_all_numbers_cited: Require all numerical claims to have citations
        """
        super().__init__("JudgeAgent")

        from src.config import DEFAULTS
        self.judge_model = judge_model or DEFAULTS.judge_model
        self.retry_threshold = retry_threshold
        self.min_threshold = min_threshold
        self.threshold_decay = max(0.0, float(threshold_decay))
        self.enable_deterministic_gate = enable_deterministic_gate
        self.require_all_numbers_cited = require_all_numbers_cited

        # Track scores across attempts
        self._attempt_scores: list = []
        # Track verification results for retry feedback
        self._last_verification: Optional[VerificationResult] = None

    def evaluate(
        self,
        question: str,
        predicted_answer: str,
        gold_answer: str = None,
        docs: List[Document] = None,
    ) -> Tuple[float, str]:
        """Evaluate the predicted answer.

        In blind mode (no gold_answer), combines LLM self-evaluation with
        numeric grounding verification against source documents for a
        stronger quality signal.

        Args:
            question: The original question
            predicted_answer: The model's answer
            gold_answer: The reference answer (if available)
            docs: Retrieved source documents (used for blind numeric verification)

        Returns:
            Tuple of (score, justification)
        """
        if not predicted_answer:
            return 0.0, "Empty answer"

        if gold_answer:
            # Use LLM-as-judge with gold answer
            from evaluation.llm_judge import llm_as_judge
            score, justification = llm_as_judge(
                question=question,
                gold_answer=gold_answer,
                predicted_answer=predicted_answer,
                judge_model=self.judge_model,
            )
        else:
            # Blind evaluation: LLM self-eval + numeric grounding
            llm_score, llm_justification = self._self_evaluate(question, predicted_answer)

            # Augment with numeric grounding check against source documents
            if docs:
                grounding_score, grounding_explanation = self._blind_numeric_verify(
                    predicted_answer, docs, question=question
                )

                # Combine: weight numeric grounding heavily for financial QA
                # LLM self-eval captures coherence/relevance (weight: 0.4)
                # Numeric grounding captures factual accuracy (weight: 0.6)
                score = 0.4 * llm_score + 0.6 * grounding_score

                justification = (
                    f"[Blind eval] LLM: {llm_score:.2f} ({llm_justification}). "
                    f"Numeric grounding: {grounding_score:.2f} ({grounding_explanation})"
                )

                # Hard override: if most numbers are ungrounded, force low score
                if grounding_score < 0.3:
                    score = min(score, 0.25)
                    justification += " [OVERRIDE: majority of numbers ungrounded]"
            else:
                score = llm_score
                justification = llm_justification

        return score, justification

    def _self_evaluate(self, question: str, answer: str) -> Tuple[float, str]:
        """Self-evaluate answer quality without gold answer.

        Args:
            question: The question
            answer: The generated answer

        Returns:
            Tuple of (score, justification)
        """
        from src.providers import get_provider

        system_prompt = """You are evaluating response coverage, not factual correctness.
You do NOT have access to the correct answer. Do not reward confident language,
extra numbers, verbosity, or a willingness to guess. Evaluate only:
1. Requirement coverage: Does it address every explicit part of the question?
2. Relevance: Is each part responsive to the requested entity, metric, and period?
3. Clarity: Does it distinguish reported facts, calculations, and uncertainty?
4. Selectivity: If required evidence is missing, does it identify the precise gap
   instead of inventing a conclusion?

Score from 0.0 to 1.0:
- 1.0: All explicit requirements are addressed clearly
- 0.7: Most requirements are addressed with minor omissions
- 0.5: Partially addresses the requested work
- 0.3: Major requirements are omitted or the response is mostly off-topic
- 0.0: Empty, unrelated, or fabricated-looking response

This score is not a confidence estimate and must not be described as accuracy.

Respond with:
SCORE: <score>
JUSTIFICATION: <explanation>"""

        user_prompt = f"""Question: {question}

Answer: {answer}

Evaluate the quality of this answer."""

        provider = get_provider(self.judge_model)
        response = provider.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tokens=200,
            temperature=0.0,
        )

        # Provider failures must remain infrastructure errors rather than being
        # converted into a model-quality score of zero.
        from evaluation.llm_judge import parse_judge_response
        return parse_judge_response(response.content)

    def _blind_numeric_verify(
        self,
        answer: str,
        docs: List[Document],
        question: str = "",
        relative_tolerance: float = 0.05,
    ) -> Tuple[float, str]:
        """Verify that numbers in the answer are grounded in source documents.

        This is the key blind verification signal: without gold answers, we check
        that every significant number in the answer appears (within tolerance) in
        at least one retrieved document. Ungrounded numbers are likely hallucinated.

        Args:
            answer: The generated answer
            docs: Retrieved source documents
            relative_tolerance: Acceptable relative difference (default 5%)

        Returns:
            Tuple of (grounding_score 0-1, explanation)
        """
        if not answer or not docs:
            return 0.5, "No answer or documents to verify"

        # Extract numbers from answer (high-confidence only)
        # Evidence quotes are not answer claims. Including their numbers lets a
        # wrong answer appear grounded merely because its citation contains the
        # correct value.
        answer_nums = extract_numbers(strip_evidence_citations(answer))
        # Skip standalone years and small untyped ordinals, but retain values
        # such as 0.5%, $0.8M, and 0.7x.
        significant_nums = [
            n for n in answer_nums
            if n.confidence >= 0.5
            and not (1900 <= n.value <= 2100 and n.unit == "")  # Skip standalone years
            and (abs(n.value) > 1 or bool(n.unit) or bool(n.currency))
        ]

        if not significant_nums:
            return 0.7, "No significant numeric claims to verify"

        # Extract numbers from all source documents
        source_nums = []
        for doc in docs:
            doc_numbers = extract_numbers(doc.page_content)
            source_nums.extend(doc_numbers)
            scale = infer_scale(doc.page_content)
            if scale:
                source_nums.extend(
                    apply_context_scale(number, scale)
                    for number in doc_numbers
                    if not number.explicit_scale and not is_likely_year(number)
                )

        if not source_nums:
            # Documents have no numbers but answer does -- suspicious
            if significant_nums:
                return 0.2, f"Answer contains {len(significant_nums)} numbers but sources contain none"
            return 0.5, "Neither answer nor sources contain numbers"

        # Check each answer number against source numbers
        grounded = []
        ungrounded = []

        sign_insensitive = any(
            term in (question or answer).lower()
            for term in ("capex", "capital expenditure", "cash paid", "outflow")
        )
        for ans_num in significant_nums:
            if find_numeric_match(
                ans_num,
                source_nums,
                relative_tolerance=relative_tolerance,
                allow_absolute_value=sign_insensitive,
            ):
                grounded.append(ans_num)
            else:
                ungrounded.append(ans_num)

        # Calculate grounding ratio
        total = len(significant_nums)
        grounded_count = len(grounded)
        grounding_ratio = grounded_count / total if total > 0 else 1.0

        # Build explanation
        if ungrounded:
            ungrounded_strs = [f"{n.raw} ({n.value:,.2f})" for n in ungrounded[:3]]
            explanation = (
                f"Numeric grounding: {grounded_count}/{total} numbers verified in sources. "
                f"Ungrounded: {', '.join(ungrounded_strs)}"
            )
        else:
            explanation = f"Numeric grounding: all {total} numbers verified in sources"

        return grounding_ratio, explanation

    def run_deterministic_verification(
        self,
        answer: str,
        docs: List[Document],
        question: str = "",
    ) -> VerificationResult:
        """Run deterministic verification as a hard gate before LLM judge.

        This checks that all numerical claims have proper [DocX: 'quote'] citations.

        Args:
            answer: The generated answer
            docs: Source documents used for the answer

        Returns:
            VerificationResult with pass/fail and details
        """
        result = deterministic_verify(
            answer=answer,
            docs=docs,
            require_all_numbers_cited=self.require_all_numbers_cited,
            question=question,
        )
        self._last_verification = result
        return result

    def get_verification_feedback(self) -> str:
        """Get feedback from last verification for retry prompt.

        Returns:
            Formatted feedback string or empty if passed
        """
        if self._last_verification is None:
            return ""
        return format_verification_feedback(self._last_verification)

    def should_retry(self, score: float, attempt: int, max_retries: int) -> bool:
        """Decide whether to retry based on score and attempt number.

        Args:
            score: The evaluation score
            attempt: Current attempt number
            max_retries: Maximum allowed retries

        Returns:
            True if should retry, False otherwise
        """
        # Don't retry if we've reached max attempts
        if attempt >= max_retries:
            return False

        adjusted_threshold = self.acceptance_threshold(attempt)

        # Check if score is below threshold
        if score < adjusted_threshold:
            return True

        return False

    def acceptance_threshold(self, attempt: int) -> float:
        """Return the policy threshold used for a specific attempt."""

        return max(
            self.min_threshold,
            self.retry_threshold - (attempt * self.threshold_decay),
        )

    def decide(self, context: Dict[str, Any]) -> AgentDecision:
        """Evaluate the answer and decide whether to retry.

        Includes a deterministic verification gate that runs BEFORE LLM judge.
        If verification fails, triggers retry without calling the LLM judge.
        If max retries reached and still failing verification, returns abstain.

        Args:
            context: Must contain 'question', 'predicted_answer'.
                     Optional: 'gold_answer', 'attempt', 'max_retries', 'documents'

        Returns:
            AgentDecision with evaluation and retry decision
        """
        question = context["question"]
        predicted_answer = context.get("predicted_answer", "")
        gold_answer = context.get("gold_answer")
        attempt = context.get("attempt", self._attempt)
        max_retries = context.get("max_retries", 1)
        docs = context.get("documents", [])
        require_finance_program = bool(context.get("require_finance_program"))
        finance_program = context.get("finance_program")
        finance_program_issues = list(context.get("finance_program_issues") or ())
        finance_question_spec = context.get("finance_question_spec")

        if require_finance_program:
            from src.finance_program import verify_program

            if finance_program_issues:
                issue_payloads = finance_program_issues
                program_verification = {
                    "passed": False,
                    "fully_verified": False,
                    "assurance_level": "unverified",
                    "issues": issue_payloads,
                    "execution": None,
                    "rendered_answer": None,
                    "evidence_coverage": 0.0,
                }
            elif finance_program is None:
                issue_payloads = [
                    {
                        "code": "missing_program",
                        "message": "A typed finance program is required for this calculation",
                    }
                ]
                program_verification = {
                    "passed": False,
                    "fully_verified": False,
                    "assurance_level": "unverified",
                    "issues": issue_payloads,
                    "execution": None,
                    "rendered_answer": None,
                    "evidence_coverage": 0.0,
                }
            elif finance_question_spec is None:
                issue_payloads = [
                    {
                        "code": "unsupported_claim",
                        "message": "The calculation has no fully resolved pre-generation contract",
                    }
                ]
                program_verification = {
                    "passed": False,
                    "fully_verified": False,
                    "assurance_level": "unverified",
                    "issues": issue_payloads,
                    "execution": None,
                    "rendered_answer": None,
                    "evidence_coverage": 0.0,
                }
            else:
                result = verify_program(
                    finance_program,
                    docs,
                    question,
                    question_spec=finance_question_spec,
                    answer_text=predicted_answer,
                    require_full_contract=True,
                )
                program_verification = result.to_dict()
                issue_payloads = program_verification["issues"]

            reason_codes = [issue["code"] for issue in issue_payloads]
            if program_verification.get("fully_verified"):
                score = float(program_verification.get("evidence_coverage", 1.0))
                self._attempt_scores.append(score)
                decision = AgentDecision(
                    agent_name=self.name,
                    decision_type="typed_finance_verification",
                    decision_value={
                        "score": score,
                        "pass": True,
                        "retry": False,
                        "abstain": False,
                        "justification": (
                            "Typed program, trusted question contract, every operand, "
                            "arithmetic trace, and rendered answer all verified."
                        ),
                        "verification_passed": True,
                        "verification_feedback": "",
                        "verification_reason_codes": [],
                        "verification_issues": [],
                        "finance_program_verification": program_verification,
                        "canonical_answer": program_verification.get(
                            "rendered_answer"
                        ),
                        "acceptance_threshold": self.acceptance_threshold(attempt),
                    },
                    confidence=score,
                    reasoning="Accepted by full-contract typed finance verification.",
                    metadata={
                        "attempt": attempt,
                        "accepted_by": "typed_finance_full_contract",
                        "has_gold_answer": False,
                        "finance_program_verification": program_verification,
                        "verification_reason_codes": [],
                    },
                )
                self.log_decision(decision)
                return decision

            feedback = "; ".join(
                issue.get("message", issue.get("code", "verification failed"))
                for issue in issue_payloads
            )
            canonical_answer = program_verification.get("rendered_answer")
            locally_repairable = bool(canonical_answer) and set(reason_codes) <= {
                "answer_result_mismatch",
                "result_value_mismatch",
            }
            if locally_repairable:
                from src.finance_program import repair_program_result

                repaired_program, repaired_result = repair_program_result(
                    finance_program,
                    docs,
                    question,
                    question_spec=finance_question_spec,
                    answer_text=predicted_answer,
                )
                repaired_verification = repaired_result.to_dict()
                if repaired_program is None or not repaired_result.fully_verified:
                    locally_repairable = False

            if locally_repairable:
                decision = AgentDecision(
                    agent_name=self.name,
                    decision_type="typed_finance_local_repair",
                    decision_value={
                        "score": float(
                            program_verification.get("evidence_coverage", 0.0)
                        ),
                        "pass": False,
                        "retry": True,
                        "abstain": False,
                        "justification": feedback,
                        "verification_passed": False,
                        "verification_feedback": feedback,
                        "verification_reason_codes": reason_codes,
                        "verification_issues": issue_payloads,
                        "finance_program_verification": program_verification,
                        "canonical_answer": repaired_result.rendered_answer,
                        "repaired_finance_program": repaired_program.model_dump(
                            mode="json"
                        ),
                        "repaired_finance_program_verification": (
                            repaired_verification
                        ),
                    },
                    confidence=float(
                        program_verification.get("evidence_coverage", 0.0)
                    ),
                    reasoning="A deterministic local result can repair the generated display/value.",
                    metadata={
                        "attempt": attempt,
                        "deterministic_gate_triggered": True,
                        "finance_program_verification": program_verification,
                        "verification_reason_codes": reason_codes,
                    },
                )
                self.log_decision(decision)
                return decision

            reason = f"Typed finance verification failed: {feedback}"
            if attempt >= max_retries:
                return self._create_abstain_decision(
                    attempt=attempt,
                    reason=reason,
                    reason_codes=reason_codes,
                    verification_issues=issue_payloads,
                    finance_program_verification=program_verification,
                )
            return self._create_retry_decision(
                attempt=attempt,
                reason=reason,
                verification_feedback=feedback,
                reason_codes=reason_codes,
                verification_issues=issue_payloads,
                finance_program_verification=program_verification,
            )

        # DETERMINISTIC GATE: Run verification before LLM judge
        verification_passed = True
        verification_message = ""

        if self.enable_deterministic_gate and docs:
            verification_result = self.run_deterministic_verification(
                predicted_answer, docs, question=question
            )
            verification_passed = verification_result.passed
            verification_message = verification_result.message

            if not verification_passed:
                # Deterministic check failed - decide based on retry budget
                if attempt >= max_retries:
                    # Max retries reached, ABSTAIN
                    return self._create_abstain_decision(
                        attempt=attempt,
                        reason=f"Deterministic verification failed after {attempt + 1} attempts: {verification_message}",
                        reason_codes=verification_result.reason_codes,
                    )
                else:
                    # Trigger retry without calling LLM judge
                    return self._create_retry_decision(
                        attempt=attempt,
                        reason=f"Deterministic verification failed: {verification_message}",
                        verification_feedback=self.get_verification_feedback(),
                        reason_codes=verification_result.reason_codes,
                    )

        # Deterministic check passed - proceed to LLM-as-judge evaluation
        # Pass docs for blind numeric verification when no gold answer
        score, justification = self.evaluate(
            question, predicted_answer, gold_answer, docs=docs
        )

        # Compute improvement against the previous attempt before appending the
        # current score. Improvement is diagnostic only; it cannot bypass tau.
        previous_score = self._attempt_scores[-1] if self._attempt_scores else None
        adjusted_threshold = self.acceptance_threshold(attempt)
        significant_improvement = (
            previous_score is not None and score >= previous_score + 0.2
        )
        should_retry = self.should_retry(score, attempt, max_retries)
        self._attempt_scores.append(score)

        # Stopping because the retry budget is exhausted is not acceptance.
        # Match the documented policy exactly: U_t must meet tau_t.
        passed = score >= adjusted_threshold

        # Build reasoning
        if should_retry:
            reasoning = (
                f"Score {score:.2f} below threshold. "
                f"Triggering retry. {justification}"
            )
        elif passed:
            reasoning = (
                f"Answer accepted with score {score:.2f} at threshold "
                f"{adjusted_threshold:.2f}. {justification}"
            )
        else:
            reasoning = (
                f"Answer failed with score {score:.2f}, "
                f"but max retries reached. {justification}"
            )

        decision = AgentDecision(
            agent_name=self.name,
            decision_type="evaluation",
            decision_value={
                "score": score,
                "pass": passed,
                "retry": should_retry,
                "abstain": False,
                "justification": justification,
                "verification_passed": verification_passed,
                "verification_feedback": justification if should_retry else "",
                "acceptance_threshold": adjusted_threshold,
            },
            confidence=score,  # Use score as confidence
            reasoning=reasoning,
            metadata={
                "attempt": attempt,
                "threshold": adjusted_threshold,
                "base_threshold": self.retry_threshold,
                "threshold_decay": self.threshold_decay,
                "accepted_by": (
                    "threshold" if score >= adjusted_threshold else "rejected"
                ),
                "significant_improvement": significant_improvement,
                "has_gold_answer": gold_answer is not None,
                "attempt_scores": self._attempt_scores.copy(),
                "verification_message": verification_message,
                "verification_reason_codes": (
                    self._last_verification.reason_codes
                    if self._last_verification is not None
                    else []
                ),
            }
        )

        self.log_decision(decision)
        return decision

    def _create_retry_decision(
        self,
        attempt: int,
        reason: str,
        verification_feedback: str = "",
        reason_codes: Optional[List[str]] = None,
        verification_issues: Optional[List[Dict[str, Any]]] = None,
        finance_program_verification: Optional[Dict[str, Any]] = None,
    ) -> AgentDecision:
        """Create a decision to retry due to verification failure.

        Args:
            attempt: Current attempt number
            reason: Reason for retry
            verification_feedback: Feedback to include in retry prompt

        Returns:
            AgentDecision indicating retry needed
        """
        decision = AgentDecision(
            agent_name=self.name,
            decision_type="evaluation",
            decision_value={
                "score": 0.0,
                "pass": False,
                "retry": True,
                "abstain": False,
                "justification": reason,
                "verification_passed": False,
                "verification_feedback": verification_feedback,
                "verification_reason_codes": list(reason_codes or ()),
                "verification_issues": list(verification_issues or ()),
                "finance_program_verification": finance_program_verification,
            },
            confidence=0.0,
            reasoning=f"Retry triggered: {reason}",
            metadata={
                "attempt": attempt,
                "deterministic_gate_triggered": True,
                "verification_reason_codes": list(reason_codes or ()),
                "finance_program_verification": finance_program_verification,
            }
        )
        self.log_decision(decision)
        return decision

    def _create_abstain_decision(
        self,
        attempt: int,
        reason: str,
        reason_codes: Optional[List[str]] = None,
        verification_issues: Optional[List[Dict[str, Any]]] = None,
        finance_program_verification: Optional[Dict[str, Any]] = None,
    ) -> AgentDecision:
        """Create a decision to abstain due to insufficient evidence.

        This is returned when max retries are reached but the deterministic
        verification still fails, indicating the answer cannot be grounded
        in the available documents.

        Args:
            attempt: Current attempt number
            reason: Reason for abstention

        Returns:
            AgentDecision indicating abstention
        """
        decision = AgentDecision(
            agent_name=self.name,
            decision_type="evaluation",
            decision_value={
                "score": 0.0,
                "pass": False,
                "retry": False,
                "abstain": True,
                "justification": "Insufficient evidence in retrieved corpus",
                "verification_passed": False,
                "verification_reason_codes": list(reason_codes or ()),
                "verification_issues": list(verification_issues or ()),
                "finance_program_verification": finance_program_verification,
            },
            confidence=0.0,
            reasoning=f"ABSTAIN: {reason}",
            metadata={
                "attempt": attempt,
                "abstain_reason": reason,
                "deterministic_gate_triggered": True,
                "verification_reason_codes": list(reason_codes or ()),
                "finance_program_verification": finance_program_verification,
            }
        )
        self.log_decision(decision)
        return decision

    def reset(self) -> None:
        """Reset agent state for a new question."""
        super().reset()
        self._attempt_scores = []
        self._last_verification = None
