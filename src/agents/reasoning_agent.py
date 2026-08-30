"""Reasoning Agent: Generates answers from retrieved documents."""

import json
from typing import Any, Dict, List, Optional

from langchain_core.documents import Document

from .base import AgentDecision, BaseAgent


# Evidence-first citation format requirement
EVIDENCE_FIRST_REQUIREMENT = """
CRITICAL - EVIDENCE-FIRST FORMAT:
For EVERY numerical claim, financial metric, or factual assertion, you MUST include an inline citation
with the exact source quote in this format:
  "[DocX: 'exact quote from document']"

Examples:
- "Revenue was $383.3B [Doc2: 'Net sales were $383,285 million']"
- "The gross margin was 43.3% [Doc1: 'Gross margin: 43.3%']"
- "Apple had 164,000 employees [Doc3: 'The Company had approximately 164,000 full-time equivalent employees']"

If you cannot find a supporting quote for a claim, DO NOT make that claim.
Every number in your answer must have a corresponding [DocX: 'quote'] citation.
"""

FINANCE_PROGRAM_REQUIREMENT = """
TYPED CALCULATION REQUIRED:
Return a dedicated visible answer value followed by exactly one strict JSON
block enclosed in <finance_program>...</finance_program>. Do not put prose,
citations, or equations in the visible answer; it must be only the final value
in the requested format (for example, `0.96x` or `10.0%`).

The trusted contract below was compiled before generation. Copy its `entity`,
`period`, `metric`, output fields, operand IDs, and `expression` exactly. Supply
one operand for every contract operand ID and no others. For each operand,
provide decimal `value` as a JSON string plus its currency, scale, unit, entity,
period, metric, and an exact evidence object with `doc_id`, `quote`,
`value_text`, `metric_label`, `period_label`, and optional row/column labels.
The exact quote must occur in that Doc and bind the selected value to the named
metric and period. Never execute code or alter the formula.

FinanceProgram JSON shape:
{
  "schema_version": "1.0",
  "answer": {
    "value": "<rounded decimal>", "currency": "<ISO code if money>",
    "scale": "one|thousand|million|billion|trillion",
    "unit": "money|number|count|shares|ratio|percent|basis_points|days",
    "entity": "<contract entity>", "period": "<contract period>",
    "metric": "<contract metric>",
    "rounding": {"places": 2, "mode": "half_up"}
  },
  "operands": [
    {
      "id": "<contract operand id>", "value": "<decimal>",
      "currency": "<ISO code if money>", "scale": "<display scale>",
      "unit": "<typed unit>", "entity": "<contract operand entity>",
      "period": "<contract operand period>",
      "metric": "<contract operand metric>",
      "evidence": {
        "doc_id": "Doc1", "quote": "<exact source substring>",
        "value_text": "<exact numeric token>",
        "metric_label": "<exact metric label>",
        "period_label": "<exact period/date label>", "occurrence": 1
      }
    }
  ],
  "expression": {"op": "<copy the trusted expression exactly>", "args": []}
}
"""

TYPED_CALCULATION_EVIDENCE_REQUIREMENT = """
CALCULATION-SPECIFIC OUTPUT RULE:
Do not place inline [DocX: quote] citations, prose, a plan, or an equation in
the dedicated visible answer. It must contain only the final formatted value.
Put provenance exclusively in each operand's `evidence` object inside the
strict <finance_program> JSON block. Every operand still requires an exact
source quote and value binding; the visible value itself is recomputed and
checked locally.
"""

TYPED_CALCULATION_INSTRUCTION = """
Use the retrieved context to fill the trusted finance contract. Select exactly
one evidence-bound value for every required operand, preserve the compiled
formula and output contract, and return only the dedicated visible value plus
one strict <finance_program> JSON block. If a required operand is absent or
ambiguous, do not invent it; return no program so the controller can diagnose
the missing evidence.
"""

# Answer generation prompts that can be varied on retry
PROMPT_STRATEGIES = {
    "standard": {
        "system": """You are a precise financial analysis assistant who approaches every question methodically.
ALWAYS enter PLAN MODE before answering: first analyze what information is needed,
identify relevant data points in the context, then formulate your answer.
Be accurate with numbers, dates, and company names.
Only answer claims that the supplied evidence supports. If a required fact is
missing, name the missing fact precisely so the controller can retrieve it; if
the retry budget is exhausted, abstain rather than guess.

""" + EVIDENCE_FIRST_REQUIREMENT,
        "instruction": """Answer the following question using the information from the provided context.

PLAN MODE REQUIRED - Before answering, you MUST:
1. IDENTIFY: What specific information does this question ask for?
2. LOCATE: Find the relevant data points in the context
3. VERIFY: Check that data matches the correct company, time period, and fiscal year
4. CALCULATE: If math is needed, show your work step-by-step
5. CITE: Include [DocX: 'exact quote'] for every numerical claim
6. ANSWER: Provide your final answer with inline citations

IMPORTANT:
- Use precise numbers, dates, and company names from the context
- Every directly reported number MUST have a [DocX: 'quote'] citation
- Do not invent an answer when the entity, period, metric, or operand is absent
- When evidence is insufficient, state exactly what is missing
"""
    },
    "conservative": {
        "system": """You are a careful financial analyst who prioritizes accuracy over confidence.
When evidence is ambiguous, acknowledge uncertainty rather than guessing.
For yes/no or categorical questions, you may answer "maybe" or "uncertain" if the evidence is mixed.
Always cite specific passages that support your answer.

""" + EVIDENCE_FIRST_REQUIREMENT,
        "instruction": """Answer the following question based on the provided context.

Before answering:
1. Identify ALL relevant passages in the context
2. Evaluate whether the evidence clearly supports an answer
3. If evidence is mixed or incomplete, express appropriate uncertainty
4. Include [DocX: 'exact quote'] citations for all factual claims

For yes/no questions:
- Answer "yes" only if evidence strongly supports it
- Answer "no" only if evidence strongly contradicts it
- Answer "maybe" if evidence is ambiguous or insufficient

Always cite the specific passages that informed your answer using [DocX: 'quote'] format.
"""
    },
    "detailed": {
        "system": """You are a thorough financial analyst who provides comprehensive answers.
Your answers should include relevant context, supporting evidence, and careful reasoning.
Always reference specific documents and data points from the provided context.

""" + EVIDENCE_FIRST_REQUIREMENT,
        "instruction": """Provide a detailed answer to the following question using the context.

Your answer should:
1. Directly address the question
2. Include specific numbers, dates, and facts from the context
3. Cite each claim with [DocX: 'exact quote'] format
4. Note any relevant caveats or limitations

Be thorough but focused on what the question actually asks.
Every numerical claim must have a supporting citation.
"""
    }
}


class ReasoningAgent(BaseAgent):
    """Agent B: Generates answers from retrieved context.

    This agent:
    1. Takes retrieved documents as context
    2. Selects an appropriate prompting strategy
    3. Generates an answer with citations
    4. Can adjust its approach on retry (more conservative, more detailed, etc.)

    The agent tracks:
    - Answer content
    - Confidence in the answer
    - Which documents were cited
    """

    def __init__(
        self,
        model_name: str = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        disable_escalation: bool = False,
        generation_seed: Optional[int] = None,
    ):
        """Initialize the reasoning agent.

        Args:
            model_name: LLM model to use for generation
            temperature: Generation temperature
            max_tokens: Maximum tokens for response
            disable_escalation: Ablation flag - always use "standard" prompt
            generation_seed: Best-effort provider request seed, when supported
        """
        super().__init__("ReasoningAgent")

        from src.config import DEFAULTS
        self.model_name = model_name or DEFAULTS.llm_model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.disable_escalation = disable_escalation
        self.generation_seed = generation_seed

        # Lazy-loaded provider
        self._provider = None

        # Strategy progression for retries
        self._prompt_strategy = "standard"

    @property
    def provider(self):
        """Lazy-load the LLM provider."""
        if self._provider is None:
            from src.providers import get_provider
            self._provider = get_provider(self.model_name)
        return self._provider

    def format_context(self, docs: List[Document]) -> str:
        """Format retrieved documents into context string.

        Args:
            docs: List of retrieved documents

        Returns:
            Formatted context string with source attribution
        """
        context_parts = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("source", "Unknown")
            page = doc.metadata.get("page", "N/A")
            context_parts.append(f"[Document {i}] (Source: {source}, Page: {page})\n{doc.page_content}")

        return "\n\n".join(context_parts)

    def extract_citations(self, answer: str, docs: List[Document]) -> List[str]:
        """Extract which documents were likely cited in the answer.

        Args:
            answer: The generated answer
            docs: The context documents

        Returns:
            List of document sources that appear to be cited
        """
        citations = []
        for doc in docs:
            source = doc.metadata.get("source", "")
            # Check if any significant overlap between doc content and answer
            # (Simple heuristic - could be improved with more sophisticated matching)
            doc_words = set(doc.page_content.lower().split())
            answer_words = set(answer.lower().split())
            overlap = len(doc_words & answer_words) / max(len(doc_words), 1)
            if overlap > 0.1:  # At least 10% word overlap
                citations.append(source)

        return list(set(citations))

    def estimate_confidence(self, answer: str, docs: List[Document]) -> float:
        """Return an evidence-support score, not a correctness probability.

        Fluency, specificity, and willingness to answer are not evidence. The
        former heuristic rewarded numbers and penalized uncertainty, which is
        unsafe for financial QA. This diagnostic score is therefore limited to
        the fraction of explicit Doc citations whose quotes occur in the cited
        document. The Judge remains responsible for claim-level verification.

        Args:
            answer: The generated answer
            docs: The context documents

        Returns:
            Confidence score between 0 and 1
        """

        from evaluation.deterministic_verify import (
            extract_evidence_citations,
            verify_quote_in_docs,
        )

        citations = extract_evidence_citations(answer or "")
        if not citations:
            return 0.0
        valid = sum(
            verify_quote_in_docs(
                citation.quote,
                docs,
                doc_index=citation.doc_number - 1,
            )
            for citation in citations
        )
        return valid / len(citations)

    def get_prompt_strategy(self, attempt: int) -> str:
        """Get prompt strategy based on attempt number.

        Args:
            attempt: The attempt number

        Returns:
            Strategy name to use
        """
        # Ablation: disable escalation - always use "standard" prompt
        if self.disable_escalation:
            return "standard"

        strategies = ["standard", "conservative", "detailed"]
        return strategies[min(attempt, len(strategies) - 1)]

    def decide(self, context: Dict[str, Any]) -> AgentDecision:
        """Generate an answer based on question and retrieved documents.

        Args:
            context: Must contain 'question' and 'documents' keys

        Returns:
            AgentDecision with the generated answer
        """
        question = context["question"]
        docs = context["documents"]
        attempt = context.get("attempt", self._attempt)
        previous_answer = context.get("previous_answer")
        retry_feedback = context.get("retry_feedback", "")
        query_plan = context.get("query_plan") or {}
        correction_plan = context.get("correction_plan") or {}
        finance_question_spec = context.get("finance_question_spec") or {}
        require_finance_program = bool(context.get("require_finance_program"))

        # Select a prompt by diagnosed failure rather than retry count. The
        # historical count-based progression remains the no-plan fallback.
        correction_action = correction_plan.get("action")
        strategy_by_action = {
            "reuse_evidence_regenerate": "conservative",
            "targeted_retrieval": "standard",
            "replan": "detailed",
            "reconcile": "detailed",
            "paper_fixed_retry": self.get_prompt_strategy(attempt),
        }
        strategy_name = (
            "standard"
            if self.disable_escalation
            else strategy_by_action.get(
                correction_action,
                self.get_prompt_strategy(attempt),
            )
        )
        strategy = PROMPT_STRATEGIES[strategy_name]

        # Format context
        formatted_context = self.format_context(docs)

        # Build prompt
        system_prompt = strategy["system"]
        instruction = strategy["instruction"]
        if require_finance_program:
            # The general evidence-first prompt requires inline citations for
            # every number, which conflicts with the strict typed wire format.
            # Calculations bind provenance inside operand objects instead.
            system_prompt = system_prompt.replace(
                EVIDENCE_FIRST_REQUIREMENT,
                TYPED_CALCULATION_EVIDENCE_REQUIREMENT,
            )
            instruction = TYPED_CALCULATION_INSTRUCTION
        plan_context = ""
        if query_plan:
            compact_plan = {
                "task_type": query_plan.get("task_type"),
                "output": query_plan.get("output"),
                "formula_id": query_plan.get("formula_id"),
                "formula_hint": query_plan.get("formula_hint"),
                "answer_metric": query_plan.get("answer_metric"),
                "evidence_needs": query_plan.get("evidence_needs", []),
                "unresolved_constraints": query_plan.get(
                    "unresolved_constraints", []
                ),
            }
            plan_context = (
                "\n\nMACHINE-COMPILED QUERY CONTRACT (planning metadata, NOT evidence):\n"
                + json.dumps(compact_plan, sort_keys=True)
                + "\nUse retrieved documents—not this contract—as factual support."
            )
        finance_program_context = ""
        if require_finance_program:
            finance_program_context = (
                "\n\n"
                + FINANCE_PROGRAM_REQUIREMENT
                + "\nTRUSTED FINANCE QUESTION CONTRACT (metadata, NOT evidence):\n"
                + json.dumps(finance_question_spec, sort_keys=True)
            )
        correction_context = ""
        if attempt > 0 and (previous_answer or retry_feedback):
            correction_context = """

SELF-CORRECTION INPUT (this is diagnostic feedback, not a reference answer):
Previous attempt:
{previous_answer}

Verifier/Judge feedback:
{retry_feedback}

Revise the answer to address the diagnosed problem. Re-check the cited source,
number, unit, sign, entity, and period instead of merely rephrasing the answer.
Controller action: {correction_action}
Affected evidence needs: {affected_need_ids}
""".format(
                previous_answer=(previous_answer or "(unavailable)")[:3000],
                retry_feedback=(retry_feedback or "(no detailed feedback)")[:2000],
                correction_action=correction_action or "unspecified",
                affected_need_ids=", ".join(
                    correction_plan.get("affected_need_ids", [])
                ) or "none",
            )

        user_prompt = f"""{instruction}{plan_context}{finance_program_context}{correction_context}

Context:
{formatted_context}

Question: {question}

Answer:"""

        # Generate answer
        try:
            response = self.provider.generate(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                seed=self.generation_seed,
            )
            raw_answer = response.content
            error = None
        except Exception as e:
            raw_answer = None
            error = str(e)

        finance_program = None
        finance_program_issues = []
        if raw_answer is not None and require_finance_program:
            from src.finance_program import parse_finance_response

            parsed = parse_finance_response(raw_answer, require_program=True)
            answer = parsed.answer_text
            finance_program = (
                parsed.program.model_dump(mode="json")
                if parsed.program is not None
                else None
            )
            finance_program_issues = [issue.to_dict() for issue in parsed.issues]
        else:
            answer = raw_answer

        # Extract citations and estimate confidence
        if answer:
            citations = self.extract_citations(answer, docs)
            confidence = self.estimate_confidence(answer, docs)
        else:
            citations = []
            confidence = 0.0

        # Build reasoning
        reasoning = (
            f"Generated answer using '{strategy_name}' strategy. "
            f"Context: {len(docs)} documents. "
            f"Citations: {len(citations)} documents referenced."
        )
        if attempt > 0:
            reasoning = f"Retry #{attempt}: {reasoning}"

        decision = AgentDecision(
            agent_name=self.name,
            decision_type="answer_generation",
            decision_value={
                "answer": answer,
                "citations": citations,
                "strategy": strategy_name,
                "error": error,
                "feedback_used": bool(correction_context),
                "correction_action": correction_action,
                "finance_program": finance_program,
                "finance_program_issues": finance_program_issues,
            },
            confidence=confidence,
            reasoning=reasoning,
            metadata={
                "attempt": attempt,
                "num_docs": len(docs),
                "model": self.model_name,
                "retry_feedback_present": bool(retry_feedback),
                "query_plan_present": bool(query_plan),
                "confidence_semantics": "exact_citation_support_not_correctness",
                "finance_program_required": require_finance_program,
                "finance_program_parsed": finance_program is not None,
            }
        )

        self.log_decision(decision)
        return decision

    def escalate_strategy(self) -> None:
        """Move to a different prompt strategy for retry."""
        self._attempt += 1
