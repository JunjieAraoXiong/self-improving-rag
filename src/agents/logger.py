"""Centralized logging for all agent decisions."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from .base import AgentDecision


class AgentLogger:
    """Centralized logging for all agent decisions.

    This logger captures every decision made by every agent, enabling:
    - Post-hoc analysis of agent behavior
    - Identification of failure patterns
    - Comparison between standard and agentic RAG
    - Paper figures and tables generation

    Output formats:
    - JSON: Full decision details for programmatic analysis
    - CSV: Flattened format for easy spreadsheet analysis
    """

    def __init__(self, output_dir: str = "agent_logs"):
        """Initialize the agent logger.

        Args:
            output_dir: Directory to save log files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Store decisions organized by question
        self.questions: Dict[str, Dict[str, Any]] = {}
        self._current_question_id: Optional[str] = None

    def start_question(self, question_id: str, question: str, gold_answer: str = None) -> None:
        """Start logging for a new question.

        Args:
            question_id: Unique identifier for the question
            question: The question text
            gold_answer: The reference answer (if available)
        """
        self._current_question_id = question_id
        self.questions[question_id] = {
            "question_id": question_id,
            "question": question,
            "gold_answer": gold_answer,
            "attempts": [],
            "final_answer": None,
            "correct": None,
            "policy_accepted": None,
            "evaluation_mode": None,
            "evidence_manifest": {},
            "query_plan": {},
            "finance_question_spec": {},
            "finance_program": None,
            "finance_verification": None,
            "correction_history": [],
            "improvement_from_retry": None,
            "abstained": False,
            "error": None,
            "terminal_state": None,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "llm_calls": 0,
            "usage_by_model": {},
            "estimated_cost_usd": 0.0,
        }

    def start_attempt(self, attempt: int) -> None:
        """Start a new attempt for the current question.

        Args:
            attempt: The attempt number (0-indexed)
        """
        if self._current_question_id is None:
            raise ValueError("No question started. Call start_question() first.")

        self.questions[self._current_question_id]["attempts"].append({
            "attempt": attempt,
            "retrieval_agent": None,
            "reasoning_agent": None,
            "judge_agent": None,
            "evidence_manifest": {},
            "correction_plan": None,
        })

    def log_query_plan(self, query_plan: Dict[str, Any]) -> None:
        """Persist the compiled finance-query contract for the question."""

        if self._current_question_id is None:
            raise ValueError("No question started. Call start_question() first.")
        self.questions[self._current_question_id]["query_plan"] = query_plan

    def log_finance_question_spec(self, spec: Dict[str, Any]) -> None:
        """Persist the trusted pre-generation calculation contract."""

        if self._current_question_id is None:
            raise ValueError("No question started. Call start_question() first.")
        self.questions[self._current_question_id]["finance_question_spec"] = spec

    def log_correction_plan(self, correction_plan: Dict[str, Any]) -> None:
        """Persist one gap-specific controller action."""

        if self._current_question_id is None:
            raise ValueError("No question started. Call start_question() first.")
        question_log = self.questions[self._current_question_id]
        if not question_log["attempts"]:
            raise ValueError("No attempt started. Call start_attempt() first.")
        question_log["attempts"][-1]["correction_plan"] = correction_plan
        question_log["correction_history"].append(correction_plan)

    def log_evidence_manifest(
        self,
        evidence_manifest: Dict[str, Dict[str, Any]],
    ) -> None:
        """Persist the positional evidence mapping for the current attempt."""

        if self._current_question_id is None:
            raise ValueError("No question started. Call start_question() first.")
        question_log = self.questions[self._current_question_id]
        if not question_log["attempts"]:
            raise ValueError("No attempt started. Call start_attempt() first.")
        question_log["attempts"][-1]["evidence_manifest"] = evidence_manifest

    def log_decision(self, agent_name: str, decision: AgentDecision) -> None:
        """Log a decision from an agent.

        Args:
            agent_name: Name of the agent (retrieval_agent, reasoning_agent, judge_agent)
            decision: The agent's decision
        """
        if self._current_question_id is None:
            raise ValueError("No question started. Call start_question() first.")

        question_log = self.questions[self._current_question_id]
        if not question_log["attempts"]:
            raise ValueError("No attempt started. Call start_attempt() first.")

        # Store the decision in the current attempt
        current_attempt = question_log["attempts"][-1]
        current_attempt[agent_name] = decision.to_dict()

    def finish_question(
        self,
        final_answer: Optional[str],
        correct: Optional[bool],
        improvement_from_retry: bool = None,
        policy_accepted: Optional[bool] = None,
        evaluation_mode: Optional[str] = None,
        evidence_manifest: Optional[Dict[str, Dict[str, Any]]] = None,
        finance_question_spec: Optional[Dict[str, Any]] = None,
        finance_program: Optional[Dict[str, Any]] = None,
        finance_verification: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
        abstained: bool = False,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        llm_calls: int = 0,
        usage_by_model: Optional[Dict[str, Dict[str, Any]]] = None,
        estimated_cost_usd: float = 0.0,
    ) -> None:
        """Finish logging for the current question.

        Args:
            final_answer: The final answer after all attempts
            correct: Oracle/independent correctness, or None when not evaluated
            improvement_from_retry: Whether retry improved the answer
            policy_accepted: Whether the policy judge accepted the final answer
            evaluation_mode: ``oracle_guided`` or ``blind_policy``
            evidence_manifest: Doc-label mapping for the returned final answer
        """
        if self._current_question_id is None:
            raise ValueError("No question started.")

        question_log = self.questions[self._current_question_id]
        question_log["final_answer"] = final_answer
        question_log["correct"] = correct
        question_log["policy_accepted"] = policy_accepted
        question_log["evaluation_mode"] = evaluation_mode
        question_log["evidence_manifest"] = evidence_manifest or {}
        question_log["finance_question_spec"] = finance_question_spec or {}
        question_log["finance_program"] = finance_program
        question_log["finance_verification"] = finance_verification
        question_log["improvement_from_retry"] = improvement_from_retry
        question_log["abstained"] = bool(abstained)
        question_log["error"] = error
        question_log["prompt_tokens"] = int(prompt_tokens)
        question_log["completion_tokens"] = int(completion_tokens)
        question_log["llm_calls"] = int(llm_calls)
        question_log["usage_by_model"] = usage_by_model or {}
        question_log["estimated_cost_usd"] = float(estimated_cost_usd)
        if error:
            question_log["terminal_state"] = "error"
        elif abstained:
            question_log["terminal_state"] = "abstained"
        elif policy_accepted:
            question_log["terminal_state"] = "accepted"
        else:
            question_log["terminal_state"] = "not_accepted"

        self._current_question_id = None

    def export_json(self, filename: str = None) -> str:
        """Export all decisions to JSON.

        Args:
            filename: Optional custom filename (without extension)

        Returns:
            Path to the saved file
        """
        if filename is None:
            filename = f"agent_decisions_{self.session_id}"

        filepath = self.output_dir / f"{filename}.json"

        with open(filepath, 'w') as f:
            json.dump(list(self.questions.values()), f, indent=2)

        return str(filepath)

    def export_csv(self, filename: str = None) -> str:
        """Export decisions to CSV (flattened format).

        Each row represents one attempt on one question.

        Args:
            filename: Optional custom filename (without extension)

        Returns:
            Path to the saved file
        """
        if filename is None:
            filename = f"agent_decisions_{self.session_id}"

        filepath = self.output_dir / f"{filename}.csv"

        rows = []
        for question_log in self.questions.values():
            for attempt in question_log["attempts"]:
                row = {
                    "question_id": question_log["question_id"],
                    "question": question_log["question"][:200],  # Truncate for CSV
                    "gold_answer": question_log["gold_answer"][:200] if question_log["gold_answer"] else None,
                    "attempt": attempt["attempt"],
                    "final_answer": question_log["final_answer"][:200] if question_log["final_answer"] else None,
                    "correct": question_log["correct"],
                    "policy_accepted": question_log.get("policy_accepted"),
                    "evaluation_mode": question_log.get("evaluation_mode"),
                    "abstained": question_log.get("abstained", False),
                    "error": question_log.get("error"),
                    "terminal_state": question_log.get("terminal_state"),
                    "prompt_tokens": question_log.get("prompt_tokens", 0),
                    "completion_tokens": question_log.get("completion_tokens", 0),
                    "llm_calls": question_log.get("llm_calls", 0),
                    "usage_by_model": json.dumps(
                        question_log.get("usage_by_model", {}), sort_keys=True
                    ),
                    "estimated_cost_usd": question_log.get(
                        "estimated_cost_usd", 0.0
                    ),
                    "query_plan": json.dumps(
                        question_log.get("query_plan", {}), sort_keys=True
                    ),
                    "finance_question_spec": json.dumps(
                        question_log.get("finance_question_spec", {}), sort_keys=True
                    ),
                    "finance_program": json.dumps(
                        question_log.get("finance_program"), sort_keys=True
                    ),
                    "finance_verification": json.dumps(
                        question_log.get("finance_verification"), sort_keys=True
                    ),
                    "correction_plan": json.dumps(
                        attempt.get("correction_plan") or {}, sort_keys=True
                    ),
                    "correction_action": (
                        (attempt.get("correction_plan") or {}).get("action")
                    ),
                    "evidence_manifest": json.dumps(
                        attempt.get("evidence_manifest", {}), sort_keys=True
                    ),
                    "final_evidence_manifest": json.dumps(
                        question_log.get("evidence_manifest", {}), sort_keys=True
                    ),
                    "improvement_from_retry": question_log["improvement_from_retry"],
                    "total_attempts": len(question_log["attempts"]),
                }

                # Flatten agent decisions
                for agent in ["retrieval_agent", "reasoning_agent", "judge_agent"]:
                    if attempt.get(agent):
                        decision = attempt[agent]
                        row[f"{agent}_confidence"] = decision.get("confidence")
                        row[f"{agent}_reasoning"] = decision.get("reasoning", "")[:200]

                        # Flatten decision_value
                        if decision.get("decision_value"):
                            for key, value in decision["decision_value"].items():
                                if isinstance(value, (str, int, float, bool)):
                                    row[f"{agent}_{key}"] = value

                rows.append(row)

        df = pd.DataFrame(rows)
        df.to_csv(filepath, index=False)

        return str(filepath)

    def summary(self) -> Dict[str, Any]:
        """Generate summary statistics for the session.

        Returns:
            Dictionary with aggregate metrics
        """
        if not self.questions:
            return {"total_questions": 0}

        total_questions = len(self.questions)
        questions_with_retry = sum(
            1 for q in self.questions.values()
            if len(q["attempts"]) > 1
        )

        evaluated_questions = sum(
            1 for q in self.questions.values()
            if q.get("correct") is not None
        )
        correct_count = sum(
            1 for q in self.questions.values()
            if q.get("correct") is True
        )
        policy_decisions = [
            q.get("policy_accepted")
            for q in self.questions.values()
            if q.get("policy_accepted") is not None
        ]

        retry_improvements = sum(
            1 for q in self.questions.values()
            if q.get("improvement_from_retry")
        )

        total_attempts = sum(
            len(q["attempts"]) for q in self.questions.values()
        )

        # Collect judge scores
        judge_scores = []
        for q in self.questions.values():
            for attempt in q["attempts"]:
                if attempt.get("judge_agent") and attempt["judge_agent"].get("decision_value"):
                    score = attempt["judge_agent"]["decision_value"].get("score")
                    if score is not None:
                        judge_scores.append(score)

        return {
            "session_id": self.session_id,
            "total_questions": total_questions,
            "questions_with_retry": questions_with_retry,
            "retry_rate": questions_with_retry / total_questions if total_questions > 0 else 0,
            "evaluated_questions": evaluated_questions,
            "labeled_correctness_rate": (
                correct_count / evaluated_questions
                if evaluated_questions > 0
                else None
            ),
            "policy_acceptance_rate": (
                sum(policy_decisions) / len(policy_decisions)
                if policy_decisions
                else None
            ),
            "retry_success_rate": retry_improvements / questions_with_retry if questions_with_retry > 0 else 0,
            "policy_score_improvement_rate": retry_improvements / questions_with_retry if questions_with_retry > 0 else 0,
            "avg_attempts": total_attempts / total_questions if total_questions > 0 else 0,
            "avg_judge_score": sum(judge_scores) / len(judge_scores) if judge_scores else 0,
        }

    def print_summary(self) -> None:
        """Print a formatted summary to console."""
        stats = self.summary()

        print("\n" + "=" * 60)
        print("AGENTIC RAG SESSION SUMMARY")
        print("=" * 60)
        print(f"Session ID: {stats.get('session_id', 'N/A')}")
        print(f"Total questions: {stats['total_questions']}")
        print(f"Questions with retry: {stats.get('questions_with_retry', 0)} ({stats.get('retry_rate', 0):.1%})")
        labeled_correctness = stats.get("labeled_correctness_rate")
        acceptance_rate = stats.get("policy_acceptance_rate")
        print(
            f"Oracle-guided fixed-threshold correctness label: {labeled_correctness:.1%}"
            if labeled_correctness is not None
            else "Correctness: N/A (not independently evaluated)"
        )
        if acceptance_rate is not None:
            print(f"Policy acceptance rate: {acceptance_rate:.1%}")
        print(f"Policy-score improvement rate: {stats.get('policy_score_improvement_rate', 0):.1%}")
        print(f"Average attempts: {stats.get('avg_attempts', 0):.2f}")
        print(f"Average judge score: {stats.get('avg_judge_score', 0):.3f}")
        print("=" * 60 + "\n")
