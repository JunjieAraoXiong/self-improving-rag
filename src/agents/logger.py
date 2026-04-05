"""Centralized logging for all agent decisions."""

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

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
            "improvement_from_retry": None,
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
        })

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
        final_answer: str,
        correct: bool,
        improvement_from_retry: bool = None
    ) -> None:
        """Finish logging for the current question.

        Args:
            final_answer: The final answer after all attempts
            correct: Whether the final answer is correct
            improvement_from_retry: Whether retry improved the answer
        """
        if self._current_question_id is None:
            raise ValueError("No question started.")

        question_log = self.questions[self._current_question_id]
        question_log["final_answer"] = final_answer
        question_log["correct"] = correct
        question_log["improvement_from_retry"] = improvement_from_retry

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

        correct_count = sum(
            1 for q in self.questions.values()
            if q.get("correct")
        )

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
            "accuracy": correct_count / total_questions if total_questions > 0 else 0,
            "retry_success_rate": retry_improvements / questions_with_retry if questions_with_retry > 0 else 0,
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
        print(f"Accuracy: {stats.get('accuracy', 0):.1%}")
        print(f"Retry success rate: {stats.get('retry_success_rate', 0):.1%}")
        print(f"Average attempts: {stats.get('avg_attempts', 0):.2f}")
        print(f"Average judge score: {stats.get('avg_judge_score', 0):.3f}")
        print("=" * 60 + "\n")
