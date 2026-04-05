"""Self-RAG: Self-Reflective RAG (Asai et al. 2023).

This is a simplified approximation that:
1. Generates an initial answer
2. Self-reflects: "Is my answer fully supported by the context?"
3. If reflection indicates issues, re-generates with more careful prompting

The original Self-RAG uses special tokens for retrieval/generation decisions,
but we approximate this with a reflection prompt.
"""

import time
from typing import Any, Dict, List, Optional

from langchain_core.documents import Document

from .base import BaselineRAG, BaselineResult


class SelfRAG(BaselineRAG):
    """Self-Reflective RAG baseline.

    Key idea: Generate, then self-critique and potentially regenerate.
    """

    def __init__(
        self,
        model_name: str = None,
        judge_model: str = None,
        temperature: float = 0.0,
        max_tokens: int = 512,
        max_iterations: int = 2,
        reflection_threshold: float = 0.6,
    ):
        """Initialize Self-RAG.

        Args:
            model_name: Generation model
            judge_model: Model for self-reflection
            temperature: Generation temperature
            max_tokens: Max tokens for response
            max_iterations: Maximum generation iterations
            reflection_threshold: Score below which to regenerate
        """
        super().__init__(model_name, judge_model, temperature, max_tokens)
        self.max_iterations = max_iterations
        self.reflection_threshold = reflection_threshold

    @property
    def name(self) -> str:
        return "Self-RAG"

    def reflect_on_answer(
        self,
        question: str,
        answer: str,
        context: str,
    ) -> tuple[float, str]:
        """Self-reflect on the answer quality.

        Returns:
            Tuple of (score, reflection_text)
        """
        reflection_prompt = f"""You are reviewing an answer for quality and grounding.

Question: {question}

Context:
{context}

Generated Answer: {answer}

Evaluate the answer on these criteria:
1. GROUNDING: Is every claim supported by the context? (0-1)
2. COMPLETENESS: Does it fully answer the question? (0-1)
3. ACCURACY: Are numbers/facts correct per the context? (0-1)

Output format:
GROUNDING: <score>
COMPLETENESS: <score>
ACCURACY: <score>
OVERALL: <average_score>
ISSUES: <brief description of any problems>"""

        response = self.judge_provider.generate(
            system_prompt="You are a critical answer reviewer.",
            user_prompt=reflection_prompt,
            max_tokens=300,
            temperature=0.0,
        )

        # Parse reflection
        reflection_text = response.content
        try:
            lines = reflection_text.strip().split('\n')
            overall_score = 0.5  # Default
            for line in lines:
                if line.startswith("OVERALL:"):
                    score_str = line.replace("OVERALL:", "").strip()
                    overall_score = float(score_str)
                    break
        except (ValueError, IndexError):
            overall_score = 0.5

        return overall_score, reflection_text

    def regenerate_with_feedback(
        self,
        question: str,
        context: str,
        previous_answer: str,
        reflection: str,
    ) -> str:
        """Regenerate answer incorporating reflection feedback."""
        system_prompt = """You are a precise financial analysis assistant.
A previous attempt had issues. Generate a better, more careful answer.
Focus on accuracy and proper grounding in the context."""

        user_prompt = f"""The previous answer had some issues identified:

Previous Answer: {previous_answer}

Issues Found: {reflection}

Please provide a corrected answer using the context below.
Be more careful about grounding claims in the provided documents.

Context:
{context}

Question: {question}

Corrected Answer:"""

        response = self.gen_provider.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        return response.content

    def process(
        self,
        question: str,
        docs: List[Document],
        gold_answer: str = None,
    ) -> BaselineResult:
        """Process question with self-reflection loop."""
        start_time = time.time()

        context = self.format_context(docs)
        metadata = {
            "iterations": 0,
            "reflections": [],
            "regenerated": False,
        }

        # Initial generation
        answer = self.generate_answer(question, context)
        metadata["iterations"] = 1

        best_answer = answer
        best_score = 0.0

        # Self-reflection loop
        for i in range(self.max_iterations):
            score, reflection = self.reflect_on_answer(question, answer, context)

            metadata["reflections"].append({
                "iteration": i + 1,
                "score": score,
                "reflection": reflection[:200] + "..." if len(reflection) > 200 else reflection,
            })

            if score > best_score:
                best_score = score
                best_answer = answer

            # Check if we should regenerate
            if score >= self.reflection_threshold:
                break  # Answer is good enough

            if i < self.max_iterations - 1:
                # Regenerate with feedback
                answer = self.regenerate_with_feedback(
                    question, context, answer, reflection
                )
                metadata["iterations"] += 1
                metadata["regenerated"] = True

        latency_ms = (time.time() - start_time) * 1000

        return BaselineResult(
            question=question,
            answer=best_answer,
            score=best_score,
            latency_ms=latency_ms,
            metadata=metadata,
        )
