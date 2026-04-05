"""Adaptive RAG: Query Complexity-Based Routing (Jeong et al. 2024).

This is a simplified approximation that:
1. Classifies query complexity (simple vs complex)
2. Routes simple queries to single-pass retrieval
3. Routes complex queries to iterative/multi-hop retrieval

The original Adaptive RAG uses a trained classifier.
We approximate with heuristics + optional LLM classification.
"""

import re
import time
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.documents import Document

from .base import BaselineRAG, BaselineResult


class AdaptiveRAG(BaselineRAG):
    """Adaptive RAG baseline.

    Key idea: Route queries based on complexity to different strategies.
    """

    def __init__(
        self,
        model_name: str = None,
        judge_model: str = None,
        temperature: float = 0.0,
        max_tokens: int = 512,
        use_llm_classifier: bool = False,
        db=None,  # ChromaDB for multi-hop retrieval
    ):
        """Initialize Adaptive RAG.

        Args:
            model_name: Generation model
            judge_model: Model for complexity classification
            temperature: Generation temperature
            max_tokens: Max tokens for response
            use_llm_classifier: Use LLM for complexity (vs heuristics)
            db: ChromaDB instance for additional retrieval
        """
        super().__init__(model_name, judge_model, temperature, max_tokens)
        self.use_llm_classifier = use_llm_classifier
        self.db = db

    @property
    def name(self) -> str:
        return "Adaptive-RAG"

    def classify_complexity(self, question: str) -> Tuple[str, float, str]:
        """Classify query complexity.

        Returns:
            Tuple of (complexity_level, confidence, reasoning)
            complexity_level: "simple", "moderate", or "complex"
        """
        if self.use_llm_classifier:
            return self._llm_classify(question)
        else:
            return self._heuristic_classify(question)

    def _heuristic_classify(self, question: str) -> Tuple[str, float, str]:
        """Rule-based complexity classification."""
        question_lower = question.lower()

        # Complexity indicators
        complexity_score = 0.0
        reasons = []

        # Multi-hop indicators (complex)
        multi_hop_patterns = [
            r'\band\b.*\band\b',  # Multiple "and" conjunctions
            r'compare|comparison|versus|vs\.?',
            r'how does .* relate to',
            r'what is the relationship between',
            r'both .* and',
        ]
        for pattern in multi_hop_patterns:
            if re.search(pattern, question_lower):
                complexity_score += 0.3
                reasons.append("multi-hop reasoning required")
                break

        # Calculation indicators (moderate to complex)
        calc_patterns = [
            r'calculate|compute|what is the .* ratio',
            r'percentage|percent change',
            r'growth rate|margin',
            r'how much .* increase|decrease',
        ]
        for pattern in calc_patterns:
            if re.search(pattern, question_lower):
                complexity_score += 0.2
                reasons.append("calculation required")
                break

        # Temporal complexity
        year_matches = re.findall(r'\b20\d{2}\b', question)
        if len(year_matches) > 1:
            complexity_score += 0.2
            reasons.append(f"multi-year comparison ({len(year_matches)} years)")

        # Question length (longer = more complex)
        word_count = len(question.split())
        if word_count > 30:
            complexity_score += 0.2
            reasons.append(f"long question ({word_count} words)")
        elif word_count > 20:
            complexity_score += 0.1

        # Determine level
        if complexity_score >= 0.5:
            level = "complex"
            confidence = min(0.9, 0.6 + complexity_score)
        elif complexity_score >= 0.2:
            level = "moderate"
            confidence = 0.7
        else:
            level = "simple"
            confidence = 0.8

        reasoning = "; ".join(reasons) if reasons else "straightforward lookup query"
        return level, confidence, reasoning

    def _llm_classify(self, question: str) -> Tuple[str, float, str]:
        """LLM-based complexity classification."""
        prompt = f"""Classify this question's complexity for a RAG system.

Question: {question}

Complexity levels:
- SIMPLE: Direct fact lookup, single piece of information
- MODERATE: Requires some reasoning or calculation
- COMPLEX: Multi-hop reasoning, comparisons across documents, or synthesis

Output format:
COMPLEXITY: <SIMPLE|MODERATE|COMPLEX>
CONFIDENCE: <0.0-1.0>
REASONING: <brief explanation>"""

        response = self.judge_provider.generate(
            system_prompt="You are a query complexity classifier.",
            user_prompt=prompt,
            max_tokens=100,
            temperature=0.0,
        )

        # Parse response
        text = response.content.strip()
        level = "moderate"
        confidence = 0.7
        reasoning = "LLM classification"

        for line in text.split('\n'):
            if line.startswith("COMPLEXITY:"):
                level_str = line.replace("COMPLEXITY:", "").strip().lower()
                if level_str in ["simple", "moderate", "complex"]:
                    level = level_str
            elif line.startswith("CONFIDENCE:"):
                try:
                    confidence = float(line.replace("CONFIDENCE:", "").strip())
                except ValueError:
                    pass
            elif line.startswith("REASONING:"):
                reasoning = line.replace("REASONING:", "").strip()

        return level, confidence, reasoning

    def single_pass_generation(
        self,
        question: str,
        docs: List[Document],
    ) -> str:
        """Simple single-pass answer generation."""
        context = self.format_context(docs[:5])  # Use fewer docs for simple queries
        return self.generate_answer(question, context)

    def iterative_generation(
        self,
        question: str,
        docs: List[Document],
    ) -> Tuple[str, Dict[str, Any]]:
        """Iterative multi-hop answer generation.

        For complex queries:
        1. Break down the question
        2. Answer sub-questions
        3. Synthesize final answer
        """
        metadata = {
            "sub_questions": [],
            "iterations": 1,
        }

        # Step 1: Decompose complex question
        decomposition_prompt = f"""Break this complex question into 2-3 simpler sub-questions.

Question: {question}

Output format (one per line):
1. <sub-question 1>
2. <sub-question 2>
3. <sub-question 3 if needed>"""

        response = self.gen_provider.generate(
            system_prompt="You decompose complex questions into simpler parts.",
            user_prompt=decomposition_prompt,
            max_tokens=200,
            temperature=0.0,
        )

        # Parse sub-questions
        sub_questions = []
        for line in response.content.strip().split('\n'):
            if re.match(r'^\d+\.', line.strip()):
                sq = re.sub(r'^\d+\.\s*', '', line.strip())
                if sq:
                    sub_questions.append(sq)

        metadata["sub_questions"] = sub_questions

        # Step 2: Answer each sub-question
        context = self.format_context(docs)
        sub_answers = []

        for sq in sub_questions[:3]:  # Limit to 3
            sub_prompt = f"""Answer this specific sub-question using the context.

Context:
{context}

Sub-question: {sq}

Answer (be concise):"""

            sub_response = self.gen_provider.generate(
                system_prompt="You answer specific questions concisely.",
                user_prompt=sub_prompt,
                max_tokens=150,
                temperature=0.0,
            )
            sub_answers.append({
                "question": sq,
                "answer": sub_response.content,
            })
            metadata["iterations"] += 1

        # Step 3: Synthesize final answer
        synthesis_prompt = f"""Synthesize the following sub-answers into a complete response.

Original Question: {question}

Sub-answers:
{chr(10).join(f"Q: {sa['question']}{chr(10)}A: {sa['answer']}" for sa in sub_answers)}

Synthesized Answer:"""

        final_response = self.gen_provider.generate(
            system_prompt="You synthesize information into coherent answers.",
            user_prompt=synthesis_prompt,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )

        return final_response.content, metadata

    def process(
        self,
        question: str,
        docs: List[Document],
        gold_answer: str = None,
    ) -> BaselineResult:
        """Process question with adaptive routing."""
        start_time = time.time()

        # Step 1: Classify complexity
        complexity, confidence, reasoning = self.classify_complexity(question)

        metadata = {
            "complexity": complexity,
            "classification_confidence": confidence,
            "classification_reasoning": reasoning,
            "strategy": "single_pass" if complexity == "simple" else "iterative",
        }

        # Step 2: Route to appropriate strategy
        if complexity == "simple":
            answer = self.single_pass_generation(question, docs)
            metadata["iterations"] = 1
        else:
            # Moderate or complex -> iterative
            answer, iter_metadata = self.iterative_generation(question, docs)
            metadata.update(iter_metadata)

        latency_ms = (time.time() - start_time) * 1000

        return BaselineResult(
            question=question,
            answer=answer,
            score=confidence,  # Use classification confidence as proxy
            latency_ms=latency_ms,
            metadata=metadata,
        )
