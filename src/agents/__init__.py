"""Agentic RAG system for self-improving document QA.

This module implements a multi-agent architecture where:
- RetrievalAgent: Decides retrieval strategy (pipeline, top_k)
- ReasoningAgent: Generates answers with citations
- JudgeAgent: Evaluates answers and triggers retries

The AgenticRAGOrchestrator coordinates all agents with retry logic.
"""

from .base import AgentDecision, BaseAgent
from .logger import AgentLogger
from .retrieval_agent import RetrievalAgent
from .reasoning_agent import ReasoningAgent
from .judge_agent import JudgeAgent
from .orchestrator import AgenticRAGOrchestrator

__all__ = [
    "AgentDecision",
    "BaseAgent",
    "AgentLogger",
    "RetrievalAgent",
    "ReasoningAgent",
    "JudgeAgent",
    "AgenticRAGOrchestrator",
]
