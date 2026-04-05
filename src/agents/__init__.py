"""Agentic RAG system for self-correcting document QA.

This module implements a multi-agent architecture where:
- RetrievalAgent: Decides retrieval strategy (pipeline, top_k)
- ReasoningAgent: Generates answers with citations
- TableAgent: Handles numeric computation via SQL and calculator
- JudgeAgent: Evaluates answers and triggers retries

The AgenticRAGOrchestrator coordinates all agents with retry logic.
"""

from .base import AgentDecision, BaseAgent
from .logger import AgentLogger
from .retrieval_agent import RetrievalAgent
from .reasoning_agent import ReasoningAgent
from .table_agent import TableAgent
from .judge_agent import JudgeAgent
from .orchestrator import AgenticRAGOrchestrator

__all__ = [
    "AgentDecision",
    "BaseAgent",
    "AgentLogger",
    "RetrievalAgent",
    "ReasoningAgent",
    "TableAgent",
    "JudgeAgent",
    "AgenticRAGOrchestrator",
]
