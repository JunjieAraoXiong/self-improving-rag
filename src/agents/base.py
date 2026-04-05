"""Base classes for the agentic RAG system."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class AgentDecision:
    """Captures a single agent decision for logging and analysis.

    This is the core data structure that enables interpretability:
    every decision made by an agent is recorded with its reasoning,
    confidence, and metadata for later analysis.

    Attributes:
        agent_name: Name of the agent making the decision
        decision_type: Category of decision (e.g., "pipeline_selection", "answer", "evaluation")
        decision_value: The actual decision made (dict with flexible structure)
        confidence: Agent's confidence in the decision (0.0-1.0)
        reasoning: Human-readable explanation of why this decision was made
        timestamp: ISO format timestamp when decision was made
        metadata: Additional context (question features, attempt number, etc.)
    """
    agent_name: str
    decision_type: str
    decision_value: Dict[str, Any]
    confidence: float
    reasoning: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentDecision":
        """Create from dictionary."""
        return cls(**data)


class BaseAgent(ABC):
    """Abstract base class for all agents in the agentic RAG system.

    Each agent:
    1. Has a specific role (retrieval, reasoning, judging)
    2. Makes decisions based on context
    3. Logs all decisions for later analysis
    4. Can adjust its behavior based on previous attempts (escalation)

    The agent pattern provides:
    - Modularity: Each component can be tested/improved independently
    - Interpretability: All decisions are logged with reasoning
    - Adaptability: Agents can escalate strategies on retry
    """

    def __init__(self, name: str):
        """Initialize the agent.

        Args:
            name: Human-readable name for this agent
        """
        self.name = name
        self.decisions: List[AgentDecision] = []
        self._attempt: int = 0

    @property
    def attempt(self) -> int:
        """Current attempt number (0-indexed)."""
        return self._attempt

    def set_attempt(self, attempt: int) -> None:
        """Set the current attempt number for escalation logic."""
        self._attempt = attempt

    def reset(self) -> None:
        """Reset agent state for a new question."""
        self.decisions = []
        self._attempt = 0

    @abstractmethod
    def decide(self, context: Dict[str, Any]) -> AgentDecision:
        """Make a decision based on the given context.

        This is the core method each agent must implement. It should:
        1. Analyze the context
        2. Make a decision
        3. Create an AgentDecision with reasoning
        4. Log the decision
        5. Return the decision

        Args:
            context: Dictionary containing relevant information for the decision

        Returns:
            AgentDecision capturing the decision and its reasoning
        """
        pass

    def log_decision(self, decision: AgentDecision) -> None:
        """Log a decision for later analysis.

        Args:
            decision: The decision to log
        """
        self.decisions.append(decision)

    def get_last_decision(self) -> Optional[AgentDecision]:
        """Get the most recent decision made by this agent."""
        return self.decisions[-1] if self.decisions else None

    def escalate_strategy(self) -> None:
        """Adjust strategy for retry attempts.

        Subclasses can override this to implement escalation logic,
        such as retrieving more documents or using a different pipeline.
        Default implementation does nothing.
        """
        pass
