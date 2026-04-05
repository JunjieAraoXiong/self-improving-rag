"""Unit tests for the agent system (orchestrator, judge, retrieval).

These tests verify agent logic without making API calls, focusing on:
- Decision making logic
- State management
- Retry behavior
- Decision logging

Usage:
    pytest tests/test_agents.py -n 4    # Parallel execution
    pytest tests/test_agents.py -v      # Verbose output
"""

import pytest
from datetime import datetime


class TestAgentDecision:
    """Tests for the AgentDecision data class."""

    def test_decision_creation(self, mock_agent_decision):
        """Test AgentDecision can be created with all fields."""
        decision = mock_agent_decision

        assert decision.agent_name == "TestAgent"
        assert decision.decision_type == "test_decision"
        assert decision.confidence == 0.85
        assert decision.decision_value["result"] == "pass"

    def test_decision_to_dict(self, mock_agent_decision):
        """Test AgentDecision serializes to dictionary."""
        decision = mock_agent_decision
        d = decision.to_dict()

        assert isinstance(d, dict)
        assert d["agent_name"] == "TestAgent"
        assert d["confidence"] == 0.85
        assert "timestamp" in d

    def test_decision_from_dict(self, mock_agent_decision):
        """Test AgentDecision can be recreated from dictionary."""
        from src.agents.base import AgentDecision

        original = mock_agent_decision
        d = original.to_dict()
        restored = AgentDecision.from_dict(d)

        assert restored.agent_name == original.agent_name
        assert restored.confidence == original.confidence
        assert restored.decision_value == original.decision_value

    def test_decision_timestamp_format(self, mock_agent_decision):
        """Test timestamp is valid ISO format."""
        decision = mock_agent_decision

        # Should parse without error
        parsed = datetime.fromisoformat(decision.timestamp)
        assert isinstance(parsed, datetime)


class TestBaseAgent:
    """Tests for the BaseAgent abstract class."""

    def test_agent_initialization(self):
        """Test agent initializes with correct state."""
        from src.agents.base import BaseAgent, AgentDecision

        class TestAgent(BaseAgent):
            def decide(self, context):
                return AgentDecision(
                    agent_name=self.name,
                    decision_type="test",
                    decision_value={},
                    confidence=1.0,
                    reasoning="test",
                )

        agent = TestAgent("MyAgent")

        assert agent.name == "MyAgent"
        assert agent.attempt == 0
        assert len(agent.decisions) == 0

    def test_agent_reset(self):
        """Test agent reset clears state."""
        from src.agents.base import BaseAgent, AgentDecision

        class TestAgent(BaseAgent):
            def decide(self, context):
                decision = AgentDecision(
                    agent_name=self.name,
                    decision_type="test",
                    decision_value={},
                    confidence=1.0,
                    reasoning="test",
                )
                self.log_decision(decision)
                return decision

        agent = TestAgent("MyAgent")
        agent.set_attempt(2)
        agent.decide({})

        assert agent.attempt == 2
        assert len(agent.decisions) == 1

        agent.reset()

        assert agent.attempt == 0
        assert len(agent.decisions) == 0

    def test_agent_decision_logging(self):
        """Test decisions are logged correctly."""
        from src.agents.base import BaseAgent, AgentDecision

        class TestAgent(BaseAgent):
            def decide(self, context):
                decision = AgentDecision(
                    agent_name=self.name,
                    decision_type="test",
                    decision_value={"context": context},
                    confidence=0.9,
                    reasoning="Based on context",
                )
                self.log_decision(decision)
                return decision

        agent = TestAgent("LoggingAgent")

        # Make multiple decisions
        agent.decide({"query": "first"})
        agent.decide({"query": "second"})

        assert len(agent.decisions) == 2
        assert agent.decisions[0].decision_value["context"]["query"] == "first"
        assert agent.decisions[1].decision_value["context"]["query"] == "second"

    def test_get_last_decision(self):
        """Test retrieving the most recent decision."""
        from src.agents.base import BaseAgent, AgentDecision

        class TestAgent(BaseAgent):
            def decide(self, context):
                decision = AgentDecision(
                    agent_name=self.name,
                    decision_type="test",
                    decision_value=context,
                    confidence=1.0,
                    reasoning="test",
                )
                self.log_decision(decision)
                return decision

        agent = TestAgent("LastDecisionAgent")

        # No decisions yet
        assert agent.get_last_decision() is None

        agent.decide({"id": 1})
        agent.decide({"id": 2})

        last = agent.get_last_decision()
        assert last is not None
        assert last.decision_value["id"] == 2


class TestJudgeAgentLogic:
    """Tests for JudgeAgent decision logic (no API calls)."""

    def test_should_retry_below_threshold(self):
        """Test retry triggers when score is below threshold."""
        from src.agents.judge_agent import JudgeAgent

        judge = JudgeAgent(
            retry_threshold=0.5,
            min_threshold=0.3,
            enable_deterministic_gate=False,  # Disable for unit testing
        )

        assert judge.should_retry(0.3, attempt=0, max_retries=2) is True
        assert judge.should_retry(0.6, attempt=0, max_retries=2) is False

    def test_should_retry_max_attempts(self):
        """Test no retry at max attempts even with low score."""
        from src.agents.judge_agent import JudgeAgent

        judge = JudgeAgent(
            retry_threshold=0.5,
            enable_deterministic_gate=False,
        )

        # At max retries, should not retry
        assert judge.should_retry(0.1, attempt=2, max_retries=2) is False

    def test_threshold_decreases_with_attempts(self):
        """Test retry threshold decreases with each attempt."""
        from src.agents.judge_agent import JudgeAgent

        judge = JudgeAgent(
            retry_threshold=0.5,
            min_threshold=0.2,
            enable_deterministic_gate=False,
        )

        # Attempt 0: threshold = 0.5
        assert judge.should_retry(0.45, attempt=0, max_retries=3) is True

        # Attempt 1: threshold = 0.4
        assert judge.should_retry(0.45, attempt=1, max_retries=3) is False

    def test_judge_reset(self):
        """Test judge reset clears attempt scores."""
        from src.agents.judge_agent import JudgeAgent

        judge = JudgeAgent(enable_deterministic_gate=False)
        judge._attempt_scores = [0.3, 0.5, 0.6]
        judge._last_verification = "some_result"

        judge.reset()

        assert judge._attempt_scores == []
        assert judge._last_verification is None

    def test_empty_answer_evaluation(self):
        """Test empty answer gets score 0."""
        from src.agents.judge_agent import JudgeAgent

        judge = JudgeAgent(enable_deterministic_gate=False)

        score, justification = judge.evaluate(
            question="What is revenue?",
            predicted_answer="",
            gold_answer="$1 billion",
        )

        assert score == 0.0
        assert "Empty" in justification


class TestJudgeAgentDecision:
    """Tests for JudgeAgent decide() method with mocked evaluation."""

    def test_decide_passes_high_score(self):
        """Test decision passes with high evaluation score."""
        from src.agents.judge_agent import JudgeAgent

        judge = JudgeAgent(enable_deterministic_gate=False)

        # Mock the evaluate method to return a high score
        original_evaluate = judge.evaluate
        judge.evaluate = lambda q, a, g: (0.9, "Good answer")

        try:
            decision = judge.decide({
                "question": "What is X?",
                "predicted_answer": "X is Y",
                "gold_answer": "X is Y",
                "attempt": 0,
                "max_retries": 2,
            })

            assert decision.decision_value["pass"] is True
            assert decision.decision_value["retry"] is False
            assert decision.decision_value["score"] == 0.9
        finally:
            judge.evaluate = original_evaluate

    def test_decide_triggers_retry_low_score(self):
        """Test decision triggers retry with low score."""
        from src.agents.judge_agent import JudgeAgent

        judge = JudgeAgent(
            retry_threshold=0.5,
            enable_deterministic_gate=False,
        )

        # Mock low score
        judge.evaluate = lambda q, a, g: (0.2, "Poor answer")

        decision = judge.decide({
            "question": "What is X?",
            "predicted_answer": "I don't know",
            "gold_answer": "X is Y",
            "attempt": 0,
            "max_retries": 2,
        })

        assert decision.decision_value["pass"] is False
        assert decision.decision_value["retry"] is True


class TestAgentIntegration:
    """Integration tests for agent interactions."""

    def test_decision_chain_logged(self):
        """Test a chain of decisions is properly logged."""
        from src.agents.base import BaseAgent, AgentDecision

        class MockOrchestrator(BaseAgent):
            def decide(self, context):
                decision = AgentDecision(
                    agent_name=self.name,
                    decision_type="orchestration",
                    decision_value={"action": context.get("action", "default")},
                    confidence=0.8,
                    reasoning="Orchestrating action",
                    metadata={"step": context.get("step", 0)},
                )
                self.log_decision(decision)
                return decision

        orchestrator = MockOrchestrator("Orchestrator")

        # Simulate multi-step workflow
        orchestrator.decide({"action": "retrieve", "step": 1})
        orchestrator.decide({"action": "reason", "step": 2})
        orchestrator.decide({"action": "judge", "step": 3})

        assert len(orchestrator.decisions) == 3
        assert [d.decision_value["action"] for d in orchestrator.decisions] == [
            "retrieve", "reason", "judge"
        ]
