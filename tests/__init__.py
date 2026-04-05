"""Parallel test infrastructure for RAG evaluation pipeline.

This package provides pytest-based tests that can run in parallel using pytest-xdist.

Usage:
    # Run all tests in parallel (4 workers)
    pytest -n 4 tests/

    # Run only fast tests (skip API calls)
    pytest -n 4 tests/ -m "not slow"

    # Run with verbose output
    pytest -n auto tests/ -v

Test Categories:
    - test_environment.py: Smoke tests for environment setup
    - test_agents.py: Unit tests for agent logic
    - test_features.py: Unit tests for feature extraction
"""
