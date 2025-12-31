"""Pytest configuration and shared fixtures."""

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from chatbot_rag.main import app


@pytest.fixture(scope="module")
def client():
    """Create a test client for the FastAPI app."""
    return TestClient(app)


@pytest.fixture(scope="session")
def attack_test_cases():
    """Load attack test cases from JSON file."""
    test_cases_file = Path(__file__).parent / "fixtures" / "attack_test_cases.json"
    if test_cases_file.exists():
        with open(test_cases_file, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"test_cases": [], "test_metadata": {}}


@pytest.fixture(scope="session")
def expected_responses():
    """Define expected response patterns for different validation results."""
    return {
        "injection_blocked": {
            "pattern": "檢測到不當的輸入內容",
            "documents_used": False,
            "steps": 0,
        },
        "irrelevant_rejected": {
            "pattern": "與我們的服務範圍不相關",
            "documents_used": False,
            "steps": 0,
        },
        "length_rejected": {
            "pattern": "問題過長",
            "documents_used": False,
            "steps": 0,
        },
        "legitimate_answered": {
            "documents_used": True,
            "min_steps": 1,
        },
    }
