"""Tests for API endpoints."""

import pytest
from fastapi.testclient import TestClient

from chatbot_rag.main import app

client = TestClient(app)


def test_root():
    """Test root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "app" in data
    assert "version" in data


def test_api_root():
    """Test API v1 root endpoint."""
    response = client.get("/api/v1/")
    assert response.status_code == 200
    data = response.json()
    assert data["message"] == "Welcome to Chatbot RAG API"
    assert data["status"] == "running"


def test_health_check():
    """Test health check endpoint."""
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"


def test_say_hello():
    """Test greeting endpoint."""
    name = "TestUser"
    response = client.get(f"/api/v1/hello/{name}")
    assert response.status_code == 200
    data = response.json()
    assert data["message"] == f"Hello {name}"
