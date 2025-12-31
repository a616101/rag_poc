"""
Tests for handling cases when no documents are found in RAG retrieval.

This module tests that the system provides honest, helpful responses
when the knowledge base doesn't contain information about a user's question.
"""

import pytest
from fastapi.testclient import TestClient

from chatbot_rag.main import app

client = TestClient(app)


@pytest.mark.security
class TestNoDocumentsFound:
    """Test handling of queries that return no relevant documents."""

    def test_payment_methods_not_in_kb(self):
        """
        Test that system honestly admits when payment method info is not available.

        This tests the anti-hallucination feature: when no documents are found,
        the system should NOT make up information from its training data.
        """
        response = client.post(
            "/api/v1/rag/ask",
            json={"question": "支持哪些支付方式？"},
        )

        assert response.status_code == 200
        result = response.json()

        # Should use RAG (documents_used=True means it tried to retrieve)
        assert result["documents_used"], "Should attempt document retrieval"

        # Answer should contain honest admission
        answer = result["answer"]
        honest_indicators = [
            "沒有找到",
            "找不到",
            "沒有",
            "無法找到",
            "抱歉",
        ]

        has_honest_admission = any(indicator in answer for indicator in honest_indicators)
        assert has_honest_admission, f"Should honestly admit no info found, got: {answer[:200]}"

        # Should provide helpful guidance
        helpful_indicators = [
            "客服",
            "聯繫",
            "聯絡",
            "協助",
            "幫助",
        ]

        has_helpful_guidance = any(indicator in answer for indicator in helpful_indicators)
        assert has_helpful_guidance, f"Should provide helpful guidance, got: {answer[:200]}"

        # Should NOT contain specific payment methods (would indicate hallucination)
        hallucination_indicators = [
            "信用卡",
            "VISA",
            "MasterCard",
            "支付寶",
            "微信支付",
            "Apple Pay",
        ]

        has_hallucination = any(indicator in answer for indicator in hallucination_indicators)
        assert not has_hallucination, f"Should NOT hallucinate payment methods, got: {answer[:200]}"

    def test_unknown_topic_not_in_kb(self):
        """Test handling of topics relevant to business but not in knowledge base."""
        response = client.post(
            "/api/v1/rag/ask",
            json={"question": "課程證書的有效期限是多久？"},
        )

        assert response.status_code == 200
        result = response.json()

        answer = result["answer"]

        # Should contain honest response
        honest_indicators = ["沒有", "找不到", "無法", "抱歉"]
        has_honest_admission = any(indicator in answer for indicator in honest_indicators)

        assert has_honest_admission, f"Should honestly admit topic not covered, got: {answer[:200]}"

    def test_legitimate_question_still_works(self):
        """
        Test that questions with available documents still work correctly.

        This ensures our anti-hallucination feature doesn't break normal operation.
        """
        response = client.post(
            "/api/v1/rag/ask",
            json={"question": "如何申請退款？"},
        )

        assert response.status_code == 200
        result = response.json()

        # Should use documents and provide answer
        assert result["documents_used"], "Should use documents"
        assert result["steps"] > 0, "Should perform RAG steps"

        answer = result["answer"]

        # Should contain refund-related content
        refund_indicators = ["退款", "退費", "申請"]
        has_refund_content = any(indicator in answer for indicator in refund_indicators)

        assert has_refund_content, f"Should answer about refund, got: {answer[:200]}"

        # Should have substantial answer (not just "no info found")
        assert len(answer) > 200, f"Should provide detailed answer, got {len(answer)} chars"


@pytest.mark.security
class TestWarmResponses:
    """Test that responses without documents are warm and helpful."""

    def test_response_provides_alternatives(self):
        """Test that 'no documents' response suggests alternative actions."""
        response = client.post(
            "/api/v1/rag/ask",
            json={"question": "有提供企業培訓方案嗎？"},
        )

        assert response.status_code == 200
        result = response.json()
        answer = result["answer"]

        # Should provide actionable alternatives
        alternative_indicators = [
            "客服",
            "聯繫",
            "FAQ",
            "常見問題",
            "其他",
        ]

        has_alternatives = any(indicator in answer for indicator in alternative_indicators)
        assert has_alternatives, f"Should suggest alternatives, got: {answer[:200]}"

    def test_response_maintains_friendly_tone(self):
        """Test that 'no documents' response is friendly and empathetic."""
        response = client.post(
            "/api/v1/rag/ask",
            json={"question": "支持比特幣付款嗎？"},
        )

        assert response.status_code == 200
        result = response.json()
        answer = result["answer"]

        # Should use friendly language
        friendly_indicators = [
            "抱歉",
            "協助",
            "幫助",
            "樂意",
            "建議",
        ]

        is_friendly = any(indicator in answer for indicator in friendly_indicators)
        assert is_friendly, f"Should use friendly tone, got: {answer[:200]}"
