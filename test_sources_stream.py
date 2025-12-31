#!/usr/bin/env python3
"""
Test script to verify sources streaming in the GraphRAG API.
Usage: python test_sources_stream.py
"""

import httpx
import json

API_URL = "http://localhost:8000/api/v1/rag/ask/stream_chat"

def test_sources_stream():
    """Test that sources event is properly sent in SSE stream."""
    payload = {
        "messages": [
            {"role": "user", "content": "糖尿病的症狀有哪些？"}
        ],
        "stream": True,
        "include_sources": True
    }

    print(f"Sending request to {API_URL}")
    print(f"Payload: {json.dumps(payload, ensure_ascii=False, indent=2)}")
    print("-" * 50)

    sources_received = False
    content_received = False

    with httpx.stream("POST", API_URL, json=payload, timeout=60.0) as response:
        for line in response.iter_lines():
            if not line or not line.startswith("data:"):
                continue

            data_str = line[5:].strip()
            if data_str == "[DONE]":
                print("\n[DONE]")
                break

            try:
                event = json.loads(data_str)

                # Check for sources event
                if event.get("type") == "sources":
                    sources_received = True
                    sources = event.get("sources", [])
                    print(f"\n📚 SOURCES RECEIVED ({len(sources)} items):")
                    for src in sources:
                        print(f"  [{src.get('index')}] {src.get('source_doc', 'unknown')}")
                        print(f"      Score: {src.get('relevance_score', 0)}")
                        print(f"      Content: {src.get('content', '')[:100]}...")

                # Check for status event
                elif event.get("type") == "status":
                    print(f"📍 Status: {event.get('node')} - {event.get('stage')}")

                # Check for content
                elif "choices" in event:
                    choices = event.get("choices", [])
                    if choices:
                        delta = choices[0].get("delta", {})
                        content = delta.get("content", "")
                        if content:
                            content_received = True
                            print(content, end="", flush=True)

            except json.JSONDecodeError as e:
                print(f"Failed to parse: {data_str[:100]}")

    print("\n" + "-" * 50)
    print("Summary:")
    print(f"  Content received: {'✅' if content_received else '❌'}")
    print(f"  Sources received: {'✅' if sources_received else '❌'}")

    if not sources_received:
        print("\n⚠️  Sources were NOT received. Check backend logs for details.")
        print("   Look for: 'Building sources: evidence_table has X items'")
        print("   And: 'Sources data: X items to send'")

if __name__ == "__main__":
    test_sources_stream()
