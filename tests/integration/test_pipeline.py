"""Integration tests for the RAG FastAPI service."""
import pytest
from httpx import ASGITransport, AsyncClient
from src.main import app


@pytest.mark.asyncio
async def test_health_endpoint() -> None:
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}


@pytest.mark.asyncio
async def test_query_endpoint_e2e() -> None:
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        payload = {
            "question": "How do I write a Python decorator?",
            "top_k": 3,
            "top_n": 2,
        }
        resp = await client.post("/query", json=payload)
        assert resp.status_code == 200

        data = resp.json()
        assert data["question"] == "How do I write a Python decorator?"
        assert "answer" in data
        assert len(data["sources"]) <= 2
        assert data["retrieval_stats"]["dense_count"] > 0
        assert data["retrieval_stats"]["sparse_count"] > 0

