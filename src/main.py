"""
RAG Service — FastAPI Application Entry Point
"""
from fastapi import FastAPI
from src.api.routes import health, query, ingest

app = FastAPI(
    title="RAG Service",
    description="Production-grade RAG with hybrid retrieval and RAGAS evaluation",
    version="0.1.0",
)

app.include_router(health.router, tags=["health"])
app.include_router(query.router, prefix="/query", tags=["query"])
app.include_router(ingest.router, prefix="/ingest", tags=["ingest"])
