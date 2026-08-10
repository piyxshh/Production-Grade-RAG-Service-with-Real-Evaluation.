"""Query endpoint: receives natural language questions and returns grounded RAG answers with citations."""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from src.pipeline.manual import run_manual_rag_pipeline

router = APIRouter()


class SourceItem(BaseModel):
    document_id: str | None = None
    chunk_id: str | None = None
    title: str
    filename: str
    chunk_index: int
    snippet: str


class RetrievalStats(BaseModel):
    dense_count: int
    sparse_count: int
    fused_count: int
    reranked_count: int


class QueryRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=1,
        description="The natural language question to ask the RAG service.",
        examples=["How do I write a Python decorator with arguments?"],
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=50,
        description="Number of candidate chunks to fetch per retriever (dense and sparse).",
    )
    top_n: int = Field(
        default=3,
        ge=1,
        le=20,
        description="Number of top chunks to retain after cross-encoder reranking.",
    )


class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: list[SourceItem]
    retrieval_stats: RetrievalStats


@router.post("", response_model=QueryResponse)
async def query_rag(payload: QueryRequest) -> QueryResponse:
    """Execute hybrid retrieval, cross-encoder reranking, and grounded answer generation."""
    try:
        result = await run_manual_rag_pipeline(
            query=payload.question,
            top_k_retrieval=payload.top_k,
            top_n_rerank=payload.top_n,
        )
        return QueryResponse(**result)
    except Exception as err:
        raise HTTPException(
            status_code=500,
            detail=f"Error executing RAG pipeline: {err}",
        )

