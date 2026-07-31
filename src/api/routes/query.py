# [YOU IMPLEMENT] POST /query
# This route receives a question, runs it through the RAG pipeline,
# and returns a grounded answer with source citations.
#
# Wire this up AFTER you have src/pipeline/manual.py working end-to-end.
from fastapi import APIRouter

router = APIRouter()

@router.post("")
async def query_rag(payload: dict):
    # TODO: call the pipeline and return the result
    raise NotImplementedError("Implement the query pipeline first")
