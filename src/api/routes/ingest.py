# [YOU IMPLEMENT] POST /ingest
# Triggers the ingestion pipeline for a given source.
from fastapi import APIRouter

router = APIRouter()

@router.post("")
async def ingest(payload: dict):
    # TODO: call ingestion pipeline
    raise NotImplementedError("Implement the ingestion pipeline first")
