"""Verification script: runs the 5-query validation suite against the manual RAG pipeline."""
import asyncio
import json
from src.pipeline.manual import run_manual_rag_pipeline

QUERIES = [
    (
        "Query 1 (Python Decorators)",
        "How do I create a decorator in Python that accepts arguments?",
    ),
    (
        "Query 2 (Requests Library)",
        "How do I configure session retries and custom headers in Python requests?",
    ),
    (
        "Query 3 (React Hooks)",
        "What are the main rules of hooks in React and how does useEffect cleanup work?",
    ),
    (
        "Query 4 (Specific Concept Match)",
        "Why is functools.wraps important when writing decorators?",
    ),
    (
        "Query 5 (Out-of-Corpus Negative Test)",
        "How do I deploy a Kubernetes cluster using AWS EKS Terraform scripts?",
    ),
]


async def main() -> None:
    print("=" * 80)
    print("STARTING 5-QUERY RAG PIPELINE EMPIRICAL VERIFICATION")
    print("=" * 80)

    for label, query in QUERIES:
        print(f"\n>>> [{label}]")
        print(f"QUESTION: {query}")

        result = await run_manual_rag_pipeline(query, top_k_retrieval=5, top_n_rerank=3)

        print(f"\nRETRIEVAL STATS: {result['retrieval_stats']}")
        print("\nSOURCES RETRIEVED:")
        for idx, src in enumerate(result["sources"], 1):
            print(f"  [{idx}] {src['title']} (Chunk {src['chunk_index']})")
            print(f"      Snippet: {src['snippet'][:120]}...")

        print("\nGENERATED ANSWER:")
        print(result["answer"])
        print("-" * 80)


if __name__ == "__main__":
    asyncio.run(main())
