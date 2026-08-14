"""Latency & System Performance Evaluation Module: measures per-stage and end-to-end timings."""
import asyncio
import statistics
import time
from typing import Any

from src.config import settings
from src.generation import generator, prompt_builder
from src.retrieval import dense, fusion, reranker, sparse


def calculate_percentiles(values: list[float]) -> dict[str, float]:
    """Compute summary statistics (Mean, P50, P90, P95, Min, Max) in milliseconds."""
    if not values:
        return {"mean_ms": 0.0, "p50_ms": 0.0, "p90_ms": 0.0, "p95_ms": 0.0, "min_ms": 0.0, "max_ms": 0.0}

    sorted_vals = sorted(values)
    n = len(sorted_vals)

    def percentile(p: float) -> float:
        k = (n - 1) * p
        f = math_floor = int(k)
        c = math_ceil = min(f + 1, n - 1)
        if f == c:
            return sorted_vals[int(k)]
        d0 = sorted_vals[f] * (c - k)
        d1 = sorted_vals[c] * (k - f)
        return d0 + d1

    return {
        "mean_ms": round(statistics.mean(values) * 1000, 2),
        "p50_ms": round(percentile(0.50) * 1000, 2),
        "p90_ms": round(percentile(0.90) * 1000, 2),
        "p95_ms": round(percentile(0.95) * 1000, 2),
        "min_ms": round(min(values) * 1000, 2),
        "max_ms": round(max(values) * 1000, 2),
    }


async def profile_query_execution(
    query: str,
    top_k: int = 5,
    top_n: int = 3,
) -> dict[str, Any]:
    """Profile a single query execution across all pipeline stages."""
    t0 = time.perf_counter()

    # Stage 1: Retrieval (Dense + Sparse parallel)
    t_retrieval_start = time.perf_counter()
    dense_task = dense.search(query, top_k=top_k)
    sparse_task = sparse.search(query, top_k=top_k)
    dense_res, sparse_res = await asyncio.gather(dense_task, sparse_task)
    t_retrieval = time.perf_counter() - t_retrieval_start

    # Stage 2: RRF Fusion
    t_fusion_start = time.perf_counter()
    fused_res = fusion.fuse_results(dense_res, sparse_res, k=settings.rrf_k, top_n=top_k * 2)
    t_fusion = time.perf_counter() - t_fusion_start

    # Stage 3: Cross-Encoder Reranker
    t_rerank_start = time.perf_counter()
    reranked_res = reranker.rerank_chunks(query, fused_res, top_n=top_n)
    t_rerank = time.perf_counter() - t_rerank_start

    selected_chunks = [item[0] for item in reranked_res]

    # Stage 4: Prompt Construction
    t_prompt_start = time.perf_counter()
    sys_prompt, user_prompt = prompt_builder.build_rag_prompt(query, selected_chunks)
    t_prompt = time.perf_counter() - t_prompt_start

    # Stage 5: Generation
    t_gen_start = time.perf_counter()
    answer = await generator.generate_answer(sys_prompt, user_prompt)
    t_gen = time.perf_counter() - t_gen_start

    t_total = time.perf_counter() - t0

    return {
        "query": query,
        "answer": answer,
        "selected_chunks": selected_chunks,
        "timings": {
            "retrieval_sec": t_retrieval,
            "fusion_sec": t_fusion,
            "rerank_sec": t_rerank,
            "prompt_sec": t_prompt,
            "generation_sec": t_gen,
            "total_sec": t_total,
        },
    }


async def benchmark_pipeline_latencies(test_set: list[dict], sample_limit: int = 50) -> dict[str, Any]:
    """Benchmark end-to-end and per-stage latencies over the evaluation dataset."""
    sample = test_set[:sample_limit]

    retrieval_times = []
    fusion_times = []
    rerank_times = []
    prompt_times = []
    gen_times = []
    total_times = []
    results = []

    for item in sample:
        profile = await profile_query_execution(item["question"])
        timings = profile["timings"]

        retrieval_times.append(timings["retrieval_sec"])
        fusion_times.append(timings["fusion_sec"])
        rerank_times.append(timings["rerank_sec"])
        prompt_times.append(timings["prompt_sec"])
        gen_times.append(timings["generation_sec"])
        total_times.append(timings["total_sec"])

        results.append(
            {
                "id": item["id"],
                "category": item["category"],
                "question": item["question"],
                "answer": profile["answer"],
                "selected_chunks": [c.chunk_index for c in profile["selected_chunks"]],
                "total_ms": round(timings["total_sec"] * 1000, 2),
            }
        )

    return {
        "sample_count": len(sample),
        "stage_latencies": {
            "retrieval": calculate_percentiles(retrieval_times),
            "fusion_rrf": calculate_percentiles(fusion_times),
            "reranker": calculate_percentiles(rerank_times),
            "prompt_assembly": calculate_percentiles(prompt_times),
            "llm_generation": calculate_percentiles(gen_times),
            "end_to_end_total": calculate_percentiles(total_times),
        },
        "query_profiles": results,
    }
