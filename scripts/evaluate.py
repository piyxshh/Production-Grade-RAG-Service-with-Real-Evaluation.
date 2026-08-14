"""Comprehensive Evaluation & Benchmarking CLI for the RAG Service.

Executes:
1. Retrieval ablation across 4 configurations (Dense, BM25, Hybrid RRF, Hybrid+Reranker).
2. Generation quality & factual grounding evaluation over 50 test questions.
3. Sub-system latency percentiles (Retrieval, Fusion, Rerank, Prompt, Generation, Total).
4. Terminal table rendering and export to evaluation/results/.

Usage:
    python scripts/evaluate.py
    python -m scripts.evaluate
"""
import asyncio
import json
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.generation_eval import (
    aggregate_generation_metrics,
    evaluate_generation_item,
)
from src.evaluation.latency_eval import benchmark_pipeline_latencies
from src.evaluation.metrics import (
    export_full_json_report,
    export_retrieval_csv,
    format_latency_table,
    format_retrieval_ablation_table,
)
from src.evaluation.retrieval_eval import run_all_retrieval_ablations
from src.pipeline.manual import run_manual_rag_pipeline


TEST_SET_PATH = PROJECT_ROOT / "evaluation" / "test_set.json"
RESULTS_DIR = PROJECT_ROOT / "evaluation" / "results"


async def main() -> None:
    print("=" * 80)
    print(" " * 20 + "RAG SERVICE BENCHMARK & EVALUATION SUITE")
    print("=" * 80)

    if not TEST_SET_PATH.is_file():
        print(f"Error: Test set not found at {TEST_SET_PATH}")
        sys.exit(1)

    with TEST_SET_PATH.open("r", encoding="utf-8") as fh:
        test_set = json.load(fh)

    print(f"\n[1/4] Loaded {len(test_set)} evaluation test cases from {TEST_SET_PATH.name}")
    category_counts = {}
    for item in test_set:
        cat = item["category"]
        category_counts[cat] = category_counts.get(cat, 0) + 1
    for cat, cnt in category_counts.items():
        print(f"      - {cat:<30}: {cnt} items")

    # Step 1: Retrieval Ablation Evaluation
    print("\n[2/4] Running Retrieval Ablation across 4 configurations...")
    retrieval_ablations = await run_all_retrieval_ablations(test_set, k_eval=5)

    print("\n" + "=" * 80)
    print("RETRIEVAL ABLATION COMPARISON (k=5)")
    print("=" * 80)
    print(format_retrieval_ablation_table(retrieval_ablations))
    print("=" * 80)

    # Step 2: Generation & Grounding Evaluation
    print("\n[3/4] Running End-to-End Generation & Grounding Evaluation...")
    generation_item_results = []
    for idx, item in enumerate(test_set, 1):
        res = await run_manual_rag_pipeline(item["question"], top_k_retrieval=5, top_n_rerank=3)
        gen_eval = evaluate_generation_item(item, res)
        generation_item_results.append(gen_eval)
        print(f"      Progress: {idx}/{len(test_set)} queries processed", end="\r")

    print(f"\n      Completed generation evaluation on {len(test_set)} queries.")
    generation_summary = aggregate_generation_metrics(generation_item_results)

    print("\n" + "=" * 80)
    print("GENERATION & GROUNDING QUALITY SUMMARY")
    print("=" * 80)
    print(f"Total Test Cases Evaluated       : {generation_summary['total_evaluated']}")
    print(f"Answerable Queries Evaluated     : {generation_summary['answerable_count']}")
    print(f"Unanswerable (Negative) Queries  : {generation_summary['unanswerable_count']}")
    print(f"Refusal Accuracy (Negative Tests): {generation_summary['refusal_accuracy_negative_tests'] * 100:.1f}%")
    print(f"Faithfulness / Grounding Score   : {generation_summary['faithfulness_score'] * 100:.1f}%")
    print(f"Answer Relevance (Token F1)      : {generation_summary['answer_relevance_f1'] * 100:.1f}%")
    print(f"Citation Precision               : {generation_summary['citation_precision'] * 100:.1f}%")
    print(f"Citation Coverage                : {generation_summary['citation_coverage'] * 100:.1f}%")
    print("=" * 80)

    # Step 3: Latency Benchmarking
    print("\n[4/4] Profiling Per-Stage and End-to-End Latency...")
    latency_report = await benchmark_pipeline_latencies(test_set)

    print("\n" + "=" * 80)
    print("SYSTEM LATENCY PROFILE (50 queries sampled)")
    print("=" * 80)
    print(format_latency_table(latency_report["stage_latencies"]))
    print("=" * 80)

    # Step 4: Export Results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_out = RESULTS_DIR / "retrieval_ablation.csv"
    json_out = RESULTS_DIR / "evaluation_report.json"

    export_retrieval_csv(retrieval_ablations, csv_out)
    export_full_json_report(
        {
            "metadata": {
                "test_set_size": len(test_set),
                "categories": category_counts,
            },
            "retrieval_ablation": {
                name: data["metrics"] for name, data in retrieval_ablations.items()
            },
            "generation_metrics": generation_summary,
            "latency_metrics": latency_report["stage_latencies"],
            "item_level_generation": generation_item_results,
            "item_level_retrieval": {
                name: data["query_details"] for name, data in retrieval_ablations.items()
            },
        },
        json_out,
    )

    print(f"\n[OK] Results successfully exported to:")
    print(f"     - CSV : {csv_out.relative_to(PROJECT_ROOT)}")
    print(f"     - JSON: {json_out.relative_to(PROJECT_ROOT)}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
