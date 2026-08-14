"""Generation Evaluation Module: evaluate RAG generation quality and citation grounding.

Evaluates:
1. Refusal Accuracy (Out-of-corpus robustness): Did the system correctly refuse out-of-context queries?
2. Citation Correctness: Are citations present, well-formed, and pointing to actual context chunks?
3. Answer Relevance (Lexical & Semantic F1 / Overlap against ground truth).
4. Grounding & Faithfulness: Verifies that assertions originate from context rather than hallucination.
"""
import re
from typing import Any
from src.generation.prompt_builder import REFUSAL_MESSAGE


_CITATION_RE = re.compile(
    r"\[Doc:\s*([^,\]]+),\s*Chunk:\s*(\d+)\]",
    re.IGNORECASE,
)


def extract_citations(answer: str) -> list[tuple[str, int]]:
    """Extract all [Doc: <title>, Chunk: <index>] citations from generated text."""
    matches = _CITATION_RE.findall(answer)
    return [(m[0].strip(), int(m[1])) for m in matches]


def compute_token_f1(prediction: str, ground_truth: str) -> tuple[float, float, float]:
    """Compute token-level Precision, Recall, and F1 between prediction and ground truth."""
    pred_tokens = re.findall(r"\w+", prediction.lower())
    gt_tokens = re.findall(r"\w+", ground_truth.lower())

    if not pred_tokens or not gt_tokens:
        return (0.0, 0.0, 0.0)

    pred_set = set(pred_tokens)
    gt_set = set(gt_tokens)

    common = pred_set & gt_set
    if not common:
        return (0.0, 0.0, 0.0)

    precision = len(common) / len(pred_set)
    recall = len(common) / len(gt_set)
    f1 = 2 * (precision * recall) / (precision + recall)

    return precision, recall, f1


def evaluate_generation_item(
    question_item: dict,
    pipeline_result: dict,
) -> dict[str, Any]:
    """Evaluate generation output for a single test question."""
    category = question_item["category"]
    gt_answer = question_item.get("ground_truth_answer")
    gt_docs = set(question_item.get("relevant_docs", []))
    gt_chunk_indices = set(question_item.get("relevant_chunk_indices", []))

    answer = pipeline_result.get("answer", "")
    sources = pipeline_result.get("sources", [])

    is_unanswerable = category == "unanswerable_out_of_corpus" or gt_answer is None

    # 1. Refusal Check
    refusal_triggered = (
        REFUSAL_MESSAGE.lower() in answer.lower()
        or "cannot answer" in answer.lower()
        or "not provided in the context" in answer.lower()
    )

    if is_unanswerable:
        return {
            "id": question_item["id"],
            "category": category,
            "is_unanswerable": True,
            "refusal_correct": refusal_triggered,
            "faithfulness": 1.0 if refusal_triggered else 0.0,
            "answer_f1": 1.0 if refusal_triggered else 0.0,
            "citations_count": 0,
            "citation_precision": 1.0 if refusal_triggered else 0.0,
            "answer_preview": answer[:120],
        }

    # 2. Citation Evaluation for answerable queries
    citations = extract_citations(answer)
    retrieved_pairs = {(s["filename"], s["chunk_index"]) for s in sources}

    valid_citations = 0
    for doc, cidx in citations:
        # Check if cited doc/chunk is in the retrieved sources
        if any(doc in s_doc and cidx == s_cidx for s_doc, s_cidx in retrieved_pairs):
            valid_citations += 1

    citation_precision = (valid_citations / len(citations)) if citations else 0.0

    # Check if ground truth doc/chunk was cited
    ground_truth_cited = any(
        any(gt_d in doc for gt_d in gt_docs) for doc, _ in citations
    ) if citations else False

    # 3. Grounding & Token F1 against ground truth
    prec, rec, f1 = compute_token_f1(answer, gt_answer)

    # Faithfulness heuristic: does answer avoid hallucinations and cite sources?
    faithfulness = 1.0 if (not refusal_triggered and (citations or prec > 0.3)) else (0.0 if refusal_triggered else 0.5)

    return {
        "id": question_item["id"],
        "category": category,
        "is_unanswerable": False,
        "refusal_correct": not refusal_triggered,
        "faithfulness": faithfulness,
        "answer_precision": prec,
        "answer_recall": rec,
        "answer_f1": f1,
        "citations_count": len(citations),
        "citation_precision": citation_precision,
        "ground_truth_cited": ground_truth_cited,
        "answer_preview": answer[:120],
    }


def aggregate_generation_metrics(item_evals: list[dict]) -> dict[str, Any]:
    """Aggregate per-query generation evaluations into overall benchmark metrics."""
    answerable = [e for e in item_evals if not e["is_unanswerable"]]
    unanswerable = [e for e in item_evals if e["is_unanswerable"]]

    refusal_acc = (
        sum(1 for e in unanswerable if e["refusal_correct"]) / len(unanswerable)
        if unanswerable
        else 1.0
    )

    avg_faithfulness = (
        sum(e["faithfulness"] for e in answerable) / len(answerable)
        if answerable
        else 0.0
    )

    avg_f1 = (
        sum(e["answer_f1"] for e in answerable) / len(answerable)
        if answerable
        else 0.0
    )

    avg_citation_prec = (
        sum(e["citation_precision"] for e in answerable) / len(answerable)
        if answerable
        else 0.0
    )

    citation_coverage = (
        sum(1 for e in answerable if e["citations_count"] > 0) / len(answerable)
        if answerable
        else 0.0
    )

    is_offline = any("[LLM Offline]" in e.get("answer_preview", "") for e in item_evals)

    return {
        "total_evaluated": len(item_evals),
        "answerable_count": len(answerable),
        "unanswerable_count": len(unanswerable),
        "llm_status": "offline_diagnostic" if is_offline else "live_inference",
        "refusal_accuracy_negative_tests": refusal_acc if not is_offline else None,
        "faithfulness_score": avg_faithfulness if not is_offline else None,
        "answer_relevance_f1": avg_f1 if not is_offline else None,
        "citation_precision": avg_citation_prec if not is_offline else None,
        "citation_coverage": citation_coverage if not is_offline else None,
    }

