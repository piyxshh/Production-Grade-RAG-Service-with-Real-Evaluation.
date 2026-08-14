"""Helper functions for formatting and exporting evaluation benchmark results."""
import csv
import json
from pathlib import Path
from typing import Any


def format_retrieval_ablation_table(ablation_results: dict[str, dict]) -> str:
    """Format retrieval ablation results into a clean ASCII table."""
    headers = ["Configuration", "Hit@1", "Hit@3", "Hit@5", "Recall@5", "MRR", "NDCG@5"]
    col_widths = [34, 8, 8, 8, 10, 8, 8]

    def fmt_row(cells: list[str]) -> str:
        return " | ".join(f"{str(c):<{col_widths[i]}}" for i, c in enumerate(cells))

    sep = "-+-".join("-" * w for w in col_widths)

    rows = [fmt_row(headers), sep]
    for config_name, data in ablation_results.items():
        m = data["metrics"]
        row = [
            config_name,
            f"{m['HitRate@1'] * 100:.1f}%",
            f"{m['HitRate@3'] * 100:.1f}%",
            f"{m['HitRate@5'] * 100:.1f}%",
            f"{m['Recall@5'] * 100:.1f}%",
            f"{m['MRR']:.3f}",
            f"{m['NDCG@5']:.3f}",
        ]
        rows.append(fmt_row(row))

    return "\n".join(rows)


def format_latency_table(stage_latencies: dict[str, dict]) -> str:
    """Format per-stage latency percentiles into an ASCII table."""
    headers = ["Pipeline Stage", "Mean (ms)", "P50 (ms)", "P90 (ms)", "P95 (ms)", "Max (ms)"]
    col_widths = [24, 11, 11, 11, 11, 11]

    def fmt_row(cells: list[str]) -> str:
        return " | ".join(f"{str(c):<{col_widths[i]}}" for i, c in enumerate(cells))

    sep = "-+-".join("-" * w for w in col_widths)

    rows = [fmt_row(headers), sep]
    for stage_name, stats in stage_latencies.items():
        row = [
            stage_name,
            f"{stats['mean_ms']:.2f}",
            f"{stats['p50_ms']:.2f}",
            f"{stats['p90_ms']:.2f}",
            f"{stats['p95_ms']:.2f}",
            f"{stats['max_ms']:.2f}",
        ]
        rows.append(fmt_row(row))

    return "\n".join(rows)


def export_retrieval_csv(ablation_results: dict[str, dict], output_path: Path) -> None:
    """Export retrieval ablation numbers to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["Configuration", "HitRate@1", "HitRate@3", "HitRate@5", "Recall@1", "Recall@3", "Recall@5", "MRR", "NDCG@3", "NDCG@5"])
        for name, data in ablation_results.items():
            m = data["metrics"]
            writer.writerow([
                name,
                f"{m['HitRate@1']:.4f}",
                f"{m['HitRate@3']:.4f}",
                f"{m['HitRate@5']:.4f}",
                f"{m['Recall@1']:.4f}",
                f"{m['Recall@3']:.4f}",
                f"{m['Recall@5']:.4f}",
                f"{m['MRR']:.4f}",
                f"{m['NDCG@3']:.4f}",
                f"{m['NDCG@5']:.4f}",
            ])


def export_full_json_report(report_data: dict[str, Any], output_path: Path) -> None:
    """Export full evaluation report to JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(report_data, fh, indent=2)

