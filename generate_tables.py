#!/usr/bin/env python3
"""
Generate Markdown tables from benchmark results.

Creates one table per number of groups, with:
- Rows: models
- Columns: number of input rows
- Cells: percent correct
"""

import json
import sys
from collections import defaultdict
from pathlib import Path


MODEL_DISPLAY_NAMES = {
    # Local Ollama models
    "gpt-oss-128k": "GPT-OSS-20B",
    "gpt-oss-120b-128k": "GPT-OSS-120B",
    "qwen2.5-coder-256k": "Qwen2.5-Coder-32B",
    # OpenAI models
    "gpt-4o": "GPT-4o",
    "gpt-4o-mini": "GPT-4o Mini",
    "gpt-4.1": "GPT-4.1",
    "gpt-4.1-mini": "GPT-4.1 Mini",
    "gpt-4.1-nano": "GPT-4.1 Nano",
    # Anthropic models
    "claude-3-5-haiku-20241022": "Claude 3.5 Haiku",
    "claude-3-7-sonnet-20250219": "Claude 3.7 Sonnet",
    "claude-sonnet-4-20250514": "Claude Sonnet 4",
    "claude-haiku-4-20250514": "Claude Haiku 4",
    "claude-sonnet-4-5-20250929": "Claude Sonnet 4.5",
    "claude-haiku-4-5-20251001": "Claude Haiku 4.5",
    "claude-opus-4-5-20251101": "Claude Opus 4.5",
}


def format_model_name(model_id: str) -> str:
    """Convert model ID to human-readable display name."""
    return MODEL_DISPLAY_NAMES.get(model_id, model_id)


def load_results(filepaths: list[str]) -> list[dict]:
    """Load and combine results from multiple JSON files."""
    all_results = []
    for filepath in filepaths:
        with open(filepath) as f:
            all_results.extend(json.load(f))
    return all_results


def compute_accuracy(results: list[dict]) -> dict:
    """
    Compute accuracy for each (model, num_groups, num_rows) combination.

    Returns dict: {(model, num_groups, num_rows): (correct_count, total_count)}
    """
    counts = defaultdict(lambda: {"correct": 0, "total": 0})

    for r in results:
        key = (r["model"], r["num_groups"], r["num_rows"])
        counts[key]["total"] += 1
        if r["correct"]:
            counts[key]["correct"] += 1

    return {k: (v["correct"], v["total"]) for k, v in counts.items()}


def generate_markdown_tables(results: list[dict]) -> str:
    """Generate Markdown tables from results."""
    accuracy = compute_accuracy(results)

    # Get unique values
    all_groups = sorted(set(r["num_groups"] for r in results))
    all_rows = sorted(set(r["num_rows"] for r in results))
    all_models = sorted(set(r["model"] for r in results))

    output = []

    for num_groups in all_groups:
        output.append(f"**{num_groups} Group{'s' if num_groups > 1 else ''}**\n")

        # Header row
        header = "| Model | " + " | ".join(f"{r} rows" for r in all_rows) + " |"
        separator = "|" + "|".join(["---"] * (len(all_rows) + 1)) + "|"
        output.append(header)
        output.append(separator)

        # Data rows
        for model in all_models:
            row_cells = [format_model_name(model)]
            for num_rows in all_rows:
                key = (model, num_groups, num_rows)
                if key in accuracy:
                    correct, total = accuracy[key]
                    pct = (correct / total) * 100 if total > 0 else 0
                    row_cells.append(f"{pct:.0f}%")
                else:
                    row_cells.append("-")
            output.append("| " + " | ".join(row_cells) + " |")

        output.append("")  # Blank line between tables

    return "\n".join(output)


def main():
    result_files = [
        "results/raw_results_20251128_095509.json",  # Local models
        "results/raw_results_20251128_125351.json",  # OpenAI
        "results/raw_results_20251128_173141.json",  # Anthropic
    ]

    # Check files exist
    for f in result_files:
        if not Path(f).exists():
            print(f"Error: {f} not found", file=sys.stderr)
            sys.exit(1)

    results = load_results(result_files)
    markdown = generate_markdown_tables(results)
    print(markdown)


if __name__ == "__main__":
    main()
