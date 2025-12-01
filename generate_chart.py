#!/usr/bin/env python3
"""
Generate charts from benchmark results.

Creates:
1. Accuracy vs number of rows for each model
2. Mean execution time vs number of rows for each model
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Output paths
SCRIPT_DIR = Path(__file__).parent
ACCURACY_CHART_PATH = SCRIPT_DIR / "accuracy_chart.png"
LATENCY_CHART_PATH = SCRIPT_DIR / "latency_chart.png"

MODEL_DISPLAY_NAMES = {
    "gpt-oss-128k": "GPT-OSS-20B",
    "gpt-oss-120b-128k": "GPT-OSS-120B",
    "qwen2.5-coder-256k": "Qwen2.5-Coder-32B",
    "gpt-4o": "GPT-4o",
    "gpt-4o-mini": "GPT-4o Mini",
    "gpt-4.1": "GPT-4.1",
    "gpt-4.1-mini": "GPT-4.1 Mini",
    "gpt-4.1-nano": "GPT-4.1 Nano",
    "gpt-5": "GPT-5",
    "claude-3-5-haiku-20241022": "Claude 3.5 Haiku",
    "claude-3-7-sonnet-20250219": "Claude 3.7 Sonnet",
    "claude-sonnet-4-20250514": "Claude Sonnet 4",
    "claude-haiku-4-20250514": "Claude Haiku 4",
    "claude-sonnet-4-5-20250929": "Claude Sonnet 4.5",
    "claude-haiku-4-5-20251001": "Claude Haiku 4.5",
    "claude-opus-4-5-20251101": "Claude Opus 4.5",
}


def format_model_name(model_id: str) -> str:
    return MODEL_DISPLAY_NAMES.get(model_id, model_id)


def load_results(filepath: str) -> list[dict]:
    with open(filepath) as f:
        return json.load(f)


def compute_accuracy_by_rows(results: list[dict]) -> dict:
    """
    Compute accuracy for each (model, num_rows) combination.

    Returns: {model: {num_rows: (accuracy, num_trials)}}
    """
    counts = defaultdict(lambda: defaultdict(lambda: {"correct": 0, "total": 0}))

    for r in results:
        model = r["model"]
        num_rows = r["num_rows"]
        counts[model][num_rows]["total"] += 1
        if r["correct"]:
            counts[model][num_rows]["correct"] += 1

    result = {}
    for model, rows_data in counts.items():
        result[model] = {}
        for num_rows, data in rows_data.items():
            accuracy = data["correct"] / data["total"] if data["total"] > 0 else 0
            result[model][num_rows] = (accuracy, data["total"])

    return result


def compute_latency_by_rows(results: list[dict]) -> dict:
    """
    Compute mean latency for each (model, num_rows) combination.

    Returns: {model: {num_rows: (mean_latency_ms, num_trials)}}
    """
    latencies = defaultdict(lambda: defaultdict(list))

    for r in results:
        model = r["model"]
        num_rows = r["num_rows"]
        # Only include successful trials with valid latency
        if r.get("latency_ms", 0) > 0 and not r.get("error"):
            latencies[model][num_rows].append(r["latency_ms"])

    result = {}
    for model, rows_data in latencies.items():
        result[model] = {}
        for num_rows, times in rows_data.items():
            if times:
                mean_latency = sum(times) / len(times)
                result[model][num_rows] = (mean_latency, len(times))

    return result


def generate_accuracy_chart(results: list[dict], output_path: Path, title: str = None):
    """Generate accuracy vs rows chart."""
    accuracy_data = compute_accuracy_by_rows(results)

    # Set up the plot
    plt.figure(figsize=(10, 6))
    plt.style.use('seaborn-v0_8-whitegrid')

    colors = plt.cm.tab10.colors
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', 'h']

    for i, (model, rows_data) in enumerate(sorted(accuracy_data.items())):
        rows = sorted(rows_data.keys())
        accuracies = [rows_data[r][0] * 100 for r in rows]  # Convert to percentage

        plt.plot(rows, accuracies,
                 marker=markers[i % len(markers)],
                 color=colors[i % len(colors)],
                 linewidth=2,
                 markersize=8,
                 label=format_model_name(model))

    plt.xlabel('Number of Rows', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.ylim(0, 105)
    plt.xlim(0, max(rows) * 1.05)

    if title:
        plt.title(title, fontsize=14)
    else:
        # Infer title from data
        num_groups = results[0].get("num_groups", 1) if results else 1
        plt.title(f'Accuracy vs Input Size ({num_groups} Group{"s" if num_groups > 1 else ""})',
                  fontsize=14)

    plt.legend(loc='best', fontsize=10)
    plt.tight_layout()

    # Save the chart
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Accuracy chart saved to {output_path}")


def generate_latency_chart(results: list[dict], output_path: Path, title: str = None):
    """Generate mean latency vs rows chart."""
    latency_data = compute_latency_by_rows(results)

    if not latency_data:
        print("No latency data available (all trials may have errors)")
        return

    # Set up the plot
    plt.figure(figsize=(10, 6))
    plt.style.use('seaborn-v0_8-whitegrid')

    colors = plt.cm.tab10.colors
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', 'h']

    all_rows = set()
    for i, (model, rows_data) in enumerate(sorted(latency_data.items())):
        rows = sorted(rows_data.keys())
        all_rows.update(rows)
        latencies = [rows_data[r][0] / 1000 for r in rows]  # Convert to seconds

        plt.plot(rows, latencies,
                 marker=markers[i % len(markers)],
                 color=colors[i % len(colors)],
                 linewidth=2,
                 markersize=8,
                 label=format_model_name(model))

    plt.xlabel('Number of Rows', fontsize=12)
    plt.ylabel('Mean Latency (seconds)', fontsize=12)

    # Start axes at 0
    if all_rows:
        max_rows = max(all_rows)
        plt.xlim(0, max_rows * 1.05)
    plt.ylim(bottom=0)

    if title:
        plt.title(title, fontsize=14)
    else:
        num_groups = results[0].get("num_groups", 1) if results else 1
        plt.title(f'Mean Latency vs Input Size ({num_groups} Group{"s" if num_groups > 1 else ""})',
                  fontsize=14)

    plt.legend(loc='best', fontsize=10)
    plt.tight_layout()

    # Save the chart
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Latency chart saved to {output_path}")


def print_summary(results: list[dict]):
    """Print summary statistics to console."""
    accuracy_data = compute_accuracy_by_rows(results)
    latency_data = compute_latency_by_rows(results)

    models = sorted(accuracy_data.keys())
    rows = sorted(set(r for m in accuracy_data.values() for r in m.keys()))

    print("\nAccuracy by row count:")
    print("-" * 60)
    header = "Rows  | " + " | ".join(f"{format_model_name(m)[:15]:>15}" for m in models)
    print(header)
    print("-" * len(header))

    for num_rows in rows:
        row_str = f"{num_rows:5} | "
        row_str += " | ".join(
            f"{accuracy_data[m].get(num_rows, (0, 0))[0] * 100:14.0f}%"
            for m in models
        )
        print(row_str)

    if latency_data:
        print("\nMean latency by row count (seconds):")
        print("-" * 60)
        print(header)
        print("-" * len(header))

        for num_rows in rows:
            row_str = f"{num_rows:5} | "
            row_str += " | ".join(
                f"{latency_data[m].get(num_rows, (0, 0))[0] / 1000:14.1f}s"
                if m in latency_data and num_rows in latency_data[m]
                else f"{'N/A':>15}"
                for m in models
            )
            print(row_str)


def main():
    parser = argparse.ArgumentParser(description="Generate charts from benchmark results")
    parser.add_argument("input_file", help="Path to raw results JSON file")
    parser.add_argument("-t", "--title", help="Base title for charts")
    args = parser.parse_args()

    if not Path(args.input_file).exists():
        print(f"Error: {args.input_file} not found")
        return 1

    results = load_results(args.input_file)

    # Generate both charts
    generate_accuracy_chart(results, ACCURACY_CHART_PATH, args.title)
    generate_latency_chart(results, LATENCY_CHART_PATH, args.title)

    # Print summary to console
    print_summary(results)

    return 0


if __name__ == "__main__":
    exit(main())
