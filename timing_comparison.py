#!/usr/bin/env python3
"""
Quick timing comparison between GPT-5 and GPT-4.1 for the 100 row, 1 group case.
"""

import time
from chatlas import ChatOpenAI

# Import data generation from benchmark
from benchmark import generate_data, USER_PROMPT_TEMPLATE, SYSTEM_PROMPT

MODELS = ["gpt-5", "gpt-4.1"]
NUM_TRIALS = 5
NUM_GROUPS = 1
NUM_ROWS = 100


def run_timing_test(model: str, num_trials: int) -> list[float]:
    """Run timing tests and return list of latencies in seconds."""
    client = ChatOpenAI(model=model, system_prompt=SYSTEM_PROMPT)
    latencies = []

    for i in range(num_trials):
        csv_data, expected = generate_data(NUM_GROUPS, NUM_ROWS, value_style="small")
        prompt = USER_PROMPT_TEMPLATE.format(csv_data=csv_data)

        start = time.perf_counter()
        response = client.chat(prompt, echo="none")
        elapsed = time.perf_counter() - start

        latencies.append(elapsed)
        print(f"  Trial {i+1}: {elapsed:.2f}s")

    return latencies


def main():
    print(f"Timing comparison: {NUM_ROWS} rows, {NUM_GROUPS} group, {NUM_TRIALS} trials each\n")

    results = {}
    for model in MODELS:
        print(f"{model}:")
        latencies = run_timing_test(model, NUM_TRIALS)
        avg = sum(latencies) / len(latencies)
        results[model] = {"latencies": latencies, "avg": avg}
        print(f"  Average: {avg:.2f}s\n")

    # Markdown table
    print("\n## Timing Results\n")

    # Header
    trial_headers = " | ".join(f"Trial {i+1}" for i in range(NUM_TRIALS))
    print(f"| Model | {trial_headers} | Average |")
    print("|" + "|".join(["---"] * (NUM_TRIALS + 2)) + "|")

    # Data rows
    for model, data in results.items():
        trial_cells = " | ".join(f"{t:.2f}s" for t in data["latencies"])
        print(f"| {model} | {trial_cells} | **{data['avg']:.2f}s** |")

    # Comparison
    if len(results) == 2:
        models = list(results.keys())
        ratio = results[models[0]]["avg"] / results[models[1]]["avg"]
        print(f"\n{models[0]} is **{ratio:.1f}x** {'slower' if ratio > 1 else 'faster'} than {models[1]}.")


if __name__ == "__main__":
    main()
