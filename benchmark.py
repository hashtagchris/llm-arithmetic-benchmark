#!/usr/bin/env python3
"""
LLM Group Counting Benchmark

Tests LLM ability to perform group-by sum aggregations on CSV data.
No tool calling allowed - purely testing arithmetic/counting ability.
"""

import argparse
import csv
import io
import json
import os
import random
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Model configurations
# Available on dgx-spark - using extended context variants
OLLAMA_MODELS = [
    "gpt-oss-128k",       # 20B with 128K context
    "gpt-oss-120b-128k",  # 120B with 128K context
    "qwen2.5-coder-256k", # 32B coder with 256K context
]

# Short context variants (original defaults)
OLLAMA_MODELS_SHORT = [
    "gpt-oss:20b",
    "gpt-oss:120b",
    "qwen2.5-coder:32b",
]

OPENAI_MODELS = [
    # "gpt-5",        # Latest frontier
    "gpt-4o",
    "gpt-4o-mini",
    "gpt-4.1",
    "gpt-4.1-mini",
    "gpt-4.1-nano",
]

ANTHROPIC_MODELS = [
    # Claude 3.5/3.7 models (no 3.7 Haiku - only Sonnet got the upgrade)
    "claude-3-7-sonnet-20250219",
    "claude-3-5-haiku-20241022",
    # Claude 4 models (no Haiku 4 - Haiku skipped from 3.5 to 4.5)
    "claude-sonnet-4-20250514",
    # Claude 4.5 models
    "claude-sonnet-4-5-20250929",
    "claude-haiku-4-5-20251001",
    "claude-opus-4-5-20251101",
]

# Benchmark grid
NUM_GROUPS = [1, 3, 10, 25]
NUM_ROWS = [10, 50, 100]

# Prompt template
SYSTEM_PROMPT = """You are a data analyst. When given CSV data, analyze it and provide the requested aggregation.
Return ONLY the result in CSV format with no additional text or explanation.
Do not use any tools or code execution - compute the answer directly."""

USER_PROMPT_TEMPLATE = """Here is a CSV data table:

{csv_data}

Please compute the sum of 'value' grouped by 'groupid' and return the result as CSV.
Return ONLY the CSV output in this exact format, with no other text:

groupid,sum_value
<group1>,<sum1>
<group2>,<sum2>
...
"""


@dataclass
class TrialResult:
    """Result of a single benchmark trial."""
    model: str
    num_groups: int
    num_rows: int
    trial_num: int
    correct: bool
    expected: dict[str, int]
    actual: dict[str, int] | None
    raw_response: str
    error: str | None = None
    latency_ms: float = 0.0


# Value style options
VALUE_STYLES = ["small", "large", "dollars"]


@dataclass
class BenchmarkConfig:
    """Configuration for a benchmark run."""
    models: list[str]
    num_groups: list[int] = field(default_factory=lambda: NUM_GROUPS)
    num_rows: list[int] = field(default_factory=lambda: NUM_ROWS)
    trials_per_case: int = 10
    ollama_base_url: str = "http://dgx-spark:11434"
    output_dir: str = "results"
    seed: int | None = None
    value_style: str = "small"  # "small" (0-10), "large" (0-100), "dollars" ($1.00-$10.00)


def generate_data(num_groups: int, num_rows: int, seed: int | None = None, value_style: str = "small") -> tuple[str, dict]:
    """
    Generate random CSV data and compute expected group sums.

    Each group appears approximately the same number of times (balanced distribution).
    The order of rows is randomized.

    Args:
        value_style: "small" (0-10), "large" (0-100), "dollars" ($1.00-$10.00)

    Returns:
        Tuple of (csv_string, expected_sums_dict)
        For dollars, sums are stored as cents (integers) for precise comparison.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    # Generate group IDs
    group_ids = [f"g{i+1}" for i in range(num_groups)]

    # Create balanced group assignments
    # Each group appears (num_rows // num_groups) times, remainder distributed
    base_count = num_rows // num_groups
    remainder = num_rows % num_groups
    group_assignments = []
    for i, group in enumerate(group_ids):
        count = base_count + (1 if i < remainder else 0)
        group_assignments.extend([group] * count)

    # Shuffle the group assignments
    random.shuffle(group_assignments)

    # Create rows with random values based on style
    rows = []
    for i, group in enumerate(group_assignments):
        if value_style == "small":
            value = random.randint(0, 10)
        elif value_style == "large":
            value = random.randint(0, 100)
        elif value_style == "dollars":
            # Generate cents from 100 to 1000 ($1.00 to $10.00)
            cents = random.randint(100, 1000)
            value = f"{cents // 100}.{cents % 100:02d}"
        else:
            value = random.randint(0, 10)
        rows.append({"txnid": i + 1, "groupid": group, "value": value})

    # Create CSV string (use Unix line endings)
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=["txnid", "groupid", "value"], lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    csv_string = output.getvalue()

    # Compute expected sums
    if value_style == "dollars":
        # For dollars, compute sums in cents (as integers) for precise comparison
        # Then convert to dollar strings
        cents_by_group = {}
        for row in rows:
            group = row["groupid"]
            # Parse "X.YY" to cents
            parts = row["value"].split(".")
            cents = int(parts[0]) * 100 + int(parts[1])
            cents_by_group[group] = cents_by_group.get(group, 0) + cents
        # Convert cents to dollar strings like "12.34"
        expected = {g: f"{c // 100}.{c % 100:02d}" for g, c in cents_by_group.items()}
    else:
        df = pd.DataFrame(rows)
        expected = df.groupby("groupid")["value"].sum().to_dict()

    return csv_string, expected


def parse_response(response: str, value_style: str = "small") -> dict | None:
    """
    Parse LLM response to extract group sums.

    Handles various response formats:
    - Clean CSV
    - CSV with markdown code blocks
    - CSV with extra whitespace
    """
    if not response:
        return None

    # Remove markdown code blocks if present
    response = re.sub(r"```(?:csv)?\n?", "", response)
    response = response.strip()

    # Try to find CSV-like content
    lines = response.strip().split("\n")

    result = {}
    header_found = False

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Skip header line
        if "groupid" in line.lower() and ("sum" in line.lower() or "value" in line.lower()):
            header_found = True
            continue

        # Try to parse as CSV row
        parts = line.split(",")
        if len(parts) >= 2:
            try:
                group_id = parts[0].strip()
                # Handle potential quotes
                group_id = group_id.strip('"\'')
                value_str = parts[1].strip().strip('"\'')
                # Remove $ sign if present
                value_str = value_str.replace("$", "")

                if value_style == "dollars":
                    # Normalize to X.YY format
                    if "." in value_str:
                        int_part, dec_part = value_str.split(".")
                        dec_part = dec_part[:2].ljust(2, "0")  # Ensure 2 decimal places
                        value = f"{int(int_part)}.{dec_part}"
                    else:
                        value = f"{int(value_str)}.00"
                else:
                    # Handle potential decimal points (round to int)
                    value = int(float(value_str))
                result[group_id] = value
            except (ValueError, IndexError):
                continue

    return result if result else None


def check_correct(expected: dict[str, int], actual: dict[str, int] | None) -> bool:
    """Check if actual results match expected (order independent)."""
    if actual is None:
        return False

    if set(expected.keys()) != set(actual.keys()):
        return False

    for key in expected:
        if expected[key] != actual.get(key):
            return False

    return True


def check_api_keys(models: list[str]):
    """Check that required API keys are set before running benchmark."""
    needs_openai = any(
        m in OPENAI_MODELS or m.startswith("gpt-")
        for m in models
    )
    needs_anthropic = any(
        m in ANTHROPIC_MODELS or m.startswith("claude-")
        for m in models
    )

    missing = []
    if needs_openai and not os.environ.get("OPENAI_API_KEY"):
        missing.append("OPENAI_API_KEY")
    if needs_anthropic and not os.environ.get("ANTHROPIC_API_KEY"):
        missing.append("ANTHROPIC_API_KEY")

    if missing:
        print(f"Error: Missing required API key(s): {', '.join(missing)}", file=sys.stderr)
        print(f"\nSet them with:", file=sys.stderr)
        for key in missing:
            print(f"  export {key}=\"your-key-here\"", file=sys.stderr)
        sys.exit(1)


def create_chat_client(model: str, config: BenchmarkConfig):
    """Create appropriate chat client based on model name."""
    from chatlas import ChatOllama, ChatOpenAI, ChatAnthropic

    if model in OLLAMA_MODELS or model in OLLAMA_MODELS_SHORT or model.startswith("ollama:"):
        model_name = model.replace("ollama:", "")
        return ChatOllama(
            model=model_name,
            base_url=config.ollama_base_url,
        )
    elif model in OPENAI_MODELS or model.startswith("gpt-"):
        return ChatOpenAI(
            model=model,
            system_prompt=SYSTEM_PROMPT,
        )
    elif model in ANTHROPIC_MODELS or model.startswith("claude-"):
        return ChatAnthropic(
            model=model,
            system_prompt=SYSTEM_PROMPT,
        )
    else:
        raise ValueError(f"Unknown model: {model}")


def is_retriable_error(error: Exception) -> bool:
    """Check if an error is retriable (timeout, connection error, etc.)."""
    error_str = str(error).lower()
    retriable_patterns = [
        "timed out",
        "timeout",
        "connection",
        "refused",
        "reset",
        "broken pipe",
        "network",
        "unavailable",
        "502",
        "503",
        "504",
    ]
    return any(pattern in error_str for pattern in retriable_patterns)


def wait_for_server(base_url: str, max_wait: int = 300) -> bool:
    """Wait for Ollama server to become available."""
    import time
    import urllib.request
    import urllib.error

    print(f"    Waiting for server at {base_url} to become available...", file=sys.stderr)
    start = time.time()
    while time.time() - start < max_wait:
        try:
            urllib.request.urlopen(f"{base_url}/api/tags", timeout=5)
            print(f"    Server is back online!", file=sys.stderr)
            return True
        except (urllib.error.URLError, TimeoutError, OSError):
            time.sleep(5)
    return False


def run_single_trial(
    model: str,
    num_groups: int,
    num_rows: int,
    trial_num: int,
    config: BenchmarkConfig,
    max_retries: int = 3,
    retry_delay: int = 10,
) -> TrialResult:
    """Run a single benchmark trial with retry logic for transient failures."""
    import time

    # Generate data with unique seed per trial
    seed = None
    if config.seed is not None:
        seed = config.seed + hash((model, num_groups, num_rows, trial_num)) % (2**31)

    csv_data, expected = generate_data(num_groups, num_rows, seed, config.value_style)

    # Create prompt
    prompt = USER_PROMPT_TEMPLATE.format(csv_data=csv_data)

    last_error = None
    for attempt in range(max_retries + 1):
        try:
            # Create client and get response
            client = create_chat_client(model, config)

            start_time = time.time()

            # For Ollama, we need to include system prompt in the message
            if model in OLLAMA_MODELS or model in OLLAMA_MODELS_SHORT or model.startswith("ollama:"):
                full_prompt = f"{SYSTEM_PROMPT}\n\n{prompt}"
                response = client.chat(full_prompt, echo="none")
            else:
                response = client.chat(prompt, echo="none")

            latency_ms = (time.time() - start_time) * 1000

            # Extract text from response
            raw_response = str(response)

            # Parse response
            actual = parse_response(raw_response, config.value_style)
            correct = check_correct(expected, actual)

            return TrialResult(
                model=model,
                num_groups=num_groups,
                num_rows=num_rows,
                trial_num=trial_num,
                correct=correct,
                expected=expected,
                actual=actual,
                raw_response=raw_response,
                latency_ms=latency_ms,
            )

        except Exception as e:
            last_error = e
            if attempt < max_retries and is_retriable_error(e):
                print(f"    Trial {trial_num + 1}: Retry {attempt + 1}/{max_retries} after error: {str(e)[:50]}", file=sys.stderr)

                # For Ollama models, try to wait for server to recover
                if model in OLLAMA_MODELS or model in OLLAMA_MODELS_SHORT or model.startswith("ollama:"):
                    if not wait_for_server(config.ollama_base_url, max_wait=120):
                        print(f"    Server did not recover, giving up on retries", file=sys.stderr)
                        break
                else:
                    time.sleep(retry_delay * (attempt + 1))  # Exponential backoff
            else:
                break

    # All retries exhausted or non-retriable error
    return TrialResult(
        model=model,
        num_groups=num_groups,
        num_rows=num_rows,
        trial_num=trial_num,
        correct=False,
        expected=expected,
        actual=None,
        raw_response="",
        error=str(last_error),
    )


def load_checkpoint(checkpoint_file: str) -> list[TrialResult]:
    """Load results from a checkpoint file."""
    with open(checkpoint_file) as f:
        raw_data = json.load(f)

    results = []
    for r in raw_data:
        results.append(TrialResult(
            model=r["model"],
            num_groups=r["num_groups"],
            num_rows=r["num_rows"],
            trial_num=r["trial_num"],
            correct=r["correct"],
            expected=r["expected"],
            actual=r["actual"],
            raw_response=r["raw_response"],
            error=r.get("error"),
            latency_ms=r.get("latency_ms"),
        ))
    return results


def get_completed_trials(results: list[TrialResult]) -> set[tuple]:
    """Get set of (model, num_groups, num_rows, trial_num) for completed trials."""
    return {(r.model, r.num_groups, r.num_rows, r.trial_num) for r in results}


def save_checkpoint(results: list[TrialResult], config: BenchmarkConfig, checkpoint_file: Path):
    """Save checkpoint of partial results."""
    raw_data = [
        {
            "model": r.model,
            "num_groups": r.num_groups,
            "num_rows": r.num_rows,
            "trial_num": r.trial_num,
            "correct": r.correct,
            "expected": r.expected,
            "actual": r.actual,
            "raw_response": r.raw_response,
            "error": r.error,
            "latency_ms": r.latency_ms,
        }
        for r in results
    ]

    with open(checkpoint_file, "w") as f:
        json.dump(raw_data, f, indent=2)


def run_benchmark(config: BenchmarkConfig, previous_results: list[TrialResult] = None) -> list[TrialResult]:
    """Run full benchmark suite, optionally resuming from previous results."""
    results = list(previous_results) if previous_results else []
    completed_trials = get_completed_trials(results) if results else set()

    # Create checkpoint file with timestamp
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_file = output_dir / f"checkpoint_{timestamp}.json"

    # Calculate total trials (excluding configurations where groups >= rows)
    valid_cases = sum(
        1 for g in config.num_groups for r in config.num_rows if g < r
    )
    total_trials = len(config.models) * valid_cases * config.trials_per_case
    remaining_trials = total_trials - len(completed_trials)

    print(f"Running benchmark with {total_trials} total trials")
    if completed_trials:
        print(f"Resuming: {len(completed_trials)} trials already completed, {remaining_trials} remaining")
    print(f"Models: {config.models}")
    print(f"Groups: {config.num_groups}")
    print(f"Rows: {config.num_rows}")
    print(f"Trials per case: {config.trials_per_case}")
    print(f"Value style: {config.value_style}")
    print(f"Checkpoint file: {checkpoint_file}")
    print()

    for model in config.models:
        print(f"{'='*60}")
        print(f"MODEL: {model}")
        print(f"{'='*60}")

        for num_groups in config.num_groups:
            for num_rows in config.num_rows:
                # Skip configurations where groups >= rows
                if num_groups >= num_rows:
                    continue

                print(f"\n  {num_groups:2d} groups x {num_rows:4d} rows:")
                case_results = []

                for trial_num in range(config.trials_per_case):
                    # Skip if already completed
                    trial_key = (model, num_groups, num_rows, trial_num)
                    if trial_key in completed_trials:
                        # Find the existing result for summary
                        existing = next(r for r in results if (r.model, r.num_groups, r.num_rows, r.trial_num) == trial_key)
                        case_results.append(existing)
                        print(f"    [skipped] Trial {trial_num + 1:2d}: already completed")
                        continue

                    result = run_single_trial(
                        model, num_groups, num_rows, trial_num, config
                    )
                    results.append(result)
                    case_results.append(result)

                    # Format timestamp and elapsed time
                    now = datetime.now().strftime("%H:%M:%S")
                    elapsed = f"{result.latency_ms / 1000:.1f}s" if result.latency_ms else "N/A"

                    # Print trial result
                    if result.error:
                        print(f"    [{now}] Trial {trial_num + 1:2d}: ERROR - {result.error[:50]}")
                    elif result.correct:
                        print(f"    [{now}] Trial {trial_num + 1:2d}: CORRECT ({elapsed})")
                    else:
                        # Show what went wrong
                        diff_info = []
                        if result.actual is None:
                            diff_info.append("failed to parse response")
                        else:
                            for k in set(list(result.expected.keys()) + list(result.actual.keys())):
                                exp = result.expected.get(k)
                                act = result.actual.get(k)
                                if exp != act:
                                    diff_info.append(f"{k}: expected {exp}, got {act}")
                        print(f"    [{now}] Trial {trial_num + 1:2d}: WRONG ({elapsed}) - {'; '.join(diff_info[:3])}")

                # Print case summary
                correct = sum(r.correct for r in case_results)
                total = len(case_results)
                accuracy = correct / total if total > 0 else 0
                status = "PASS" if accuracy == 1.0 else ("FAIL" if accuracy == 0 else "PARTIAL")
                print(f"    Summary: {correct}/{total} ({accuracy:.0%}) {status}")

                # Save checkpoint after each configuration
                save_checkpoint(results, config, checkpoint_file)

        print()

    return results


def summarize_results(results: list[TrialResult]) -> pd.DataFrame:
    """Create summary DataFrame from results."""
    rows = []

    # Group by model, num_groups, num_rows
    from itertools import groupby
    from operator import attrgetter

    results_sorted = sorted(results, key=lambda r: (r.model, r.num_groups, r.num_rows))

    for (model, num_groups, num_rows), group in groupby(
        results_sorted, key=lambda r: (r.model, r.num_groups, r.num_rows)
    ):
        trials = list(group)
        correct_count = sum(r.correct for r in trials)
        total_count = len(trials)
        accuracy = correct_count / total_count if total_count > 0 else 0
        avg_latency = np.mean([r.latency_ms for r in trials if r.latency_ms > 0])
        error_count = sum(1 for r in trials if r.error is not None)

        rows.append({
            "model": model,
            "num_groups": num_groups,
            "num_rows": num_rows,
            "correct": correct_count,
            "total": total_count,
            "accuracy": accuracy,
            "avg_latency_ms": avg_latency,
            "errors": error_count,
        })

    return pd.DataFrame(rows)


def save_results(results: list[TrialResult], config: BenchmarkConfig):
    """Save results to files."""
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save raw results as JSON
    raw_file = output_dir / f"raw_results_{timestamp}.json"
    raw_data = [
        {
            "model": r.model,
            "num_groups": r.num_groups,
            "num_rows": r.num_rows,
            "trial_num": r.trial_num,
            "correct": r.correct,
            "expected": r.expected,
            "actual": r.actual,
            "raw_response": r.raw_response,
            "error": r.error,
            "latency_ms": r.latency_ms,
        }
        for r in results
    ]
    with open(raw_file, "w") as f:
        json.dump(raw_data, f, indent=2)
    print(f"Saved raw results to {raw_file}")

    # Save summary as CSV
    summary = summarize_results(results)
    summary_file = output_dir / f"summary_{timestamp}.csv"
    summary.to_csv(summary_file, index=False)
    print(f"Saved summary to {summary_file}")

    # Print summary table
    print("\n" + "=" * 80)
    print("BENCHMARK SUMMARY")
    print("=" * 80)

    # Pivot table for easy viewing
    pivot = summary.pivot_table(
        values="accuracy",
        index=["num_groups", "num_rows"],
        columns="model",
        aggfunc="first",
    )
    print(pivot.to_string(float_format=lambda x: f"{x:.0%}"))

    return summary


def main():
    parser = argparse.ArgumentParser(description="LLM Group Counting Benchmark")
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Models to benchmark (default: all configured models)",
    )
    parser.add_argument(
        "--ollama-only",
        action="store_true",
        help="Only run Ollama models",
    )
    parser.add_argument(
        "--openai-only",
        action="store_true",
        help="Only run OpenAI models",
    )
    parser.add_argument(
        "--anthropic-only",
        action="store_true",
        help="Only run Anthropic models",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=10,
        help="Number of trials per case (default: 10)",
    )
    parser.add_argument(
        "--ollama-url",
        default=None,
        help="Ollama API base URL (overrides --ollama-host)",
    )
    parser.add_argument(
        "--ollama-host",
        default="dgx-spark",
        choices=["dgx-spark", "home-desktop", "localhost"],
        help="Ollama host machine (default: dgx-spark)",
    )
    parser.add_argument(
        "--output-dir",
        default="results",
        help="Output directory for results",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick test: fewer groups/rows/trials",
    )
    parser.add_argument(
        "--row-sweep",
        nargs=3,
        type=int,
        metavar=("START", "END", "STEP"),
        help="Sweep rows from START to END by STEP with 1 group (e.g., --row-sweep 10 50 5)",
    )
    parser.add_argument(
        "--groups",
        type=int,
        default=1,
        help="Number of groups for --row-sweep mode (default: 1)",
    )
    parser.add_argument(
        "--value-style",
        choices=VALUE_STYLES,
        default="small",
        help="Value style: small (0-10), large (0-100), dollars ($1.00-$10.00)",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        metavar="CHECKPOINT_FILE",
        help="Resume from a checkpoint file, running only missing trials",
    )

    args = parser.parse_args()

    # Determine models to test
    if args.models:
        models = args.models
    elif args.ollama_only:
        models = OLLAMA_MODELS
    elif args.openai_only:
        models = OPENAI_MODELS
    elif args.anthropic_only:
        models = ANTHROPIC_MODELS
    else:
        models = OLLAMA_MODELS + OPENAI_MODELS + ANTHROPIC_MODELS

    # Validate API keys before starting
    check_api_keys(models)

    # Determine Ollama URL
    if args.ollama_url:
        ollama_base_url = args.ollama_url
    else:
        ollama_base_url = f"http://{args.ollama_host}:11434"

    # Create config
    if args.row_sweep:
        start, end, step = args.row_sweep
        row_values = list(range(start, end + 1, step))
        config = BenchmarkConfig(
            models=models,
            num_groups=[args.groups],
            num_rows=row_values,
            trials_per_case=args.trials,
            ollama_base_url=ollama_base_url,
            output_dir=args.output_dir,
            seed=args.seed,
            value_style=args.value_style,
        )
    elif args.quick:
        config = BenchmarkConfig(
            models=models,
            num_groups=[1, 3, 10],
            num_rows=[10, 50, 100],
            trials_per_case=3,
            ollama_base_url=ollama_base_url,
            output_dir=args.output_dir,
            seed=args.seed,
            value_style=args.value_style,
        )
    else:
        config = BenchmarkConfig(
            models=models,
            trials_per_case=args.trials,
            ollama_base_url=ollama_base_url,
            output_dir=args.output_dir,
            seed=args.seed,
            value_style=args.value_style,
        )

    # Load previous results if resuming
    previous_results = None
    if args.resume:
        if not Path(args.resume).exists():
            print(f"Error: Checkpoint file not found: {args.resume}", file=sys.stderr)
            sys.exit(1)
        print(f"Loading checkpoint from: {args.resume}")
        previous_results = load_checkpoint(args.resume)
        print(f"Loaded {len(previous_results)} previous results")

    # Run benchmark
    results = run_benchmark(config, previous_results)

    # Save and display results
    save_results(results, config)


if __name__ == "__main__":
    main()
