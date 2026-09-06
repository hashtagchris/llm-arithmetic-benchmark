#!/usr/bin/env python3
"""Benchmark LLM accuracy at completing a Markdown pricing table."""

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from decimal import Decimal, InvalidOperation
from datetime import datetime
from pathlib import Path


SCRIPT_DIR = Path(__file__).parent
PROMPT_PATH = SCRIPT_DIR / "prompt.md"
EXPECTED_RESPONSE_PATH = SCRIPT_DIR / "expected_response.md"

OLLAMA_MODELS = [
    "gpt-oss-128k",
    "gpt-oss-120b-128k",
    "qwen2.5-coder-256k",
]

OLLAMA_MODELS_SHORT = [
    "gpt-oss:20b",
    "gpt-oss:120b",
    "qwen2.5-coder:32b",
]

OPENAI_MODELS = [
    "gpt-4o",
    "gpt-4o-mini",
    "gpt-4.1",
    "gpt-4.1-mini",
    "gpt-4.1-nano",
]

ANTHROPIC_MODELS = [
    "claude-3-7-sonnet-20250219",
    "claude-3-5-haiku-20241022",
    "claude-sonnet-4-20250514",
    "claude-sonnet-4-5-20250929",
    "claude-haiku-4-5-20251001",
    "claude-opus-4-5-20251101",
]

SYSTEM_PROMPT = (
    "Complete the requested Markdown table using only your internal reasoning. "
    "Do not use tools or code execution. Return only the completed table."
)

NUMERIC_CELL_PATTERN = re.compile(
    r"^\s*\$?\s*"
    r"([+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)"
    r"\s*%?\s*(?:per\s+(?:GB|day))?\s*$",
    re.IGNORECASE,
)


@dataclass
class TrialResult:
    """Result of one pricing-table completion attempt."""

    model: str
    trial_num: int
    correct: bool
    expected: list[list[str]]
    actual: list[list[str]] | None
    raw_response: str
    warnings: list[str] = field(default_factory=list)
    differing_cells: int = 0
    error: str | None = None
    latency_ms: float = 0.0


@dataclass
class BenchmarkConfig:
    """Configuration for a benchmark run."""

    models: list[str]
    trials_per_model: int = 10
    ollama_base_url: str = "http://dgx-spark:11434"
    output_dir: str = "results"
    verbose: bool = False


def load_benchmark_assets(
    prompt_path: Path = PROMPT_PATH,
    expected_path: Path = EXPECTED_RESPONSE_PATH,
) -> tuple[str, str]:
    """Load the model prompt and canonical expected response."""
    return prompt_path.read_text(), expected_path.read_text()


def _is_separator_row(cells: list[str]) -> bool:
    """Return whether cells form a Markdown table separator row."""
    return bool(cells) and all(
        bool(cell) and not cell.replace("-", "").replace(":", "").strip()
        for cell in cells
    )


def normalize_markdown_table(markdown: str) -> list[list[str]] | None:
    """Normalize a Markdown table while ignoring presentation-only spacing."""
    rows = []

    for raw_line in markdown.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("```") or "|" not in line:
            continue

        cells = line.strip("|").split("|")
        normalized_cells = [" ".join(cell.split()) for cell in cells]

        if _is_separator_row(normalized_cells):
            continue
        rows.append(normalized_cells)

    if not rows:
        return None

    column_count = len(rows[0])
    if column_count < 2 or any(len(row) != column_count for row in rows):
        return None

    return rows


def format_normalized_table(table: list[list[str]] | None) -> str:
    """Render normalized cells for diagnostic output."""
    if table is None:
        return "<unparseable>"
    return "\n".join(" | ".join(row) for row in table)


def check_correct(
    expected: list[list[str]],
    actual: list[list[str]] | None,
) -> bool:
    """Check tables, accepting numerically equivalent data cells."""
    correct, _ = compare_tables(expected, actual)
    return correct


def _numeric_cell_value(cell: str) -> Decimal | None:
    """Extract a numeric value from a currency, percentage, or rate cell."""
    match = NUMERIC_CELL_PATTERN.fullmatch(cell)
    if not match:
        return None
    try:
        return Decimal(match.group(1).replace(",", ""))
    except InvalidOperation:
        return None


def _cell_location(
    row_index: int,
    column_index: int,
) -> str:
    """Format a stable row and column location for diagnostics."""
    return f"row {row_index + 1}, column {column_index + 1}"


def _cells_match(
    expected_cell: str,
    actual_cell: str,
    row_index: int,
    column_index: int,
) -> tuple[bool, bool]:
    """Return whether cells match and whether numeric tolerance was required."""
    if expected_cell == actual_cell:
        return True, False
    if row_index == 0 or column_index == 0:
        return False, False

    expected_value = _numeric_cell_value(expected_cell)
    actual_value = _numeric_cell_value(actual_cell)
    numerically_equal = (
        expected_value is not None
        and actual_value is not None
        and expected_value == actual_value
    )
    return numerically_equal, numerically_equal


def compare_tables(
    expected: list[list[str]],
    actual: list[list[str]] | None,
) -> tuple[bool, list[str]]:
    """Compare table structure and cells, returning accepted-value warnings."""
    if actual is None or len(expected) != len(actual):
        return False, []

    expected_cells = sum(len(row) for row in expected)
    actual_cells = sum(len(row) for row in actual)
    if expected_cells != actual_cells:
        return False, []

    warnings = []
    correct = True
    for row_index, (expected_row, actual_row) in enumerate(zip(expected, actual)):
        if len(expected_row) != len(actual_row):
            return False, []
        for column_index, (expected_cell, actual_cell) in enumerate(
            zip(expected_row, actual_row)
        ):
            matches, numeric_warning = _cells_match(
                expected_cell,
                actual_cell,
                row_index,
                column_index,
            )
            if not matches:
                correct = False
            elif numeric_warning:
                warnings.append(
                    "cell differs in formatting but is numerically equal: "
                    f"expected {expected_cell!r}, got {actual_cell!r}"
                )

    return correct, warnings


def count_differing_cells(
    expected: list[list[str]],
    actual: list[list[str]] | None,
) -> int:
    """Count textually different, missing, or extra cells."""
    if actual is None:
        return sum(len(row) for row in expected)

    differing = 0
    row_count = max(len(expected), len(actual))
    for row_index in range(row_count):
        expected_row = expected[row_index] if row_index < len(expected) else []
        actual_row = actual[row_index] if row_index < len(actual) else []
        column_count = max(len(expected_row), len(actual_row))

        for column_index in range(column_count):
            expected_cell = (
                expected_row[column_index]
                if column_index < len(expected_row)
                else None
            )
            actual_cell = (
                actual_row[column_index] if column_index < len(actual_row) else None
            )
            if expected_cell != actual_cell:
                differing += 1

    return differing


def describe_mismatch(
    expected: list[list[str]],
    actual: list[list[str]] | None,
) -> str:
    """Describe a structural mismatch or the first three differing cells."""
    if actual is None:
        return "response did not contain a parseable rectangular Markdown table"

    if len(expected) != len(actual):
        return f"row count mismatch: expected {len(expected)}, got {len(actual)}"

    expected_cells = sum(len(row) for row in expected)
    actual_cells = sum(len(row) for row in actual)
    if expected_cells != actual_cells:
        return f"cell count mismatch: expected {expected_cells}, got {actual_cells}"

    differences = []
    for row_index, (expected_row, actual_row) in enumerate(zip(expected, actual)):
        for column_index, (expected_cell, actual_cell) in enumerate(
            zip(expected_row, actual_row)
        ):
            matches, _ = _cells_match(
                expected_cell,
                actual_cell,
                row_index,
                column_index,
            )
            if matches:
                continue

            location = _cell_location(row_index, column_index)
            differences.append(
                f"{location}: expected {expected_cell!r}, got {actual_cell!r}"
            )
            if len(differences) == 3:
                return "first differing cells: " + "; ".join(differences)

    if differences:
        return "differing cells: " + "; ".join(differences)
    return "tables differ"


def log_verbose_comparison(
    expected: list[list[str]],
    actual: list[list[str]] | None,
) -> None:
    """Log expected and actual normalized tables."""
    print("    Expected:", file=sys.stderr)
    print(format_normalized_table(expected), file=sys.stderr)
    print("    Actual:", file=sys.stderr)
    print(format_normalized_table(actual), file=sys.stderr)


def check_api_keys(models: list[str]) -> None:
    """Check that required provider credentials and executables are available."""
    needs_openai = any(
        (m in OPENAI_MODELS or m.startswith("gpt-"))
        and not is_ollama_model(m)
        and not is_copilot_model(m)
        for m in models
    )
    needs_anthropic = any(
        (m in ANTHROPIC_MODELS or m.startswith("claude-"))
        and not is_copilot_model(m)
        for m in models
    )

    missing = []
    if needs_openai and not os.environ.get("OPENAI_API_KEY"):
        missing.append("OPENAI_API_KEY")
    if needs_anthropic and not os.environ.get("ANTHROPIC_API_KEY"):
        missing.append("ANTHROPIC_API_KEY")

    if missing:
        print(
            f"Error: Missing required API key(s): {', '.join(missing)}",
            file=sys.stderr,
        )
        print("\nSet them with:", file=sys.stderr)
        for key in missing:
            print(f'  export {key}="your-key-here"', file=sys.stderr)
        raise SystemExit(1)

    if any(is_copilot_model(model) for model in models) and not shutil.which("copilot"):
        print(
            "Error: The copilot executable is required for copilot: models",
            file=sys.stderr,
        )
        raise SystemExit(1)


def create_chat_client(model: str, config: BenchmarkConfig):
    """Create the appropriate chat client for a model."""
    from chatlas import ChatAnthropic, ChatOllama, ChatOpenAI

    if (
        model in OLLAMA_MODELS
        or model in OLLAMA_MODELS_SHORT
        or model.startswith("ollama:")
    ):
        return ChatOllama(
            model=model.replace("ollama:", ""),
            base_url=config.ollama_base_url,
        )
    if model in OPENAI_MODELS or model.startswith("gpt-"):
        return ChatOpenAI(model=model, system_prompt=SYSTEM_PROMPT)
    if model in ANTHROPIC_MODELS or model.startswith("claude-"):
        return ChatAnthropic(model=model, system_prompt=SYSTEM_PROMPT)
    raise ValueError(f"Unknown model: {model}")


def is_copilot_model(model: str) -> bool:
    """Return whether a model should run through GitHub Copilot CLI."""
    return model.startswith("copilot:")


def is_ollama_model(model: str) -> bool:
    """Return whether a model is served by Ollama."""
    return (
        model in OLLAMA_MODELS
        or model in OLLAMA_MODELS_SHORT
        or model.startswith("ollama:")
    )


def get_model_response(model: str, config: BenchmarkConfig, prompt: str) -> str:
    """Run a prompt through the selected provider and return response text."""
    if is_copilot_model(model):
        copilot_model = model.removeprefix("copilot:")
        if not copilot_model:
            raise ValueError("Copilot model name cannot be empty")

        command = [
            "copilot",
            "--model",
            copilot_model,
            "--available-tools=",
            "--allow-all-tools",
            "--disable-builtin-mcps",
            "--no-custom-instructions",
            "--no-remote",
            "--no-remote-export",
            "--silent",
            "--no-color",
            "--prompt",
            prompt,
        ]
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            detail = result.stderr.strip() or result.stdout.strip()
            raise RuntimeError(
                f"Copilot CLI exited with status {result.returncode}: {detail}"
            )
        return result.stdout.strip()

    client = create_chat_client(model, config)
    if is_ollama_model(model):
        return str(client.chat(f"{SYSTEM_PROMPT}\n\n{prompt}", echo="none"))
    return str(client.chat(prompt, echo="none"))


def is_retriable_error(error: Exception) -> bool:
    """Return whether an error is likely transient."""
    error_text = str(error).lower()
    patterns = [
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
    return any(pattern in error_text for pattern in patterns)


def wait_for_server(base_url: str, max_wait: int = 300) -> bool:
    """Wait for an Ollama server to become available."""
    print(
        f"    Waiting for server at {base_url} to become available...",
        file=sys.stderr,
    )
    start = time.time()
    while time.time() - start < max_wait:
        try:
            urllib.request.urlopen(f"{base_url}/api/tags", timeout=5)
            print("    Server is back online!", file=sys.stderr)
            return True
        except (urllib.error.URLError, TimeoutError, OSError):
            time.sleep(5)
    return False


def run_single_trial(
    model: str,
    trial_num: int,
    config: BenchmarkConfig,
    prompt: str,
    expected: list[list[str]],
    max_retries: int = 3,
    retry_delay: int = 10,
) -> TrialResult:
    """Run one benchmark trial with retry logic for transient failures."""
    last_error = None

    for attempt in range(max_retries + 1):
        try:
            start_time = time.time()
            raw_response = get_model_response(model, config, prompt)
            latency_ms = (time.time() - start_time) * 1000
            actual = normalize_markdown_table(raw_response)
            correct, warnings = compare_tables(expected, actual)
            differing_cells = count_differing_cells(expected, actual)

            for warning in warnings:
                print(f"    WARNING: {warning}", file=sys.stderr)

            if config.verbose:
                log_verbose_comparison(expected, actual)

            return TrialResult(
                model=model,
                trial_num=trial_num,
                correct=correct,
                expected=expected,
                actual=actual,
                raw_response=raw_response,
                warnings=warnings,
                differing_cells=differing_cells,
                latency_ms=latency_ms,
            )
        except Exception as error:
            last_error = error
            if attempt >= max_retries or not is_retriable_error(error):
                break

            print(
                f"    Trial {trial_num + 1}: Retry {attempt + 1}/{max_retries} "
                f"after error: {str(error)[:50]}",
                file=sys.stderr,
            )
            if is_ollama_model(model):
                if not wait_for_server(config.ollama_base_url, max_wait=120):
                    print(
                        "    Server did not recover, giving up on retries",
                        file=sys.stderr,
                    )
                    break
            else:
                time.sleep(retry_delay * (attempt + 1))

    if config.verbose:
        log_verbose_comparison(expected, None)

    return TrialResult(
        model=model,
        trial_num=trial_num,
        correct=False,
        expected=expected,
        actual=None,
        raw_response="",
        differing_cells=count_differing_cells(expected, None),
        error=str(last_error),
    )


def load_checkpoint(checkpoint_file: str) -> list[TrialResult]:
    """Load prior trial results from a checkpoint."""
    with open(checkpoint_file) as file:
        return [TrialResult(**result) for result in json.load(file)]


def save_checkpoint(results: list[TrialResult], checkpoint_file: Path) -> None:
    """Save partial trial results."""
    with open(checkpoint_file, "w") as file:
        json.dump([asdict(result) for result in results], file, indent=2)


def run_benchmark(
    config: BenchmarkConfig,
    previous_results: list[TrialResult] | None = None,
) -> list[TrialResult]:
    """Run all configured pricing-table trials."""
    prompt, expected_markdown = load_benchmark_assets()
    expected = normalize_markdown_table(expected_markdown)
    if expected is None:
        raise ValueError(f"Invalid expected response table: {EXPECTED_RESPONSE_PATH}")

    results = list(previous_results) if previous_results else []
    completed_trials = {(result.model, result.trial_num) for result in results}

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_file = output_dir / f"checkpoint_{timestamp}.json"
    total_trials = len(config.models) * config.trials_per_model

    print(f"Running benchmark with {total_trials} total trials")
    if completed_trials:
        remaining = total_trials - len(completed_trials)
        print(
            f"Resuming: {len(completed_trials)} trials already completed, "
            f"{remaining} remaining"
        )
    print(f"Models: {config.models}")
    print(f"Trials per model: {config.trials_per_model}")
    print(f"Checkpoint file: {checkpoint_file}\n")

    for model in config.models:
        print(f"{'=' * 60}\nMODEL: {model}\n{'=' * 60}")
        model_results = []

        for trial_num in range(config.trials_per_model):
            trial_key = (model, trial_num)
            if trial_key in completed_trials:
                result = next(
                    item
                    for item in results
                    if (item.model, item.trial_num) == trial_key
                )
                model_results.append(result)
                print(f"  [skipped] Trial {trial_num + 1:2d}: already completed")
                continue

            result = run_single_trial(
                model,
                trial_num,
                config,
                prompt,
                expected,
            )
            results.append(result)
            model_results.append(result)

            now = datetime.now().strftime("%H:%M:%S")
            elapsed = f"{result.latency_ms / 1000:.1f}s" if result.latency_ms else "N/A"
            if result.error:
                print(
                    f"  [{now}] Trial {trial_num + 1:2d}: "
                    f"ERROR - {result.error[:50]}"
                )
            elif result.correct:
                print(f"  [{now}] Trial {trial_num + 1:2d}: CORRECT ({elapsed})")
            else:
                mismatch = describe_mismatch(result.expected, result.actual)
                print(
                    f"  [{now}] Trial {trial_num + 1:2d}: "
                    f"WRONG ({elapsed}) - {mismatch}"
                )

            save_checkpoint(results, checkpoint_file)

        correct = sum(result.correct for result in model_results)
        total = len(model_results)
        accuracy = correct / total if total else 0
        status = "PASS" if accuracy == 1 else ("FAIL" if accuracy == 0 else "PARTIAL")
        print(f"  Summary: {correct}/{total} ({accuracy:.0%}) {status}\n")

    return results


def summarize_results(results: list[TrialResult]) -> list[dict]:
    """Aggregate accuracy, latency, and differing cells by model."""
    summaries = []
    for model in sorted({result.model for result in results}):
        model_results = [result for result in results if result.model == model]
        latencies = [
            result.latency_ms for result in model_results if result.latency_ms > 0
        ]
        correct = sum(result.correct for result in model_results)
        total = len(model_results)
        summaries.append(
            {
                "model": model,
                "correct": correct,
                "total": total,
                "accuracy": correct / total if total else 0,
                "avg_latency_seconds": (
                    f"{sum(latencies) / len(latencies) / 1000:.2f}"
                    if latencies
                    else "0.00"
                ),
                "avg_differing_cells": (
                    f"{sum(result.differing_cells for result in model_results) / total:.2f}"
                    if total
                    else "0.00"
                ),
                "errors": sum(result.error is not None for result in model_results),
            }
        )
    return summaries


def save_results(
    results: list[TrialResult],
    config: BenchmarkConfig,
) -> list[dict]:
    """Save raw and summarized benchmark results."""
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    raw_file = output_dir / f"raw_results_{timestamp}.json"
    with open(raw_file, "w") as file:
        json.dump([asdict(result) for result in results], file, indent=2)
    print(f"Saved raw results to {raw_file}")

    summary = summarize_results(results)
    summary_file = output_dir / f"summary_{timestamp}.csv"
    with open(summary_file, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=summary[0].keys())
        writer.writeheader()
        writer.writerows(summary)
    print(f"Saved summary to {summary_file}")

    print(f"\n{'=' * 64}\nBENCHMARK SUMMARY\n{'=' * 64}")
    print(f"{'Model':<35} {'Correct':>8} {'Total':>7} {'Accuracy':>10}")
    for row in summary:
        print(
            f"{row['model']:<35} {row['correct']:>8} {row['total']:>7} "
            f"{row['accuracy']:>9.0%}"
        )
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="LLM Markdown pricing-table completion benchmark"
    )
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
        help="Number of trials per model (default: 10)",
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
        "--quick",
        action="store_true",
        help="Run one trial per model",
    )
    parser.add_argument(
        "--resume",
        metavar="CHECKPOINT_FILE",
        help="Resume from a checkpoint file, running only missing trials",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Log normalized expected versus actual results for every trial",
    )
    return parser.parse_args(argv)


def select_models(args: argparse.Namespace) -> list[str]:
    """Select models from command-line filters."""
    if args.models:
        return args.models
    if args.ollama_only:
        return OLLAMA_MODELS
    if args.openai_only:
        return OPENAI_MODELS
    if args.anthropic_only:
        return ANTHROPIC_MODELS
    return OLLAMA_MODELS + OPENAI_MODELS + ANTHROPIC_MODELS


def main(argv: list[str] | None = None) -> int:
    """Run the benchmark CLI."""
    args = parse_args(argv)
    models = select_models(args)
    check_api_keys(models)

    ollama_base_url = args.ollama_url or f"http://{args.ollama_host}:11434"
    config = BenchmarkConfig(
        models=models,
        trials_per_model=1 if args.quick else args.trials,
        ollama_base_url=ollama_base_url,
        output_dir=args.output_dir,
        verbose=args.verbose,
    )

    previous_results = None
    if args.resume:
        if not Path(args.resume).exists():
            print(
                f"Error: Checkpoint file not found: {args.resume}",
                file=sys.stderr,
            )
            return 1
        print(f"Loading checkpoint from: {args.resume}")
        previous_results = load_checkpoint(args.resume)
        print(f"Loaded {len(previous_results)} previous results")

    results = run_benchmark(config, previous_results)
    save_results(results, config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
