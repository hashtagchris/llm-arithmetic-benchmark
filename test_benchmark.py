#!/usr/bin/env python3
"""Offline tests for the Markdown pricing-table benchmark."""

import io
import subprocess
from contextlib import redirect_stderr
from decimal import Decimal, ROUND_HALF_UP

import benchmark
from benchmark import (
    BenchmarkConfig,
    EXPECTED_RESPONSE_PATH,
    PROMPT_PATH,
    check_correct,
    compare_tables,
    count_differing_cells,
    describe_mismatch,
    format_normalized_table,
    get_model_response,
    load_benchmark_assets,
    normalize_markdown_table,
    parse_args,
    run_single_trial,
    summarize_results,
)


EXPECTED_MISSING_CELLS = {
    "400 GB per day": ("$3.52 per GB", "$3.52 per GB", "23.48%"),
    "500 GB per day": ("$3.46 per GB", "$3.46 per GB", "24.78%"),
    "1,000 GB per day": ("$3.40 per GB", "$3.40 per GB", "26.09%"),
    "2,000 GB per day": ("$3.32 per GB", "$3.32 per GB", "27.83%"),
    "5,000 GB per day": ("$3.22 per GB", "$3.22 per GB", "30.00%"),
    "10,000 GB per day": ("$3.13 per GB", "$3.13 per GB", "32.00%"),
    "25,000 GB per day": ("$3.04 per GB", "$3.04 per GB", "34.00%"),
    "50,000 GB per day": ("$2.94 per GB", "$2.94 per GB", "36.00%"),
}

TIER_PRICES = {
    "400 GB per day": (Decimal("400"), Decimal("1408")),
    "500 GB per day": (Decimal("500"), Decimal("1730")),
    "1,000 GB per day": (Decimal("1000"), Decimal("3400")),
    "2,000 GB per day": (Decimal("2000"), Decimal("6640")),
    "5,000 GB per day": (Decimal("5000"), Decimal("16100")),
    "10,000 GB per day": (Decimal("10000"), Decimal("31280")),
    "25,000 GB per day": (Decimal("25000"), Decimal("75900")),
    "50,000 GB per day": (Decimal("50000"), Decimal("147200")),
}


def rows_by_tier(table: list[list[str]]) -> dict[str, list[str]]:
    """Index normalized data rows by pricing tier."""
    return {row[0]: row for row in table[1:]}


def test_assets() -> None:
    """Verify dedicated prompt and expected-response files."""
    prompt, expected_markdown = load_benchmark_assets()

    assert PROMPT_PATH.name == "prompt.md"
    assert EXPECTED_RESPONSE_PATH.name == "expected_response.md"
    assert "400 GB per day | $1,408.00 per day |" in prompt
    assert "strict round-half-up" in prompt
    assert "exactly two decimal places for all USD values" in prompt
    assert "four significant figures" in prompt
    assert "first discarded digit is 5 or greater" in prompt
    assert "23.48%" not in prompt
    assert "$3.52 per GB" not in prompt
    assert "divide" not in prompt.lower()
    assert "savings =" not in prompt.lower()

    expected = normalize_markdown_table(expected_markdown)
    assert expected is not None
    assert len(expected) == 13
    assert all(len(row) == 5 for row in expected)


def test_expected_response_values() -> None:
    """Verify canonical missing cells and independent savings arithmetic."""
    _, expected_markdown = load_benchmark_assets()
    expected = normalize_markdown_table(expected_markdown)
    assert expected is not None
    tiers = rows_by_tier(expected)
    pay_as_you_go = Decimal("4.60")

    for row in tiers.values():
        for cell in row[1:4]:
            if cell != "N/A":
                amount = cell.split()[0]
                assert amount.startswith("$")
                assert len(amount.rsplit(".", 1)[-1]) == 2

    for tier, expected_cells in EXPECTED_MISSING_CELLS.items():
        assert tuple(tiers[tier][2:]) == expected_cells

        capacity, daily_price = TIER_PRICES[tier]
        unit_price = daily_price / capacity
        rounded_unit_price = unit_price.quantize(
            Decimal("0.01"),
            rounding=ROUND_HALF_UP,
        )
        savings = ((pay_as_you_go - unit_price) / pay_as_you_go) * 100
        rounded_savings = savings.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
        assert tiers[tier][2] == f"${rounded_unit_price:.2f} per GB"
        assert tiers[tier][3] == f"${rounded_unit_price:.2f} per GB"
        assert tiers[tier][4] == f"{rounded_savings:.2f}%"


def test_markdown_normalization() -> None:
    """Accept harmless Markdown formatting differences."""
    _, expected_markdown = load_benchmark_assets()
    expected = normalize_markdown_table(expected_markdown)
    assert expected is not None

    lines = expected_markdown.splitlines()
    variant_lines = ["```markdown", "", f"|  {lines[0].replace('|', '  |  ')}  |"]
    variant_lines.append("| :--- | ---: | :---: | --- | ---: |")
    variant_lines.extend(f"|  {line.replace('|', '  |  ')}  |" for line in lines[2:])
    variant_lines.extend(["", "```"])
    actual = normalize_markdown_table("\n".join(variant_lines))

    assert check_correct(expected, actual)


def test_incorrect_tables() -> None:
    """Reject changed, missing, extra, and malformed table cells."""
    _, expected_markdown = load_benchmark_assets()
    expected = normalize_markdown_table(expected_markdown)
    assert expected is not None

    wrong_value = normalize_markdown_table(
        expected_markdown.replace("23.48%", "23.49%")
    )
    missing_row = normalize_markdown_table(
        "\n".join(expected_markdown.splitlines()[:-1])
    )
    extra_row = normalize_markdown_table(
        expected_markdown + "\n100,000 GB per day | $1 per day | $1 | $1 | 1.000%"
    )
    malformed = normalize_markdown_table(
        expected_markdown.replace(
            "400 GB per day | $1,408.00 per day | $3.52 per GB",
            "400 GB per day | $1,408.00 per day $3.52 per GB",
        )
    )

    assert not check_correct(expected, wrong_value)
    assert not check_correct(expected, missing_row)
    assert not check_correct(expected, extra_row)
    assert not check_correct(expected, malformed)
    assert not check_correct(expected, None)


def test_numeric_equivalence_warnings() -> None:
    """Accept equal numbers with missing units or different formatting."""
    prompt, expected_markdown = load_benchmark_assets()
    expected = normalize_markdown_table(expected_markdown)
    assert expected is not None

    actual = [row.copy() for row in expected]
    actual[1][1] = "$4.600"
    actual[1][2] = "4.60"
    actual[5][1] = "$1,408.00"
    actual[5][2] = "$3.520"
    actual[5][3] = "3.52"
    actual[5][4] = "23.480"

    correct, warnings = compare_tables(expected, actual)
    assert correct
    assert len(warnings) == 6
    assert all("numerically equal" in warning for warning in warnings)

    class FakeClient:
        def chat(self, prompt: str, echo: str) -> str:
            return format_normalized_table(actual)

    original_create_chat_client = benchmark.create_chat_client
    benchmark.create_chat_client = lambda model, config: FakeClient()
    output = io.StringIO()
    try:
        with redirect_stderr(output):
            result = run_single_trial(
                "gpt-4o-mini",
                0,
                BenchmarkConfig(models=["gpt-4o-mini"]),
                prompt,
                expected,
            )
    finally:
        benchmark.create_chat_client = original_create_chat_client

    assert result.correct
    assert result.warnings == warnings
    assert result.differing_cells == 6
    assert output.getvalue().count("WARNING:") == 6
    assert "row " not in output.getvalue()
    assert "column " not in output.getvalue()

    wrong_label = [row.copy() for row in actual]
    wrong_label[5][0] = "400"
    assert not check_correct(expected, wrong_label)


def test_differing_cell_count_and_summary() -> None:
    """Count textual cell differences and summarize latency in seconds."""
    _, expected_markdown = load_benchmark_assets()
    expected = normalize_markdown_table(expected_markdown)
    assert expected is not None

    actual = [row.copy() for row in expected]
    actual[5][2] = "3.52"
    actual[5][3] = "$0.01 per GB"
    assert count_differing_cells(expected, actual) == 2
    assert count_differing_cells(expected, None) == 65
    assert count_differing_cells(expected, expected[:-1]) == 5

    results = [
        benchmark.TrialResult(
            model="model-a",
            trial_num=0,
            correct=True,
            expected=expected,
            actual=actual,
            raw_response="",
            differing_cells=2,
            latency_ms=1234,
        ),
        benchmark.TrialResult(
            model="model-a",
            trial_num=1,
            correct=False,
            expected=expected,
            actual=expected,
            raw_response="",
            differing_cells=3,
            latency_ms=2345,
        ),
    ]
    summary = summarize_results(results)

    assert summary[0]["avg_latency_seconds"] == "1.79"
    assert summary[0]["differing_cells"] == 5
    assert "avg_latency_ms" not in summary[0]


def test_mismatch_description() -> None:
    """Describe shape errors or at most the first three differing cells."""
    _, expected_markdown = load_benchmark_assets()
    expected = normalize_markdown_table(expected_markdown)
    assert expected is not None

    assert "parseable rectangular Markdown table" in describe_mismatch(expected, None)

    missing_row = [row.copy() for row in expected[:-1]]
    assert describe_mismatch(expected, missing_row) == (
        "row count mismatch: expected 13, got 12"
    )

    fewer_cells = [row[:-1] for row in expected]
    assert describe_mismatch(expected, fewer_cells) == (
        "cell count mismatch: expected 65, got 52"
    )

    different = [row.copy() for row in expected]
    different[4][2] = "$0.01 per GB"
    different[4][3] = "$0.02 per GB"
    different[4][4] = "0.03%"
    different[5][2] = "$0.04 per GB"
    description = describe_mismatch(expected, different)

    assert description.startswith("first differing cells:")
    assert "row 5, column 3" in description
    assert repr(expected[4][0]) not in description
    assert repr(expected[0][2]) not in description
    assert "$0.01 per GB" in description
    assert "$0.02 per GB" in description
    assert "0.03%" in description
    assert "$0.04 per GB" not in description
    assert description.count("expected ") == 3


def test_verbose_output() -> None:
    """Wire the verbose switch into expected-versus-actual trial logging."""
    prompt, expected_markdown = load_benchmark_assets()
    expected = normalize_markdown_table(expected_markdown)
    assert expected is not None

    class FakeClient:
        def chat(self, prompt: str, echo: str) -> str:
            assert echo == "none"
            return expected_markdown

    original_create_chat_client = benchmark.create_chat_client
    benchmark.create_chat_client = lambda model, config: FakeClient()
    output = io.StringIO()
    try:
        with redirect_stderr(output):
            result = run_single_trial(
                "gpt-4o-mini",
                0,
                BenchmarkConfig(models=["gpt-4o-mini"], verbose=True),
                prompt,
                expected,
            )
    finally:
        benchmark.create_chat_client = original_create_chat_client

    diagnostic = output.getvalue()
    assert result.correct
    assert "Expected:" in diagnostic
    assert "Actual:" in diagnostic
    assert format_normalized_table(expected) in diagnostic

    quiet_output = io.StringIO()
    benchmark.create_chat_client = lambda model, config: FakeClient()
    try:
        with redirect_stderr(quiet_output):
            quiet_result = run_single_trial(
                "gpt-4o-mini",
                0,
                BenchmarkConfig(models=["gpt-4o-mini"]),
                prompt,
                expected,
            )
    finally:
        benchmark.create_chat_client = original_create_chat_client

    assert quiet_result.correct
    assert quiet_output.getvalue() == ""
    assert parse_args(["--verbose"]).verbose
    assert not parse_args([]).verbose


def test_copilot_provider() -> None:
    """Run prefixed models through an isolated Copilot CLI invocation."""
    captured = {}

    def fake_run(command, capture_output, text, check):
        captured["command"] = command
        assert capture_output
        assert text
        assert not check
        return subprocess.CompletedProcess(command, 0, stdout="completed table\n", stderr="")

    original_run = benchmark.subprocess.run
    benchmark.subprocess.run = fake_run
    try:
        response = get_model_response(
            "copilot:gpt-5.4",
            BenchmarkConfig(models=["copilot:gpt-5.4"]),
            "benchmark prompt",
        )
    finally:
        benchmark.subprocess.run = original_run

    command = captured["command"]
    assert response == "completed table"
    assert command[:3] == ["copilot", "--model", "gpt-5.4"]
    assert "--available-tools=" in command
    assert "--disable-builtin-mcps" in command
    assert "--no-custom-instructions" in command
    assert command[-2:] == ["--prompt", "benchmark prompt"]


def test_copilot_provider_error() -> None:
    """Surface Copilot CLI failures instead of returning an empty response."""
    def fake_run(command, capture_output, text, check):
        return subprocess.CompletedProcess(
            command,
            1,
            stdout="",
            stderr="unknown model",
        )

    original_run = benchmark.subprocess.run
    benchmark.subprocess.run = fake_run
    try:
        try:
            get_model_response(
                "copilot:not-a-model",
                BenchmarkConfig(models=["copilot:not-a-model"]),
                "benchmark prompt",
            )
        except RuntimeError as error:
            assert "unknown model" in str(error)
        else:
            raise AssertionError("Expected Copilot CLI failure")
    finally:
        benchmark.subprocess.run = original_run


if __name__ == "__main__":
    test_assets()
    test_expected_response_values()
    test_markdown_normalization()
    test_incorrect_tables()
    test_numeric_equivalence_warnings()
    test_differing_cell_count_and_summary()
    test_mismatch_description()
    test_verbose_output()
    test_copilot_provider()
    test_copilot_provider_error()
    print("All tests passed!")
