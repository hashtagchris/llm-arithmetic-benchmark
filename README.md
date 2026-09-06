# LLM Arithmetic Benchmark

This benchmark measures whether language models can fill the missing cells in a Markdown pricing table using only internal reasoning.

The model receives [`prompt.md`](prompt.md), which contains the incomplete table and the required rounding behavior. The canonical completed table is stored separately in [`expected_response.md`](expected_response.md) and is never included in the model prompt.

## Scoring

Each model must return the full completed Markdown table. A response is correct only when its normalized rows and cells match the canonical expected response.

The comparison ignores presentation-only differences:

- whitespace around or within cells
- leading and trailing table pipes
- blank lines
- standard Markdown separator alignment
- a surrounding Markdown code fence

Headers, row order, row and column counts, source values, calculated values, punctuation, significant figures, and trailing zeros must match.

Added USD values use exactly two decimal places, while savings percentages use exactly four significant figures. Both use strict round-half-up rounding, as specified in the prompt.

## Installation

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/hashtagchris/llm-arithmetic-benchmark.git
cd llm-arithmetic-benchmark
uv sync
```

## Usage

```bash
# One trial against a specific model
uv run python benchmark.py --models gpt-4o-mini --trials 1

# Print normalized expected and actual tables for every trial
uv run python benchmark.py --models gpt-4o-mini --trials 1 --verbose

# Quick run uses one trial per selected model
uv run python benchmark.py --openai-only --quick

# Run configured provider groups
uv run python benchmark.py --openai-only --trials 10
uv run python benchmark.py --anthropic-only --trials 10
uv run python benchmark.py --ollama-only --trials 10

# Custom Ollama server
uv run python benchmark.py \
  --ollama-url http://localhost:11434 \
  --models ollama:llama3.1:70b

# Resume an interrupted run
uv run python benchmark.py --models gpt-4o-mini --resume results/checkpoint_<timestamp>.json
```

Set `OPENAI_API_KEY` or `ANTHROPIC_API_KEY` when using the corresponding hosted models.

## Output

Results are written to `results/`:

| File | Contents |
|---|---|
| `checkpoint_<timestamp>.json` | Partial results saved after each completed trial |
| `raw_results_<timestamp>.json` | Full trial data, including raw and normalized responses |
| `summary_<timestamp>.csv` | Accuracy, latency, and error totals by model |

## Running Tests

The tests are offline and do not call model APIs.

```bash
uv run python test_benchmark.py
```

## License

[MIT](LICENSE)
