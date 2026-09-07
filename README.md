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

Headers, pricing-tier labels, row order, and row and column counts must match.
Numeric data cells are compared by value, so qualifiers such as `per GB` and
`per day`, currency/percentage markers, and trailing precision may be omitted
or formatted differently. Numerically equal but textually different cells pass
with a warning that shows the expected and actual text.

All USD values use exactly two decimal places, while savings percentages use exactly four significant figures. Both use strict round-half-up rounding, as specified in the prompt.

Wrong trials report an unparseable response or a row/cell count mismatch. When
the table shape matches, the output identifies the first three differing cells.

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

# Run models through GitHub Copilot CLI
uv run python benchmark.py \
  --models copilot:gpt-5.4 copilot:claude-sonnet-5 \
  --trials 10

# Resume an interrupted run
uv run python benchmark.py --models gpt-4o-mini --resume results/checkpoint_<timestamp>.json
```

Set `OPENAI_API_KEY` or `ANTHROPIC_API_KEY` when using the corresponding hosted models.
Models prefixed with `copilot:` are run through the installed and authenticated
GitHub Copilot CLI. Each trial uses a fresh non-interactive session with tools,
built-in MCP servers, repository instructions, memory, and remote export disabled.
The prefix is removed before passing the model name to `copilot --model`.

## Output

Results are written to `results/`:

| File | Contents |
|---|---|
| `checkpoint_<timestamp>.json` | Partial results saved after each completed trial |
| `raw_results_<timestamp>.json` | Full trial data, including raw and normalized responses |
| `summary_<timestamp>.csv` | Accuracy, average latency in seconds, average differing cells per trial (`avg_differing_cells`), and error totals by model |

## Sample results

On its failed run copilot-gpt-5.4 returned 31.96% instead of 32.00%, and 36.09% instead of 36.00%.

muse-glimmer was close on its failed runs. It frequently returned 33.9x% instead of 34.00%.

model|correct|total|accuracy|avg_latency_seconds|avg_differing_cells|errors
-|-:|-:|-:|-:|-:|-:
copilot:gpt-5.6-sol|25|25|100%|15.65|0.00|0
copilot:gpt-5.4|24|25|96%|14.48|0.12|0
copilot:claude-sonnet-5|23|25|92%|16.56|0.28|0
copilot:auto|21|25|84%|16.72|44|0
copilot:mai-code-1.1-flash|20|25|80%|28.68|0.28|0
ollama:muse-glimmer:30b-mlx|3|25|12%|321.03|0.96|0
ollama:gemma3:4b|0|25|0%|7.55|43.00|0

## Running Tests

The tests are offline and do not call model APIs.

```bash
uv run python test_benchmark.py
```

## License

[MIT](LICENSE)

## Related

* https://github.com/wesm/llm-arithmetic-benchmark - the repo this code was forked from
* https://github.com/maxim-saplin/llm_arithmetic
