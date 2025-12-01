# LLM Arithmetic Benchmark

An empirical study measuring how accurately LLMs perform arithmetic aggregation tasks without tool use.

## Overview

This benchmark tests whether LLMs can correctly compute `SUM(value) GROUP BY groupid` on CSV data using only their internal reasoning—no calculators, code execution, or external tools.

**Why this matters:** Many LLM applications assume models can handle basic arithmetic in-context. This benchmark quantifies where that assumption breaks down as data size increases.

### Example Task

Given this CSV:

```csv
txnid,groupid,value
1,a1,5
2,a1,15
3,a1,25
4,a1,35
5,a2,50
```

The model must output:

```
a1: 80
a2: 50
```

## Benchmark Parameters

| Parameter | Values |
|-----------|--------|
| Groups | 1, 3, 10, 25, 50 |
| Rows | 10, 50, 100, 500, 1000 |
| Trials per configuration | 10-100 (configurable) |

## Supported Models

### Cloud APIs
- **OpenAI**: `gpt-4o`, `gpt-4o-mini`, `gpt-4.1`, `gpt-4.1-mini`, `gpt-4.1-nano`
- **Anthropic**: `claude-sonnet-4-20250514`, `claude-haiku-4-20250514`, `claude-sonnet-4-5-20250929`, `claude-opus-4-5-20251101`

### Local Models (via Ollama)
Any model available through your Ollama instance, e.g.:
- `llama3.1:70b`
- `qwen2.5:32b`
- `mistral:7b`

## Installation

Requires Python 3.10+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/wesm/llm-arithmetic-benchmark.git
cd llm-arithmetic-benchmark
uv sync
```

## Usage

```bash
# Quick test (reduced grid for fast iteration)
uv run python benchmark.py --quick --models "gpt-4o-mini"

# Full benchmark on OpenAI models (requires OPENAI_API_KEY)
uv run python benchmark.py --openai-only --trials 10

# Full benchmark on Anthropic models (requires ANTHROPIC_API_KEY)
uv run python benchmark.py --anthropic-only --trials 10

# Local models via Ollama
uv run python benchmark.py --ollama-only --trials 10

# Specific models
uv run python benchmark.py --models "gpt-4o" "claude-sonnet-4-5-20250929" --trials 10

# Custom Ollama server
uv run python benchmark.py --ollama-url "http://localhost:11434" --models "llama3.1:70b"
```

## Configuration

Set these environment variables as needed:

| Variable | Purpose |
|----------|---------|
| `OPENAI_API_KEY` | Required for OpenAI models |
| `ANTHROPIC_API_KEY` | Required for Anthropic models |

## Output

Results are saved to `results/`:

| File | Contents |
|------|----------|
| `raw_results_<timestamp>.json` | Full trial data including model responses |
| `summary_<timestamp>.csv` | Aggregated accuracy by model/groups/rows |

### Sample Output

```
model                gpt-4o-mini
num_groups num_rows
1          10              <acc%>
           50              <acc%>
           100             <acc%>
3          10              <acc%>
           50              <acc%>
           100             <acc%>
10         10              <acc%>
           50              <acc%>
           100             <acc%>
```

## Running Tests

```bash
uv run python test_benchmark.py
```

## Contributing

Contributions welcome! Please open an issue to discuss significant changes before submitting a PR.

## License

[MIT](LICENSE)
