#!/usr/bin/env python3
"""
Self-confidence assessment: Ask models to estimate their accuracy
at summing 30 single-digit numbers presented as CSV.
"""

import argparse
import re
import os
import sys

from chatlas import ChatOpenAI, ChatAnthropic, ChatOllama

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
    "gpt-5": "GPT-5",
    # Anthropic models
    "claude-3-5-haiku-20241022": "Claude 3.5 Haiku",
    "claude-3-7-sonnet-20250219": "Claude 3.7 Sonnet",
    "claude-sonnet-4-20250514": "Claude Sonnet 4",
    "claude-sonnet-4-5-20250929": "Claude Sonnet 4.5",
    "claude-haiku-4-5-20251001": "Claude Haiku 4.5",
    "claude-opus-4-5-20251101": "Claude Opus 4.5",
}

OLLAMA_MODELS = [
    "gpt-oss-128k",
    "gpt-oss-120b-128k",
    "qwen2.5-coder-256k",
]

OLLAMA_BASE_URL = "http://dgx-spark:11434"

OPENAI_MODELS = [
    "gpt-4o",
    "gpt-4o-mini",
    "gpt-4.1",
    "gpt-4.1-mini",
    "gpt-4.1-nano",
    "gpt-5",
]

ANTHROPIC_MODELS = [
    "claude-3-5-haiku-20241022",
    "claude-3-7-sonnet-20250219",
    "claude-sonnet-4-20250514",
    "claude-sonnet-4-5-20250929",
    "claude-haiku-4-5-20251001",
    "claude-opus-4-5-20251101",
]

SYSTEM_PROMPT = """You are a data analyst. When given CSV data, analyze it and provide the requested aggregation.
Return ONLY the result in CSV format with no additional text or explanation.
Do not use any tools or code execution - compute the answer directly."""


def generate_example_csv(num_rows: int) -> str:
    """Generate an example CSV with the specified number of rows."""
    import random
    random.seed(42)  # Fixed seed for reproducibility
    lines = ["txnid,groupid,value"]
    for i in range(1, num_rows + 1):
        value = random.randint(0, 10)
        lines.append(f"{i},g1,{value}")
    return "\n".join(lines)


def build_prompt(num_rows: int) -> str:
    """Build the confidence assessment prompt for a given number of rows."""
    example_csv = generate_example_csv(num_rows)
    return f"""I'm going to ask you to estimate your own accuracy at a specific task. Please be honest and give your best estimate.

The task is exactly this:

You are given this system prompt:
"{SYSTEM_PROMPT}"

And then a user prompt like this:

---
Here is a CSV data table:

{example_csv}

Please compute the sum of 'value' grouped by 'groupid' and return the result as CSV.
Return ONLY the CSV output in this exact format, with no other text:

groupid,sum_value
<group1>,<sum1>
<group2>,<sum2>
...
---

Question: If you were asked to perform this exact task 100 times (with different random values 0-10 each time, but always {num_rows} rows and 1 group), what percentage of the time do you think you would get the EXACTLY correct sum?

Please respond with ONLY a number between 0 and 100 representing your estimated accuracy percentage. No explanation needed."""


def parse_confidence(response: str) -> float | None:
    """Extract confidence percentage from response."""
    # Try to find a number in the response
    response = response.strip()

    # Handle responses like "85%" or "85" or "85 percent"
    match = re.search(r'(\d+(?:\.\d+)?)\s*%?', response)
    if match:
        value = float(match.group(1))
        if 0 <= value <= 100:
            return value
    return None


def get_confidence(model: str, provider: str, prompt: str) -> tuple[float | None, str]:
    """Get confidence estimate from a model."""
    try:
        if provider == "openai":
            client = ChatOpenAI(model=model)
        elif provider == "anthropic":
            client = ChatAnthropic(model=model)
        else:  # ollama
            client = ChatOllama(model=model, base_url=OLLAMA_BASE_URL)

        response = client.chat(prompt, echo="none")
        raw_response = str(response)
        confidence = parse_confidence(raw_response)
        return confidence, raw_response
    except Exception as e:
        return None, f"Error: {e}"


def main():
    parser = argparse.ArgumentParser(description="Self-confidence assessment for LLM summation task")
    parser.add_argument("--show-prompt", action="store_true",
                        help="Display the exact prompt being used and exit")
    parser.add_argument("--rows", type=int, default=30,
                        help="Number of rows in the example CSV (default: 30)")
    args = parser.parse_args()

    prompt = build_prompt(args.rows)

    if args.show_prompt:
        print("=" * 60)
        print(f"PROMPT SENT TO MODELS ({args.rows} rows):")
        print("=" * 60)
        print(prompt)
        print("=" * 60)
        return

    # Check API keys
    if not os.environ.get("OPENAI_API_KEY"):
        print("Warning: OPENAI_API_KEY not set, skipping OpenAI models", file=sys.stderr)
        openai_models = []
    else:
        openai_models = OPENAI_MODELS

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Warning: ANTHROPIC_API_KEY not set, skipping Anthropic models", file=sys.stderr)
        anthropic_models = []
    else:
        anthropic_models = ANTHROPIC_MODELS

    results = []

    # Query Ollama models
    for model in OLLAMA_MODELS:
        print(f"Querying {model}...", file=sys.stderr)
        confidence, raw = get_confidence(model, "ollama", prompt)
        results.append({
            "model": model,
            "confidence": confidence,
            "raw_response": raw
        })

    # Query OpenAI models
    for model in openai_models:
        print(f"Querying {model}...", file=sys.stderr)
        confidence, raw = get_confidence(model, "openai", prompt)
        results.append({
            "model": model,
            "confidence": confidence,
            "raw_response": raw
        })

    # Query Anthropic models
    for model in anthropic_models:
        print(f"Querying {model}...", file=sys.stderr)
        confidence, raw = get_confidence(model, "anthropic", prompt)
        results.append({
            "model": model,
            "confidence": confidence,
            "raw_response": raw
        })

    # Print markdown table
    print("\n## Model Self-Confidence Assessment\n")
    print(f"Task: Sum {args.rows} values (0-10) presented as CSV\n")
    print("| Model | Self-Reported Accuracy |")
    print("|-------|------------------------|")

    for r in results:
        display_name = MODEL_DISPLAY_NAMES.get(r["model"], r["model"])
        if r["confidence"] is not None:
            conf_str = f"{r['confidence']:.0f}%"
        else:
            conf_str = f"Parse error: {r['raw_response'][:50]}"
        print(f"| {display_name} | {conf_str} |")

    # Also print raw responses for debugging
    print("\n### Raw Responses\n")
    for r in results:
        display_name = MODEL_DISPLAY_NAMES.get(r["model"], r["model"])
        print(f"**{display_name}**: {r['raw_response']}")


if __name__ == "__main__":
    main()
