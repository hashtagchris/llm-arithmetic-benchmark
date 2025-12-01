#!/usr/bin/env python3
"""
Test the benchmark infrastructure without calling LLMs.
"""

from benchmark import (
    generate_data,
    parse_response,
    check_correct,
    SYSTEM_PROMPT,
    USER_PROMPT_TEMPLATE,
)


def test_data_generation():
    """Test that data generation produces valid CSV and correct expected sums."""
    print("Testing data generation...")

    for num_groups in [1, 3, 10]:
        for num_rows in [10, 50, 100]:
            csv_data, expected = generate_data(num_groups, num_rows, seed=42)

            # Verify CSV has correct structure
            lines = csv_data.strip().split("\n")
            assert lines[0] == "txnid,groupid,value", f"Bad header: {lines[0]}"
            assert len(lines) == num_rows + 1, f"Expected {num_rows + 1} lines, got {len(lines)}"

            # Verify expected has right number of groups (might be fewer if some groups have no rows)
            assert len(expected) <= num_groups, f"Too many groups: {len(expected)}"
            assert len(expected) >= 1, "No groups found"

            # Verify sums are positive integers
            for group, total in expected.items():
                assert isinstance(total, (int, float)), f"Bad sum type: {type(total)}"
                assert total > 0, f"Non-positive sum: {total}"

            print(f"  {num_groups} groups, {num_rows} rows: OK")

    print("Data generation tests passed!\n")


def test_response_parsing():
    """Test parsing of various response formats."""
    print("Testing response parsing...")

    # Clean CSV
    response1 = """groupid,sum_value
g1,150
g2,200
g3,100"""
    result1 = parse_response(response1)
    assert result1 == {"g1": 150, "g2": 200, "g3": 100}, f"Failed clean CSV: {result1}"
    print("  Clean CSV: OK")

    # With markdown code block
    response2 = """```csv
groupid,sum_value
g1,150
g2,200
```"""
    result2 = parse_response(response2)
    assert result2 == {"g1": 150, "g2": 200}, f"Failed markdown CSV: {result2}"
    print("  Markdown code block: OK")

    # With extra whitespace
    response3 = """
groupid,sum_value
  g1 , 150
g2,200

"""
    result3 = parse_response(response3)
    assert result3 == {"g1": 150, "g2": 200}, f"Failed whitespace CSV: {result3}"
    print("  Extra whitespace: OK")

    # With decimal values (should round)
    response4 = """groupid,sum_value
g1,150.0
g2,200.5"""
    result4 = parse_response(response4)
    assert result4 == {"g1": 150, "g2": 200}, f"Failed decimal CSV: {result4}"
    print("  Decimal values: OK")

    # With quotes
    response5 = """groupid,sum_value
"g1","150"
"g2","200"
"""
    result5 = parse_response(response5)
    assert result5 == {"g1": 150, "g2": 200}, f"Failed quoted CSV: {result5}"
    print("  Quoted values: OK")

    # Various header formats
    response6 = """groupid,sum(value)
g1,150
g2,200"""
    result6 = parse_response(response6)
    assert result6 == {"g1": 150, "g2": 200}, f"Failed alt header: {result6}"
    print("  Alternative header: OK")

    print("Response parsing tests passed!\n")


def test_correctness_check():
    """Test the correctness checking logic."""
    print("Testing correctness check...")

    expected = {"g1": 100, "g2": 200, "g3": 300}

    # Exact match
    assert check_correct(expected, {"g1": 100, "g2": 200, "g3": 300})
    print("  Exact match: OK")

    # Different order (should pass)
    assert check_correct(expected, {"g3": 300, "g1": 100, "g2": 200})
    print("  Different order: OK")

    # Wrong value
    assert not check_correct(expected, {"g1": 100, "g2": 200, "g3": 301})
    print("  Wrong value detection: OK")

    # Missing group
    assert not check_correct(expected, {"g1": 100, "g2": 200})
    print("  Missing group detection: OK")

    # Extra group
    assert not check_correct(expected, {"g1": 100, "g2": 200, "g3": 300, "g4": 400})
    print("  Extra group detection: OK")

    # None response
    assert not check_correct(expected, None)
    print("  None response detection: OK")

    print("Correctness check tests passed!\n")


def test_prompt_generation():
    """Test that prompts are generated correctly."""
    print("Testing prompt generation...")

    csv_data, expected = generate_data(3, 10, seed=42)
    prompt = USER_PROMPT_TEMPLATE.format(csv_data=csv_data)

    assert "txnid,groupid,value" in prompt
    assert "sum of 'value'" in prompt
    assert "grouped by 'groupid'" in prompt
    print("  Prompt structure: OK")

    print(f"\nExample prompt:\n{'-' * 40}")
    print(SYSTEM_PROMPT)
    print(f"\n{prompt[:500]}...")
    print(f"\nExpected result: {expected}")
    print("Prompt generation tests passed!\n")


def test_deterministic_seed():
    """Test that seed produces deterministic results."""
    print("Testing deterministic seeding...")

    csv1, expected1 = generate_data(5, 50, seed=12345)
    csv2, expected2 = generate_data(5, 50, seed=12345)

    assert csv1 == csv2, "CSV data not deterministic"
    assert expected1 == expected2, "Expected sums not deterministic"
    print("  Same seed produces same data: OK")

    csv3, expected3 = generate_data(5, 50, seed=54321)
    assert csv1 != csv3, "Different seeds should produce different data"
    print("  Different seeds produce different data: OK")

    print("Deterministic seeding tests passed!\n")


if __name__ == "__main__":
    test_data_generation()
    test_response_parsing()
    test_correctness_check()
    test_prompt_generation()
    test_deterministic_seed()
    print("=" * 50)
    print("All tests passed!")
    print("=" * 50)
