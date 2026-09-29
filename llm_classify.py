"""
LLM fallback for columns generate_key_metric_cols.py couldn't confidently
classify (window functions without a matching GROUP BY, no GROUP BY at all,
unresolvable wildcards). Only touches the FLAGGED list -- never overrides
anything the heuristic already classified confidently as key/metric.

Uses DeepSeek's API (OpenAI-compatible chat completions).

Requires: pip install requests --break-system-packages
Requires: export DEEPSEEK_API_KEY=... (never hardcode the key in this file)

Usage:
    python llm_classify.py path/to/query.sql
    # runs the heuristic first, then asks the LLM to resolve only what's flagged,
    # then writes the final key_cols.txt / metric_cols.txt

Or as a library:
    from llm_classify import classify_with_llm_fallback
    keys, metrics, still_unsure = classify_with_llm_fallback(sql)

** NOT TESTED AGAINST THE LIVE API ** -- api.deepseek.com isn't reachable from
the sandbox this was built in. The JSON parsing/merging logic below is tested
against a mocked response (see test_with_mock_response()). Run once against
the real API and sanity-check the output before trusting it on a full batch.
"""

import os
import json
import sys

from generate_key_metric_cols import classify_columns

DEEPSEEK_API_URL = "https://api.deepseek.com/chat/completions"
DEEPSEEK_MODEL = "deepseek-chat"


def _strip_reason(flagged_entry: str) -> str:
    """Flagged entries look like 'cumulative_ltv (window aggregate — ...)' --
    strip the parenthetical reason to get back the bare column name."""
    return flagged_entry.split(" (", 1)[0].strip()


def _build_prompt(sql: str, flagged_names: list[str], known_keys: list[str], known_metrics: list[str]) -> str:
    return f"""You are helping classify SQL output columns for a data reconciliation tool.
The tool compares an OLD query's output to a NEW (migrated) query's output, row by row,
matched on "key" columns (dimensions used to join old vs new), while "metric" columns
are the numeric values being compared for differences (deltas).

Here is the full query:
```sql
{sql}
```

These columns were ALREADY confidently classified (do not reconsider them):
- KEYS: {known_keys}
- METRICS: {known_metrics}

Classify ONLY these remaining columns as either "key" or "metric":
{flagged_names}

Rules of thumb:
- A "key" is a dimension/category a row can be grouped or joined on (dates, IDs, labels, deciles, flags used for segmentation).
- A "metric" is a numeric value being measured/aggregated (sums, counts, ratios, running totals, joined aggregate values).
- If a column is a running total / cumulative sum (e.g. via a window function), it is almost always a "metric".
- If you genuinely cannot tell, use "unsure" rather than guessing.

Respond with ONLY a JSON object, no markdown fences, no explanation, in this exact form:
{{"column_name": "key", "another_column": "metric", "unclear_one": "unsure"}}
"""


def _call_deepseek(prompt: str, api_key: str) -> str:
    import requests
    resp = requests.post(
        DEEPSEEK_API_URL,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={"model": DEEPSEEK_MODEL, "messages": [{"role": "user", "content": prompt}], "temperature": 0},
        timeout=60,
    )
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"]


def _parse_llm_json(raw_text: str) -> dict:
    text = raw_text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    return json.loads(text.strip())


def classify_with_llm_fallback(sql: str, api_key: str | None = None) -> tuple[list[str], list[str], list[str]]:
    keys, metrics, flagged = classify_columns(sql)

    if not flagged:
        return keys, metrics, []

    flagged_names = [_strip_reason(f) for f in flagged]

    api_key = api_key or os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        print("No DEEPSEEK_API_KEY found in environment -- skipping LLM fallback, "
              "returning heuristic-only result with these still flagged:", flagged_names)
        return keys, metrics, flagged_names

    prompt = _build_prompt(sql, flagged_names, keys, metrics)

    try:
        raw = _call_deepseek(prompt, api_key)
        classification = _parse_llm_json(raw)
    except Exception as e:
        print(f"LLM call or parsing failed ({e}) -- returning heuristic-only result, "
              f"these remain flagged for manual review: {flagged_names}")
        return keys, metrics, flagged_names

    still_unsure = []
    for name in flagged_names:
        verdict = classification.get(name, "unsure")
        if verdict == "key":
            keys.append(name)
        elif verdict == "metric":
            metrics.append(name)
        else:
            still_unsure.append(name)

    return keys, metrics, still_unsure


def test_with_mock_response():
    mock_flagged_names = ["approved", "total_trx", "total_amt", "cumulative_ltv", "mob_check"]
    mock_response = '''```json
{"approved": "metric", "total_trx": "metric", "total_amt": "metric", "cumulative_ltv": "metric", "mob_check": "unsure"}
```'''
    parsed = _parse_llm_json(mock_response)
    assert parsed["approved"] == "metric"
    assert parsed["mob_check"] == "unsure"
    print("Mock parsing test passed:", parsed)

    keys = ["crd", "mob"]
    metrics = ["total_ltv"]
    still_unsure = []
    for name in mock_flagged_names:
        verdict = parsed.get(name, "unsure")
        if verdict == "key":
            keys.append(name)
        elif verdict == "metric":
            metrics.append(name)
        else:
            still_unsure.append(name)
    print("Final keys:", keys)
    print("Final metrics:", metrics)
    print("Still unsure:", still_unsure)
    assert still_unsure == ["mob_check"]
    print("Merge logic test passed.")


if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "--test-mock":
        test_with_mock_response()
        sys.exit(0)

    if len(sys.argv) < 2:
        print("Usage: python llm_classify.py path/to/query.sql")
        print("       python llm_classify.py --test-mock   (runs the offline parsing test)")
        sys.exit(1)

    with open(sys.argv[1]) as f:
        sql = f.read()

    keys, metrics, still_unsure = classify_with_llm_fallback(sql)

    with open("key_cols.txt", "w") as f:
        f.write("\n".join(keys))
    with open("metric_cols.txt", "w") as f:
        f.write("\n".join(metrics))

    print(f"KEY COLS ({len(keys)}): {keys}")
    print(f"METRIC COLS ({len(metrics)}): {metrics}")
    if still_unsure:
        print(f"\nSTILL UNSURE after LLM pass ({len(still_unsure)}) -- genuinely needs manual decision:")
        for u in still_unsure:
            print(f"  - {u}")
