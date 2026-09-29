"""
LLM-assisted fallback for detecting key vs metric columns, used ONLY when
the deterministic heuristic in auto_recon.py returns nothing (typically
SELECT * queries, or ones where no aggregate function was found).

WHY A FALLBACK, NOT A REPLACEMENT: the deterministic heuristic is reliable
when it works — it's based on a real signal (aggregate function presence),
not a guess. An LLM call is strictly less certain than that. Use the cheap,
reliable method first; only reach for the expensive, uncertain method when
the reliable one has nothing to offer.

LLM-suggested columns are LESS trustworthy than heuristic-detected ones —
always confirm the suggested split makes sense before generating/running
the recon query. This is explicitly flagged in the output, not silently
treated the same as a heuristic result.

Uses DeepSeek's API (same as llm_refine.py).

Setup:
    pip install openai
    export DEEPSEEK_API_KEY="your_key_here"
"""

import json
import os

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

DEEPSEEK_BASE_URL = "https://api.deepseek.com"
MODEL = "deepseek-chat"

SYSTEM_PROMPT = """You are helping classify SQL output columns for a data reconciliation tool.

Given a SQL query, identify its OUTER (final) SELECT columns and classify each as:
- "key": a dimension/grouping column (dates, IDs, categories, names) — values
  used to match rows between an old and new version of this query.
- "metric": a measure/aggregate column (counts, sums, amounts, rates) —
  values you'd compare for drift between old and new.

If the query uses SELECT * or you cannot determine actual output column names
from the SQL text alone, return empty lists for both — do not guess column
names that don't appear in the text.

Return ONLY valid JSON in this exact shape, nothing else:
{"keys": ["col1", "col2"], "metrics": ["col3", "col4"]}"""


def infer_key_metric_columns_llm(sql: str, api_key: str | None = None) -> tuple[list[str], list[str]]:
    """
    Returns (keys, metrics). Empty lists mean the LLM also couldn't determine
    columns (e.g. genuine SELECT * with no visible schema) — at that point,
    you need the actual result schema (e.g. from a LIMIT 0 dry run in
    BigQuery), not more guessing from SQL text alone.
    """
    if OpenAI is None:
        raise RuntimeError("pip install openai first")

    key = api_key or os.environ.get("DEEPSEEK_API_KEY")
    if not key:
        raise RuntimeError("Set DEEPSEEK_API_KEY env var or pass api_key=")

    client = OpenAI(api_key=key, base_url=DEEPSEEK_BASE_URL)

    response = client.chat.completions.create(
        model=MODEL,
        max_tokens=1000,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"```sql\n{sql}\n```"},
        ],
    )

    raw = response.choices[0].message.content.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1] if "\n" in raw else raw
        if raw.endswith("```"):
            raw = raw.rsplit("```", 1)[0]
        raw = raw.strip()

    try:
        parsed = json.loads(raw)
        return parsed.get("keys", []), parsed.get("metrics", [])
    except json.JSONDecodeError:
        return [], []  # fail safe — treat unparseable LLM output as "couldn't determine"
