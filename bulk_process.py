"""
Bulk pipeline: upload a CSV (query_id, mode_url, raw_sql — same format as
fetched_queries_fixed.csv), get back everything at once per query:
  - draft migrated SQL
  - complexity bucket
  - warnings
  - an auto-generated reconciliation query (best-effort — see caveats)

This does NOT replace manual review. It removes the mechanical typing labor
(running migrate() + writing key_cols/metric_cols + calling generate_recon
by hand for each query) so you spend your time reading output, not producing
boilerplate.

CAVEATS, read before trusting this on anything with real consequences:
- The auto-detected key/metric columns are a HEURISTIC (see auto_recon.py's
  docstring for exact failure modes). Always spot-check the generated recon
  query's key_cols/metric_cols line before running it — don't blind-trust it
  the way you must never blind-trust the draft SQL either.
- If a query's outer SELECT is `SELECT *` or otherwise not introspectable,
  recon generation is skipped for that row — you'll see an explicit note in
  the output, not a silently wrong recon query.
- This can only build a recon query if BOTH old and new versions share
  compatible structure (which they should, given migrate() only renames
  tables/columns and injects filters — it doesn't restructure the query).

Usage:
    python bulk_process.py fetched_queries_fixed.csv bulk_output.csv
"""

import csv
import sys

from migrate_query import migrate, classify_complexity
from auto_recon import infer_key_metric_columns
from generate_recon import generate_recon_sql

try:
    from auto_recon_llm import infer_key_metric_columns_llm
except ImportError:
    infer_key_metric_columns_llm = None


def process_row(query_id: str, raw_sql: str, use_llm_fallback: bool = False) -> dict:
    complexity = classify_complexity(raw_sql)
    draft_sql, warnings = migrate(raw_sql)

    keys, metrics = infer_key_metric_columns(raw_sql)
    used_llm_fallback = False

    if not keys and not metrics and use_llm_fallback and infer_key_metric_columns_llm is not None:
        try:
            keys, metrics = infer_key_metric_columns_llm(raw_sql)
            used_llm_fallback = bool(keys or metrics)
        except Exception:
            pass  # LLM fallback failed — fall through to the SKIPPED path below

    if not keys and not metrics:
        recon_sql = "-- SKIPPED: could not auto-detect key/metric columns " \
                    "(likely SELECT * or unparseable outer SELECT), even with " \
                    "LLM fallback. Build this recon query manually with generate_recon.py."
    elif not metrics:
        recon_sql = f"-- WARNING: no metric (aggregate) columns detected — " \
                    f"only keys found: {keys}. This query may not have " \
                    f"aggregated output, or the heuristic missed something. " \
                    f"Review before trusting."
    else:
        try:
            recon_sql = generate_recon_sql(raw_sql, draft_sql, keys, metrics)
        except Exception as e:
            recon_sql = f"-- SKIPPED: recon generation failed ({e}). Build manually."

    return {
        "query_id": query_id,
        "complexity": complexity,
        "draft_migrated_sql": draft_sql,
        "warnings": " | ".join(warnings) if warnings else "(none)",
        "auto_detected_keys": ", ".join(keys),
        "auto_detected_metrics": ", ".join(metrics),
        "used_llm_fallback": used_llm_fallback,
        "recon_query": recon_sql,
    }


def run(input_csv: str, output_csv: str, use_llm_fallback: bool = False):
    with open(input_csv, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    results = [process_row(row["query_id"], row["raw_sql"], use_llm_fallback) for row in rows]

    fieldnames = ["query_id", "complexity", "draft_migrated_sql", "warnings",
                  "auto_detected_keys", "auto_detected_metrics", "used_llm_fallback", "recon_query"]
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(results)

    skipped = sum(1 for r in results if r["recon_query"].startswith("-- SKIPPED"))
    llm_used = sum(1 for r in results if r["used_llm_fallback"])
    print(f"Processed {len(results)} queries -> {output_csv}")
    print(f"  Recon auto-generated for {len(results) - skipped}")
    print(f"  (of which, LLM fallback used for {llm_used} — REVIEW these keys/metrics before trusting)")
    print(f"  Recon SKIPPED (needs manual generate_recon.py) for {skipped}")


if __name__ == "__main__":
    if len(sys.argv) not in (3, 4):
        print("Usage: python bulk_process.py fetched_queries_fixed.csv bulk_output.csv [--llm-fallback]")
        sys.exit(1)
    use_llm = len(sys.argv) == 4 and sys.argv[3] == "--llm-fallback"
    run(sys.argv[1], sys.argv[2], use_llm)