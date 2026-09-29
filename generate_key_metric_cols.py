"""
Auto-generates key_cols.txt and metric_cols.txt from a query's outer SELECT —
replacing the manual "copy query into ChatGPT and ask" step.

HEURISTIC (mirrors BigQuery's own GROUP BY ALL semantics):
- GROUP BY ALL: any output column whose expression contains an aggregate
  function (SUM, COUNT, AVG, etc.) is a METRIC. Everything else is a KEY.
- Explicit GROUP BY 1,2,3 or GROUP BY col_a, col_b: those referenced output
  columns are KEYS, everything else with an aggregate is a METRIC.
- alias.* wildcards (very common in this project's LTV-style queries, e.g.
  "SELECT t1.*, total.approved, SUM(x) OVER (...) AS cumulative_ltv FROM (...) t1")
  are recursively expanded: the referenced derived table's own SELECT list is
  found and classified the same way, then merged into the outer result.
- Window functions (OVER) with no GROUP BY at the same level are flagged for
  manual review, not guessed — these are usually metrics (running totals,
  cumulative sums) but the semantics vary enough that auto-classifying them
  confidently isn't safe.
- No GROUP BY at all anywhere in scope -> flagged for manual review.

This is a heuristic tool, same philosophy as migrate_query.py: it removes the
mechanical work, it does not replace judgment. Always skim FLAGGED output
before trusting the key/metric split, especially on window-function columns.

Usage:
    python generate_key_metric_cols.py path/to/query.sql
    # writes key_cols.txt and metric_cols.txt next to it

Or as a library:
    from generate_key_metric_cols import classify_columns
    keys, metrics, flagged = classify_columns(sql)
"""

import sys
import os
import sqlglot
from sqlglot import exp

DIALECT = "bigquery"

_AGG_FUNC_NAMES = {
    "SUM", "COUNT", "AVG", "MIN", "MAX", "ARRAYAGG", "STRINGAGG",
    "APPROXQUANTILES", "APPROXCOUNTDISTINCT", "ANYVALUE", "LOGICALAND",
    "LOGICALOR", "BITAND", "BITOR", "BITXOR", "CORR", "COVARPOP", "COVARSAMP",
    "STDDEV", "STDDEVPOP", "STDDEVSAMP", "VARIANCE", "VARPOP", "VARSAMP",
}


def _contains_aggregate(expr: exp.Expression) -> bool:
    if any(True for _ in expr.find_all(exp.AggFunc)):
        return True
    for func in expr.find_all(exp.Func):
        name = (func.sql_name() or "").upper().replace("_", "")
        if name in _AGG_FUNC_NAMES:
            return True
    return False


def _contains_window(expr: exp.Expression) -> bool:
    return any(True for _ in expr.find_all(exp.Window))


def _group_by_info(select: exp.Select) -> tuple[bool, set[str], set[int]]:
    """Returns (is_group_by_all, referenced_col_names, referenced_ordinals)."""
    group = select.args.get("group")
    if not group:
        return False, set(), set()
    # GROUP BY ALL is stored as group.args["all"] = True with an EMPTY
    # expressions list -- confirmed via direct sqlglot inspection 2026-08-13.
    # It is NOT represented as a literal/column inside expressions.
    if group.args.get("all"):
        return True, set(), set()
    names, ordinals = set(), set()
    for e in group.expressions:
        if isinstance(e, exp.Column):
            names.add(e.name)
        elif isinstance(e, exp.Literal) and e.is_int:
            ordinals.add(int(e.this))
    return False, names, ordinals


def _find_derived_table(select: exp.Select, alias: str) -> exp.Select | None:
    """Find the Select node for a FROM/JOIN-ed derived table (subquery) by its alias."""
    for subq in select.find_all(exp.Subquery):
        if subq.alias_or_name == alias:
            inner = subq.this
            if isinstance(inner, exp.Select):
                return inner
    return None


def classify_columns(sql: str) -> tuple[list[str], list[str], list[str]]:
    """Returns (key_cols, metric_cols, flagged_for_review), for the query's
    outermost SELECT, recursively expanding any alias.* wildcards found."""
    tree = sqlglot.parse_one(sql, read=DIALECT)
    outer_select = tree if isinstance(tree, exp.Select) else tree.find(exp.Select)
    if outer_select is None:
        raise ValueError("Could not find a SELECT statement to classify")

    is_group_all, group_names, group_ordinals = _group_by_info(outer_select)
    has_group_by = bool(outer_select.args.get("group"))

    keys: list[str] = []
    metrics: list[str] = []
    flagged: list[str] = []
    seen: set[str] = set()

    def add(name: str, bucket: list[str]):
        if name not in seen:
            seen.add(name)
            bucket.append(name)

    for i, projection in enumerate(outer_select.expressions):
        expr = projection.this if isinstance(projection, exp.Alias) else projection

        # --- wildcard expansion: alias.* ---
        if isinstance(expr, exp.Column) and isinstance(expr.this, exp.Star):
            qualifier = expr.table
            inner_select = _find_derived_table(outer_select, qualifier) if qualifier else None
            if inner_select is None:
                flagged.append(f"{qualifier}.* (could not find the referenced subquery to expand — expand manually)")
                continue
            inner_keys, inner_metrics, inner_flagged = classify_columns(inner_select.sql(dialect=DIALECT))
            for k in inner_keys:
                add(k, keys)
            for m in inner_metrics:
                add(m, metrics)
            for f in inner_flagged:
                add(f, flagged)
            continue

        alias = projection.alias_or_name
        has_agg = _contains_aggregate(expr)
        has_window = _contains_window(expr)

        if has_window:
            if has_agg:
                # e.g. SUM(x) OVER (...) — a window aggregate, almost always a metric
                # (running total / cumulative sum), but flagged rather than silently
                # bucketed since window semantics vary — confirm before trusting.
                flagged.append(f"{alias} (window aggregate — likely a metric, e.g. running total, but confirm)")
            else:
                flagged.append(f"{alias} (window function, no aggregate — confirm manually)")
            continue

        if is_group_all:
            add(alias, metrics if has_agg else keys)
        elif has_group_by:
            is_key = (i + 1) in group_ordinals or alias in group_names
            if is_key:
                add(alias, keys)
            elif has_agg:
                add(alias, metrics)
            else:
                flagged.append(f"{alias} (has an aggregate-free expression but isn't in GROUP BY — confirm)")
        else:
            flagged.append(f"{alias} (no GROUP BY found at this level — confirm manually)")

    return keys, metrics, flagged


def write_cols_files(sql: str, out_dir: str = "."):
    keys, metrics, flagged = classify_columns(sql)

    with open(f"{out_dir}/key_cols.txt", "w") as f:
        f.write("\n".join(keys))
    with open(f"{out_dir}/metric_cols.txt", "w") as f:
        f.write("\n".join(metrics))

    print(f"KEY COLS ({len(keys)}): {keys}")
    print(f"METRIC COLS ({len(metrics)}): {metrics}")
    if flagged:
        print(f"\n⚠️  FLAGGED FOR MANUAL REVIEW ({len(flagged)}) — NOT written to either file:")
        for f_ in flagged:
            print(f"  - {f_}")
        print("Decide manually and add these to the right .txt file before running generate_recon.py.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python generate_key_metric_cols.py path/to/query.sql")
        sys.exit(1)

    input_path = sys.argv[1]
    # write output files next to the input SQL file (e.g. scratch/), not the
    # current working directory -- matches the existing workflow where
    # old_query.sql, new_query.sql, key_cols.txt, metric_cols.txt all live
    # together in the same folder.
    out_dir = os.path.dirname(input_path) or "."

    with open(input_path) as f:
        sql = f.read()

    write_cols_files(sql, out_dir=out_dir)
    print(f"\nWrote {out_dir}/key_cols.txt and {out_dir}/metric_cols.txt")