"""
Infers key vs metric columns from a query's outer SELECT list, so
generate_recon.py can run in bulk without you manually typing key_cols.txt /
metric_cols.txt for every single query.

Heuristic: a column is a METRIC if its expression contains an aggregate
function (SUM, COUNT, AVG, MIN, MAX, ARRAY_AGG, STRING_AGG, etc.). Otherwise
it's treated as a KEY (dimension).

THIS IS A HEURISTIC, NOT A GUARANTEE. Known failure modes:
- A window function (ROW_NUMBER, RANK) isn't an aggregate but also isn't a
  real dimension — gets misclassified as a key. Usually harmless (just an
  extra join key that always matches), but check the output.
- A raw numeric column with no aggregate wrapper (e.g. a pre-computed rate
  passed through unchanged) gets classified as a key, not a metric — you
  won't get a delta for it. If a column you care about isn't showing up
  under metrics, this is why — add it manually.
- SELECT * makes this impossible to introspect — falls back to empty lists,
  you MUST supply columns manually for those queries.

Usage:
    from auto_recon import infer_key_metric_columns
    keys, metrics = infer_key_metric_columns(old_sql)
"""

import sqlglot
from sqlglot import exp

AGGREGATE_FUNCS = {
    "SUM", "COUNT", "AVG", "MIN", "MAX", "ARRAY_AGG", "STRING_AGG",
    "COUNTIF", "ANY_VALUE", "STDDEV", "VARIANCE", "APPROX_COUNT_DISTINCT",
}


def _contains_aggregate(expression: exp.Expression) -> bool:
    for func in expression.find_all(exp.Func):
        name = func.sql_name().upper() if hasattr(func, "sql_name") else type(func).__name__.upper()
        if name in AGGREGATE_FUNCS:
            return True
    for cls_name in ("Sum", "Count", "Avg", "Min", "Max", "ArrayAgg", "GroupConcat"):
        if expression.find(getattr(exp, cls_name, type(None))):
            return True
    return False


def infer_key_metric_columns(sql: str, dialect: str = "bigquery") -> tuple[list[str], list[str]]:
    try:
        tree = sqlglot.parse_one(sql, read=dialect)
    except Exception:
        return [], []

    select = tree if isinstance(tree, exp.Select) else tree.find(exp.Select)
    if select is None:
        return [], []

    keys, metrics = [], []
    has_wildcard = False
    for projection in select.expressions:
        alias = projection.alias_or_name
        if alias == "*" or not alias:
            has_wildcard = True  # confirmed bug 2026-07-28: a MIXED select
            # (table.* plus explicit columns) used to silently drop the
            # wildcard-covered columns and proceed with only the explicit
            # ones as keys — producing a dangerously coarse join key that
            # caused row fan-out and meaningless deltas. Now we bail out
            # entirely instead of returning a partial, misleading result.
            continue
        if _contains_aggregate(projection):
            metrics.append(alias)
        else:
            keys.append(alias)

    if has_wildcard:
        return [], []  # force caller to treat this as "could not determine" — safe, not silent-wrong

    return keys, metrics


if __name__ == "__main__":
    sample = """
    SELECT
      crd_month,
      score_decile,
      acquisition,
      count(distinct user_id) as dpd91_users,
      sum(loan_amount) as dpd91_amount,
      min(user_id) as sample_1
    FROM cte_total
    GROUP BY 1,2,3
    """
    keys, metrics = infer_key_metric_columns(sample)
    print("keys:", keys)
    print("metrics:", metrics)