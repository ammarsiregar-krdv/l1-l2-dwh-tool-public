"""
Injects a secondary sort key into NTILE() window functions that only order by
one column (e.g. NTILE(10) OVER (PARTITION BY x ORDER BY b_score ASC)), making
decile assignment deterministic instead of leaving ties to BigQuery's
undefined tie-break order.

Confirmed 2026-07-29: this measurably reduces (but does NOT fully eliminate)
discrepancies caused by score rounding (FLOAT64 -> NUMERIC) creating exact
ties. Verified on a bnpl-engine query: brought bucket-level metrics to exact
match, reduced but did not fully close a small residual LTV drift. Did NOT
fix the cicilan case, which has far more ties and needs a different answer
(likely a business decision, not a query fix) — see the b010ee1f39f5 thread.

Because of that mixed result, this is deliberately NOT applied automatically
by generate_recon.py. Use it explicitly, and always re-verify results —
don't assume it fixes every query the same way it fixed this one.

Usage:
    from apply_ntile_tiebreak import apply_ntile_tiebreak
    fixed_sql, count = apply_ntile_tiebreak(new_sql, order_col="b_score", secondary_col="user_id")
"""

import sqlglot
from sqlglot import exp


def apply_ntile_tiebreak(sql: str, order_col: str = "b_score", secondary_col: str = "user_id",
                          dialect: str = "bigquery") -> tuple[str, int]:
    """
    Returns (modified_sql, count) where count is how many NTILE windows were
    changed. If count is 0, either there's no matching NTILE pattern, or the
    secondary column was already present (idempotent — safe to run twice).
    """
    tree = sqlglot.parse_one(sql, read=dialect)
    count = 0

    for window in tree.find_all(exp.Window):
        func = window.this
        func_name = func.sql_name() if hasattr(func, "sql_name") else type(func).__name__
        if func_name.upper() != "NTILE":
            continue

        order = window.args.get("order")
        if order is None:
            continue

        order_exprs = order.expressions
        # only touch windows that order by the target column specifically —
        # don't blindly add a tiebreak to every NTILE in the query
        orders_by_target = any(
            expr.this.name.lower() == order_col.lower()
            for expr in order_exprs if isinstance(expr, exp.Ordered) and isinstance(expr.this, exp.Column)
        )
        if not orders_by_target:
            continue

        already_present = any(
            isinstance(expr.this, exp.Column) and expr.this.name.lower() == secondary_col.lower()
            for expr in order_exprs
        )
        if already_present:
            continue

        order_exprs.append(exp.Ordered(this=exp.column(secondary_col), desc=False))
        count += 1

    return tree.sql(dialect=dialect, pretty=True), count
