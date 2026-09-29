"""
Generates the reconciliation query scaffolding (old_logic/new_logic CTEs,
FULL OUTER JOIN on key columns, delta calculations) — the same boilerplate
you wrote by hand for query 0de8fc4b38e5, now reusable for any query pair.

You still write the actual old/new SELECT bodies yourself (that's the part
that requires real judgment) — this only automates the comparison wrapper.

Usage:
    python generate_recon.py old_query.sql new_query.sql key_cols.txt metric_cols.txt [output.sql] [--fix-ntile-tiebreak]

Where key_cols.txt / metric_cols.txt are one column name per line, e.g.:
    key_cols.txt:
        crd_month
        score_decile
        mob_dpd91
        acquisition
    metric_cols.txt:
        dpd91_users
        dpd91_amount
        approved_users

--fix-ntile-tiebreak: applies the confirmed NTILE(...) ORDER BY b_score ASC ->
add user_id as secondary sort fix to new_sql ONLY (never old_sql, which is
your ground truth and should stay untouched). See apply_ntile_tiebreak.py's
docstring for what this does and does not fix — it's a real, verified
improvement for some queries (bnpl), not a universal solution (cicilan still
needs a separate answer).

FIXED 2026-07-29: the join condition now wraps every key column in
CAST(... AS STRING) + COALESCE(..., '__NULL_KEY__') instead of plain equality.
Confirmed necessary by query 552079fe492f — a key column (approved_apptype_details)
was intentionally NULL for an entire category of rows (ape_flow = '1. old_ape'),
and SQL's `NULL = NULL` is never true, so those rows silently never matched even
though every other column was identical, producing a huge fake "discrepancy".
The CAST to STRING (rather than a type-specific sentinel) avoids type-mismatch
errors across key columns of different types (DATE, INT64, BOOL, STRING, etc.)
in the same query.
"""

import sys

try:
    from apply_ntile_tiebreak import apply_ntile_tiebreak
except ImportError:
    apply_ntile_tiebreak = None

_NULL_KEY_SENTINEL = "__NULL_KEY__"


def generate_recon_sql(old_sql: str, new_sql: str, key_cols: list[str], metric_cols: list[str]) -> str:
    old_sql = old_sql.strip().rstrip(";")
    new_sql = new_sql.strip().rstrip(";")

    key_select = ",\n  ".join(f"COALESCE(old.{k}, new_data.{k}) AS {k}" for k in key_cols)
    join_condition = "\n  AND ".join(
        f"COALESCE(CAST(old.{k} AS STRING), '{_NULL_KEY_SENTINEL}') = "
        f"COALESCE(CAST(new_data.{k} AS STRING), '{_NULL_KEY_SENTINEL}')"
        for k in key_cols
    )
    order_by = ", ".join(key_cols)

    metric_lines = []
    for i, m in enumerate(metric_cols, start=1):
        metric_lines.append(
            f"  -- {i}. Recon: {m}\n"
            f"  old.{m} AS old_{m},\n"
            f"  new_data.{m} AS new_{m},\n"
            f"  (COALESCE(new_data.{m}, 0) - COALESCE(old.{m}, 0)) AS delta_{m}"
        )
    metric_select = ",\n\n".join(metric_lines)

    return f"""WITH old_logic AS (
{old_sql}
),
new_logic AS (
{new_sql}
)
SELECT
  -- 1. Dimensions (Keys)
  {key_select},

{metric_select}

FROM old_logic AS old
FULL OUTER JOIN new_logic AS new_data
  ON {join_condition}
ORDER BY {order_by};
"""


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    flags = [a for a in sys.argv[1:] if a.startswith("--")]

    if len(args) not in (4, 5):
        print("Usage: python generate_recon.py old_query.sql new_query.sql key_cols.txt "
              "metric_cols.txt [output.sql] [--fix-ntile-tiebreak]")
        sys.exit(1)

    old_sql = open(args[0], encoding="utf-8").read()
    new_sql = open(args[1], encoding="utf-8").read()
    key_cols = [line.strip() for line in open(args[2]) if line.strip()]
    metric_cols = [line.strip() for line in open(args[3]) if line.strip()]
    output_path = args[4] if len(args) == 5 else None

    if "--fix-ntile-tiebreak" in flags:
        if apply_ntile_tiebreak is None:
            print("ERROR: apply_ntile_tiebreak.py not found next to this script.")
            sys.exit(1)
        new_sql, n = apply_ntile_tiebreak(new_sql)
        print(f"Applied NTILE tiebreak to {n} window(s) in new_sql (old_sql left untouched).",
              file=sys.stderr)

    result = generate_recon_sql(old_sql, new_sql, key_cols, metric_cols)

    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(result)
        print(f"Wrote {output_path}", file=sys.stderr)
    else:
        print(result)