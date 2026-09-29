"""
Generates a DRAFT migrated query + a list of warnings. Never trust the draft
blindly — it exists to remove the mechanical grunt work, not the validation step.

Usage:
    from migrate_query import migrate, classify_complexity
    new_sql, warnings = migrate(old_sql)
    bucket = classify_complexity(old_sql)
"""

import re
import sqlglot
from sqlglot import exp
from mapping_config import (OLD_PROJECT, NEW_PROJECT, TABLE_MAP, MANDATORY_FILTERS,
                             COLUMN_MAP, VALUE_REMAP_WATCHLIST, ALWAYS_FLAG_COLUMNS,
                             TYPE_OR_STRUCTURE_WATCHLIST, KNOWN_OLD_DATASETS, DROPPED_COLUMNS)

DIALECT = "bigquery"

_VERSION_LITERAL_RE = re.compile(r"^v(\d+)$", re.IGNORECASE)

CLEARED_ENGINE_VERSIONS = {
    ("bnpl", 1), ("bnpl", 2), ("bnpl", 3), ("bnpl", 4),
    ("cicilan", 1), ("cicilan", 2), ("cicilan", 3), ("cicilan", 4),
    ("starter", 4),
}


def _table_key(table: exp.Table) -> str | None:
    """Return 'schema.table' for a parsed Table node, ignoring project/catalog."""
    if table.this is None:
        return None
    schema = table.args.get("db")
    if schema is None:
        return None
    return f"{schema.name}.{table.this.name}"


_MODE_TAG_RE = re.compile(r"\{%-?.*?-?%\}", re.DOTALL)
_MODE_VAR_RE = re.compile(r"\{\{-?.*?-?\}\}", re.DOTALL)
_MODE_PARAM_YAML_RE = re.compile(
    r"\n\s*\n([a-zA-Z_][\w ]*):\s*\n(?:[ \t]+.*\n?)+", re.MULTILINE
)


def _strip_mode_param_yaml(sql: str) -> tuple[str, list[str]]:
    info = []
    match = _MODE_PARAM_YAML_RE.search(sql)
    if not match:
        return sql, info
    param_name = match.group(1).strip()
    block = match.group(0).strip()
    info.append(f"MODE PARAMETER DEFINITION FOUND for '{param_name}': {block[:300]}"
                f"{'...' if len(block) > 300 else ''} — this tells you the REAL allowed "
                f"range for this filter; check it before assuming date-range risk.")
    return sql[:match.start()], info


_MODE_VAR_NAME_RE = re.compile(r"\{\{-?\s*(\w+)\s*-?\}\}")


def _strip_mode_templating(sql: str) -> tuple[str, bool]:
    had_template = bool(_MODE_TAG_RE.search(sql) or _MODE_VAR_RE.search(sql))
    cleaned = _MODE_TAG_RE.sub("", sql)
    cleaned = _MODE_VAR_NAME_RE.sub(
        lambda m: f"__MODE_PARAM_{m.group(1).upper()}__", cleaned
    )
    return cleaned, had_template


_STRING_AGG_RE = re.compile(r"string_agg\s*\(", re.IGNORECASE)
_IGNORE_NULLS_RE = re.compile(r"\s*IGNORE\s+NULLS\s*", re.IGNORECASE)


def _strip_string_agg_ignore_nulls(sql: str) -> tuple[str, int]:
    out = []
    i = 0
    count = 0
    while True:
        m = _STRING_AGG_RE.search(sql, i)
        if not m:
            out.append(sql[i:])
            break
        out.append(sql[i:m.end()])
        depth = 1
        j = m.end()
        start_inner = j
        while j < len(sql) and depth > 0:
            if sql[j] == "(":
                depth += 1
            elif sql[j] == ")":
                depth -= 1
            j += 1
        inner = sql[start_inner:j - 1]
        new_inner, n = _IGNORE_NULLS_RE.subn(" ", inner)
        count += n
        out.append(new_inner)
        out.append(")")
        i = j
    return "".join(out), count


def classify_complexity(sql: str) -> str:
    sql, _ = _strip_mode_templating(sql)
    sql, _ = _strip_mode_param_yaml(sql)
    sql, _ = _strip_string_agg_ignore_nulls(sql)
    try:
        tree = sqlglot.parse_one(sql, read=DIALECT)
    except Exception as e:
        return f"PARSE_ERROR — {e}"
    n_ctes = len(list(tree.find_all(exp.CTE)))
    n_subq = len(list(tree.find_all(exp.Subquery)))
    n_windows = len(list(tree.find_all(exp.Window)))
    n_joins = len(list(tree.find_all(exp.Join)))
    score = n_ctes * 3 + n_subq * 2 + n_windows * 3 + n_joins
    if score == 0:
        return "SIMPLE"
    elif score <= 4:
        return "MODERATE"
    else:
        return "COMPLEX — review manually before trusting any auto-draft"


def _find_owning_join(table: exp.Table) -> exp.Join | None:
    """Return the Join node that directly introduces this table, if any --
    i.e. the join whose 'this' IS (or contains, for a wrapped derived table)
    this specific table. Returns None if the table is the primary FROM table,
    sits inside a UNION/UNION ALL branch, or is otherwise not the direct
    target of a JOIN clause.

    FIXED 2026-08-18: confirmed by a real bug (query 796e2bc762af) that this
    function was walking PAST UNION ALL boundaries to find an outer LEFT JOIN,
    even when the table sits inside one branch of a UNION combining two
    DIFFERENT old sources (e.g. views.acquired_users_by_spg_all_type UNION ALL
    views.acquired_users_by_kredimitra_all_type, both now dwh.dim_user_offline).
    This caused two different mandatory filters to be attributed to the SAME
    outer join's ON clause -- producing either a logically-impossible AND of
    two different table_reference values (silently breaking the join), or in
    some nesting shapes, the filter being dropped entirely and never injected
    anywhere (confirmed by a second real query where table_reference was
    missing from BOTH UNION branches after migration).

    Each UNION branch is its own independent query with its own WHERE clause --
    a table inside one branch must ever be filtered via THAT branch's own
    WHERE, never via an outer join's ON, since the ON clause has no way to
    express "this filter applies only to rows from branch 1."
    """
    enclosing_select = table.find_ancestor(exp.Select)
    if enclosing_select is not None and isinstance(enclosing_select.parent, exp.Union):
        return None

    join = table.find_ancestor(exp.Join)
    if join is None:
        return None
    if join.this is table:
        return join
    if isinstance(join.this, (exp.Subquery,)) and table in join.this.find_all(exp.Table):
        return join
    return None


def migrate(sql: str) -> tuple[str, list[str]]:
    warnings: list[str] = []

    sql, had_template = _strip_mode_templating(sql)
    if had_template:
        warnings.append(
            "MODE TEMPLATING DETECTED: this query used Mode's Liquid syntax "
            "({% if %} / {{ param }}) for dynamic filters. These were stripped "
            "(replaced with __MODE_PARAM__ placeholders) so the draft below "
            "could be parsed at all — the dynamic filter logic is NOT migrated "
            "automatically. Re-add the actual Mode filter blocks by hand after "
            "confirming the underlying column/table renames are correct."
        )

    sql, yaml_info = _strip_mode_param_yaml(sql)
    warnings.extend(yaml_info)

    sql, n_string_agg_fixes = _strip_string_agg_ignore_nulls(sql)
    if n_string_agg_fixes:
        warnings.append(
            f"STRING_AGG(...IGNORE NULLS...) FOUND ({n_string_agg_fixes}x): this is a known "
            f"sqlglot parser gap (confirmed against latest version 2026-07-21) — 'IGNORE NULLS' "
            f"was stripped from STRING_AGG calls so this query could be parsed at all. "
            f"YOU MUST manually add 'IGNORE NULLS' back into every STRING_AGG call in the final "
            f"query before using it, or NULL values will incorrectly get included in the "
            f"concatenated string."
        )

    try:
        tree = sqlglot.parse_one(sql, read=DIALECT)
    except Exception as e:
        return sql, [f"PARSE FAILED, no draft generated: {e}"]

    # --- 0. build alias -> OLD table key map, PER ENCLOSING SELECT, BEFORE any renaming ---
    alias_scope_map: dict[tuple[int, str], str] = {}
    for table in tree.find_all(exp.Table):
        key = _table_key(table)
        if key is None:
            continue
        enclosing_select = table.find_ancestor(exp.Select)
        if enclosing_select is None:
            continue
        alias_scope_map[(id(enclosing_select), table.alias_or_name)] = key

    touched_new_tables: set[tuple[str, str | None]] = set()

    # --- 1. rename tables, tracking BOTH the enclosing SELECT (for FROM/INNER-joined
    #        tables, where WHERE-injection is safe) AND the owning Join node (for
    #        LEFT-joined tables, where the filter MUST go in the ON clause) ---
    #
    # FIXED 2026-08-12: confirmed by a real bug (query ed2e1cf7f48c, dim_user_offline)
    # that injecting a mandatory filter into the outer WHERE clause is UNSAFE when the
    # filtered table is the target of a LEFT JOIN. SQL's NULL semantics mean
    # "WHERE t2.col = 'x'" evaluates to NULL (not TRUE) for any row where the LEFT JOIN
    # found no match — silently dropping that row from the result entirely, converting
    # the LEFT JOIN into a de-facto INNER JOIN. Confirmed impact: 2,052 of 195,583 rows
    # (~1.05%) silently dropped in one real query, which was enough to measurably shift
    # NTILE decile boundaries for the entire cohort. This likely affects every
    # already-migrated query using dim_internal_teams/dim_internal_collectors too
    # (current_flag=1 filter, same LEFT JOIN + WHERE pattern) — worth a re-audit.
    filter_targets: dict[int, tuple[exp.Select, set[str]]] = {}
    join_filter_targets: dict[int, tuple[exp.Join, set[str]]] = {}

    for table in tree.find_all(exp.Table):
        key = _table_key(table)
        if key is None:
            continue
        mapping = TABLE_MAP.get(key)
        if mapping is None:
            dataset = key.split(".")[0]
            if dataset in KNOWN_OLD_DATASETS:
                warnings.append(f"UNMAPPED TABLE '{key}' — dataset '{dataset}' is a known "
                                 f"migration-scope source but this specific table isn't mapped "
                                 f"yet. Add it to mapping_config.py.")
            continue

        new_full = mapping["new"]
        new_db, new_tbl = new_full.split(".")
        table.set("db", exp.to_identifier(new_db))
        table.set("this", exp.to_identifier(new_tbl))
        # FIXED 2026-09-29: always fully-qualify the renamed table with NEW_PROJECT,
        # even when the original reference had no project prefix at all (e.g. bare
        # `l2alpha.transaction`, no backticks). The old guard only added a project
        # when the ORIGINAL reference already had one -- an unqualified old reference
        # stayed unqualified after rename, silently resolving against whichever
        # project happens to be the BQ session/job's default at execution time, not
        # necessarily NEW_PROJECT. Confirmed real: scratch/new_query.sql for this
        # exact table was hand-fixed by adding the project prefix manually after
        # running migrate() -- this makes that manual step unnecessary going forward.
        table.set("catalog", exp.to_identifier(NEW_PROJECT))
        # sqlglot only renders catalog.db.table as ONE fused `a.b.c` backtick block
        # (matching every other migrated reference in this file's own regression
        # tests) when the Table node's meta['quoted_table'] flag is set -- that flag
        # is set by the PARSER when the ORIGINAL source text already had the whole
        # path in one backtick pair, and survives .set() on individual fields.
        # A table with NO catalog before this rename never had that flag set, so
        # without this line it silently renders as `project`.dataset.table instead
        # (three separately-quoted parts) -- confirmed by direct sqlglot inspection,
        # not just assumed. Both forms may well be valid BigQuery syntax, but there
        # is no reason to introduce a second, untested quoting style when this one
        # already matches everything else this tool has ever produced.
        table.meta["quoted_table"] = True

        source_type = mapping["source_type"]
        touched_new_tables.add((new_full, source_type))
        filt_key = (new_full, source_type)

        if filt_key in MANDATORY_FILTERS:
            owning_join = _find_owning_join(table)
            if owning_join is not None and owning_join.side == "LEFT":
                jid = id(owning_join)
                if jid not in join_filter_targets:
                    join_filter_targets[jid] = (owning_join, set())
                join_filter_targets[jid][1].add(MANDATORY_FILTERS[filt_key])
                warnings.append(
                    f"'{new_full}' is LEFT-joined — mandatory filter "
                    f"({MANDATORY_FILTERS[filt_key]}) injected into the JOIN's ON clause, "
                    f"NOT the outer WHERE, to avoid silently dropping unmatched rows "
                    f"(confirmed bug 2026-08-12, see code comment)."
                )
            else:
                enclosing_select = table.find_ancestor(exp.Select)
                if enclosing_select is not None:
                    sel_id = id(enclosing_select)
                    if sel_id not in filter_targets:
                        filter_targets[sel_id] = (enclosing_select, set())
                    filter_targets[sel_id][1].add(MANDATORY_FILTERS[filt_key])
                else:
                    warnings.append(f"Could not find an enclosing SELECT for a '{new_full}' "
                                     f"reference — add filter manually: {MANDATORY_FILTERS[filt_key]}")
        elif source_type is not None and filt_key not in MANDATORY_FILTERS:
            warnings.append(f"'{new_full}' (source_type={source_type}) has no mandatory filter "
                             f"registered in MANDATORY_FILTERS — confirm none is needed")

    # --- 2. rename columns (only when we recognize the name AND can confirm — or at least not
    #        rule out — that it belongs to a table actually in this migration's scope) ---
    already_flagged_names: set[str] = set()
    for col in tree.find_all(exp.Column):
        old_name = col.this.name
        enclosing_select = col.find_ancestor(exp.Select)
        resolved_table_key = None
        if enclosing_select is not None:
            qualifier = col.table
            if qualifier:
                resolved_table_key = alias_scope_map.get((id(enclosing_select), qualifier))
            else:
                candidates = {a: k for (sid, a), k in alias_scope_map.items() if sid == id(enclosing_select)}
                if len(candidates) == 1:
                    resolved_table_key = next(iter(candidates.values()))
                elif len(candidates) > 1:
                    resolved_table_key = "AMBIGUOUS"

        confirmed_out_of_scope = (
            resolved_table_key is not None
            and resolved_table_key != "AMBIGUOUS"
            and resolved_table_key not in TABLE_MAP
        )
        if confirmed_out_of_scope:
            continue

        unresolved = resolved_table_key is None or resolved_table_key == "AMBIGUOUS"
        if unresolved and old_name in COLUMN_MAP and old_name not in already_flagged_names:
            warnings.append(
                f"COULD NOT CONFIRM SOURCE TABLE for column '{old_name}' (unqualified or "
                f"ambiguous scope) — renamed based on name only, same as before this check "
                f"existed. If this instance actually belongs to an out-of-scope table (e.g. "
                f"appsheet.*), this specific rename is WRONG. Verify manually."
            )

        if old_name in DROPPED_COLUMNS and old_name not in already_flagged_names:
            warnings.append(f"DROPPED COLUMN '{old_name}': no counterpart in the new table. "
                             f"{DROPPED_COLUMNS[old_name]}")
            already_flagged_names.add(old_name)

        if old_name in TYPE_OR_STRUCTURE_WATCHLIST and old_name not in already_flagged_names:
            warnings.append(f"TYPE/STRUCTURE CHECK on '{old_name}': {TYPE_OR_STRUCTURE_WATCHLIST[old_name]}")

        if old_name in ALWAYS_FLAG_COLUMNS:
            if old_name not in already_flagged_names:
                warnings.append(f"AMBIGUOUS COLUMN '{old_name}' NOT auto-renamed (appears on "
                                 f"multiple aliases in this query — check each occurrence): "
                                 f"{ALWAYS_FLAG_COLUMNS[old_name]}")
                already_flagged_names.add(old_name)
            continue

        already_flagged_names.add(old_name)
        if old_name in COLUMN_MAP:
            new_name = COLUMN_MAP[old_name]
            col.this.set("this", new_name)
            if old_name in VALUE_REMAP_WATCHLIST:
                warnings.append(f"VALUE-LOGIC CHECK on column '{old_name}'->'{new_name}': "
                                 f"{VALUE_REMAP_WATCHLIST[old_name]}")
            else:
                warnings.append(f"Renamed column '{old_name}' -> '{new_name}' — confirm semantics match")

    # --- 3a. inject mandatory filters that belong in a LEFT JOIN's ON clause ---
    for join_node, filters in join_filter_targets.values():
        combined = None
        for f in sorted(filters):
            cond = exp.condition(f)
            combined = cond if combined is None else exp.and_(combined, cond)
        existing_on = join_node.args.get("on")
        join_node.set("on", exp.and_(existing_on, combined) if existing_on is not None else combined)
        if len(filters) > 1:
            warnings.append(f"Multiple different mandatory filters applied to the SAME LEFT "
                             f"JOIN ({sorted(filters)}) — check manually.")

    # --- 3b. inject mandatory filters for FROM-table/INNER-joined tables into the WHERE ---
    for select_node, filters in filter_targets.values():
        for f in sorted(filters):
            select_node.where(f, copy=False)
        if len(filters) > 1:
            warnings.append(f"Multiple different mandatory filters applied to the SAME select "
                             f"scope ({sorted(filters)}) — this usually means the same fact table "
                             f"is joined twice under different source_types in one scope, which "
                             f"AND-combining filters cannot correctly express. Check manually.")

    if filter_targets or join_filter_targets:
        n_total = len(filter_targets) + len(join_filter_targets)
        warnings.append(f"Injected mandatory filter(s) into {n_total} location(s) in this query "
                         f"({len(join_filter_targets)} inside LEFT JOIN ON clauses, "
                         f"{len(filter_targets)} inside WHERE clauses) — verify EACH one.")

    # --- 3c. auto-convert confirmed 'vN' -> N literal format for the version column ---
    for lit in tree.find_all(exp.Literal):
        if not lit.is_string:
            continue
        m = _VERSION_LITERAL_RE.match(lit.this)
        if not m:
            continue
        parent = lit.parent
        sibling_is_version_col = parent is not None and any(
            node.this.name == "version" for node in parent.find_all(exp.Column)
        )
        if sibling_is_version_col:
            lit.replace(exp.Literal.number(int(m.group(1))))
            warnings.append(f"Auto-converted version literal 'v{m.group(1)}' -> {m.group(1)} "
                             f"(format confirmed 2026-07-21, still correct after the 2026-07-27 backfill)")

    # --- 3d. informational note for bscore engine/version combos (RESOLVED 2026-07-27) ---
    engines_referenced = {lit.this for lit in tree.find_all(exp.Literal) if lit.is_string}
    if any(e in ("bnpl", "cicilan", "starter") or "kid_risk" in e for e in engines_referenced):
        warnings.append(
            "RECONCILIATION STATUS (resolved 2026-07-27): the bnpl/cicilan v1-v3 gap that used "
            "to block this combo was fixed via backfill — verified v1/v3 exact match, v2 within "
            "0.001%. No longer a blocker. This note is now purely informational; if you see a "
            "NEW discrepancy, it's not this old issue recurring — investigate as a fresh problem."
        )

    new_sql = tree.sql(dialect=DIALECT, pretty=True)
    deduped_warnings = list(dict.fromkeys(warnings))
    return new_sql, deduped_warnings


if __name__ == "__main__":
    # regression case: FROM-table + INNER JOIN (existing behavior, must be unchanged)
    sample = """
    SELECT
        debtor_id,
        teams.id AS team_id,
        user_id,
        DATE(MIN(visit_date)) AS visit_date,
        MIN(result) AS result
    FROM `kre-data-warehouse-prod-6615.dragon.visit_reports` vr
    JOIN `kre-data-warehouse-prod-6615.komodo_dak.teams` teams ON vr.team_id = teams.id
    WHERE DATE(visit_date) >= DATE_TRUNC(CURRENT_DATE() - 1, MONTH)
    GROUP BY 1, 2, 3
    """
    print("=== REGRESSION: FROM-table + INNER JOIN (filter must stay in WHERE) ===")
    print(classify_complexity(sample))
    new_sql, warns = migrate(sample)
    print(new_sql)
    for w in warns:
        print("-", w)

    # new case: table needing a mandatory filter is LEFT-joined — filter must go in ON
    left_join_case = """
    SELECT a.user_id, teams.name
    FROM `kre-data-warehouse-prod-6615.komodo_dak.field_assignments` a
    LEFT JOIN `kre-data-warehouse-prod-6615.komodo_dak.teams` teams ON teams.id = a.team_id
    WHERE 1=1
    """
    print("\n=== NEW CASE: LEFT JOIN (filter must go in ON, not WHERE) ===")
    new_sql2, warns2 = migrate(left_join_case)
    print(new_sql2)
    for w in warns2:
        print("-", w)

    # new case: two DIFFERENT old sources combined via UNION ALL inside one
    # outer LEFT JOIN -- each branch needs its OWN filter in its OWN WHERE,
    # never merged onto the shared outer join's ON clause (confirmed real bug,
    # query 796e2bc762af, 2026-08-18)
    union_in_join_case = """
    SELECT a.user_id, c.spg_channel
    FROM t a
    LEFT JOIN (
      SELECT creation_time, user_id, 'Kredimitra' AS spg_channel
      FROM `kre-data-warehouse-prod-6615.views.acquired_users_by_kredimitra_all_type`
      UNION ALL
      SELECT creation_time, user_id, 'Instore SPG' AS spg_channel
      FROM `kre-data-warehouse-prod-6615.views.acquired_users_by_spg_all_type`
    ) c ON a.user_id = c.user_id
    """
    print("\n=== NEW CASE: two different sources UNION ALL'd inside one outer LEFT JOIN ===")
    new_sql3, warns3 = migrate(union_in_join_case)
    print(new_sql3)
    for w in warns3:
        print("-", w)