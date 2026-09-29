"""
Streamlit front-end for the L1 -> L2 migration helper.

Run locally with:
    pip install streamlit sqlglot google-cloud-bigquery --break-system-packages
    streamlit run app.py

IMPORTANT: app.py, mapping_writer.py, and bq_dry_run.py must all live in the
SAME folder as migrate_query.py / mapping_config.py (repo root). Python only
auto-adds a script's own directory to sys.path -- putting any of these three
in scratch/ or a subfolder will produce a ModuleNotFoundError on import,
same failure mode as the run_single.py/scratch/ issue already hit once.

This is a UI wrapper. It does not reimplement migrate_query.py's,
generate_recon.py's, or generate_key_metric_cols.py's logic -- it imports
and calls them directly, exactly as the CLI does.
"""

import base64
import io
import csv as csv_mod
from pathlib import Path

import streamlit as st


def _get_secret(key, default=None):
    """st.secrets.get() raises StreamlitSecretNotFoundError -- not a KeyError,
    so the default Mapping.get() mixin doesn't catch it -- when there is NO
    secrets.toml file at all (confirmed by direct testing, not assumed: it
    works fine once a file exists but lacks the key, only breaks when the
    file is fully absent). Every local run with no secrets file configured,
    i.e. every run on this machine today, would otherwise crash outright on
    the very first line below. This wrapper is the only thing making the
    'local behavior is completely unchanged' claim in this file actually true."""
    try:
        return st.secrets.get(key, default)
    except Exception:
        return default


# ---------------------------------------------------------------------------
# HOSTED-DEPLOYMENT BOOTSTRAP -- must run before any mapping_config-dependent
# import below (migrate_query.py does `from mapping_config import ...` at ITS
# OWN top level, so mapping_config.py must exist on disk before that import
# executes, not just before app.py's own `import mapping_config` line).
#
# Locally: mapping_config.py already exists in this folder, this block is a
# no-op, nothing changes about your existing workflow.
#
# On a hosted deployment (e.g. Streamlit Community Cloud) where the backing
# git repo deliberately never contains mapping_config.py -- real project IDs
# and business logic have no business sitting in a repo whose full history
# would still expose them even after a later .gitignore -- this reconstructs
# the file from a secret instead. See the deployment README for how to set
# MAPPING_CONFIG_PY_B64.
# ---------------------------------------------------------------------------
_MAPPING_CONFIG_PATH = Path(__file__).parent / "mapping_config.py"
if not _MAPPING_CONFIG_PATH.exists():
    _mc_b64 = _get_secret("MAPPING_CONFIG_PY_B64")
    if not _mc_b64:
        st.error(
            "mapping_config.py is missing on disk and no MAPPING_CONFIG_PY_B64 "
            "secret is configured, so this app cannot start. This is expected "
            "on a fresh hosted deployment before secrets are set -- see the "
            "deployment README."
        )
        st.stop()
    try:
        _MAPPING_CONFIG_PATH.write_bytes(base64.b64decode(_mc_b64))
    except Exception as e:
        st.error(f"MAPPING_CONFIG_PY_B64 secret is set but couldn't be decoded ({e}). "
                 f"Re-check it was base64-encoded correctly, no line breaks.")
        st.stop()

# ---------------------------------------------------------------------------
# WHOLE-APP PASSWORD GATE -- only active when an APP_PASSWORD secret exists.
# Locally, with no secrets.toml, st.secrets.get() returns None and this is a
# complete no-op -- nothing changes about running this on your own machine.
#
# This is Streamlit's own documented fallback for teams without SSO ("adds
# some level of security... NOT comparable to proper authentication with an
# SSO provider" -- their words). It's a shared password, not per-user auth:
# no identity, no audit trail, no revoking one person without changing it for
# everyone. It stops a random internet visitor from seeing anything. It does
# not stop someone the password was shared with from sharing it further.
# ---------------------------------------------------------------------------
_app_password = _get_secret("APP_PASSWORD")
if _app_password and not st.session_state.get("_authed"):
    st.title("Kredivo L1 → L2 Migration Helper")
    entered = st.text_input("Password", type="password", key="_password_input")
    if entered:
        if entered == _app_password:
            st.session_state["_authed"] = True
            st.rerun()
        else:
            st.error("Wrong password.")
    st.stop()

# Feature flag: the mapping-write form. Defaults ON (matches existing local
# behavior with no secrets.toml at all). Set ENABLE_MAPPING_WRITE = "false" in
# the hosted deployment's secrets to hide it there -- writes made through a
# hosted instance don't survive a container restart anyway (no persistent
# volume by default on Community Cloud), so the real edit workflow should
# stay on your machine, where mapping_writer.py's backup+rollback and your
# git history actually mean something. Teammates get read + Input + Recon +
# Batch on the hosted copy; mapping changes are still yours to make and push.
ENABLE_MAPPING_WRITE = str(_get_secret("ENABLE_MAPPING_WRITE", "true")).lower() != "false"

from migrate_query import migrate, classify_complexity
import mapping_config
from generate_key_metric_cols import classify_columns
from generate_recon import generate_recon_sql
from mapping_writer import add_table_mapping, MappingWriteError

try:
    from apply_ntile_tiebreak import apply_ntile_tiebreak
    NTILE_FIX_AVAILABLE = True
except ImportError:
    apply_ntile_tiebreak = None
    NTILE_FIX_AVAILABLE = False

try:
    from bulk_process import process_row
    BULK_AVAILABLE = True
    BULK_IMPORT_ERROR = None
except ImportError as e:
    process_row = None
    BULK_AVAILABLE = False
    BULK_IMPORT_ERROR = str(e)

try:
    from bq_dry_run import estimate_bytes, DryRunError
    BQ_AVAILABLE = True
except ImportError:
    estimate_bytes = None
    DryRunError = Exception
    BQ_AVAILABLE = False

MAPPING_CONFIG_PATH = mapping_config.__file__

st.set_page_config(page_title="L1 -> L2 Migration Helper", layout="wide")

# ---------------------------------------------------------------------------
# Heuristic DPD/DNC/collection-logic flag. Deliberately over-inclusive
# (keyword match on the raw + migrated SQL text) rather than under-inclusive:
# a false positive here just means an extra reminder banner; a false negative
# means a DPD-touching query ships without the extra row-level check.
# ---------------------------------------------------------------------------
_SENSITIVE_KEYWORDS = (
    "dpd", "dnc", "exclusion", "collection", "b_score", "bscore",
    "field_report", "komodo", "dragon",
)


def is_collection_sensitive(*sql_texts: str) -> bool:
    combined = " ".join(sql_texts).lower()
    return any(kw in combined for kw in _SENSITIVE_KEYWORDS)


def render_dry_run_button(sql: str, key_prefix: str):
    """Shared dry-run UI block, used by both the Input and Recon tabs."""
    st.markdown("**BigQuery cost check (dry-run only — never executes, never returns rows):**")
    if not BQ_AVAILABLE:
        st.caption("google-cloud-bigquery not installed — "
                    "`pip install google-cloud-bigquery --break-system-packages` to enable this.")
        return
    if st.button("Estimate bytes scanned", key=f"{key_prefix}_dry_run_btn"):
        billing_project = st.session_state.get("bq_billing_project") or None
        try:
            with st.spinner("Running BigQuery dry-run..."):
                est = estimate_bytes(sql, billing_project=billing_project)
        except DryRunError as e:
            st.error(str(e))
        else:
            threshold = st.session_state.get("bq_gb_threshold", 10.0)
            if est["total_bytes_gb"] >= threshold:
                st.error(f"~{est['total_bytes_gb']} GB estimated — over your {threshold} GB warning "
                         f"threshold. Check for a missing partition filter before running this for real.")
            else:
                st.success(f"~{est['total_bytes_gb']} GB estimated")
            if est["referenced_tables"]:
                st.caption("Tables referenced: " + ", ".join(est["referenced_tables"]))


# ---------------------------------------------------------------------------
# Sidebar: cross-cutting settings only. The mapping-config browser moved to
# its own tab (see Known Mappings) since it's a real feature now, not a
# read-only reference sitting in the sidebar.
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Settings")
    st.session_state["bq_billing_project"] = st.text_input(
        "BQ billing project for dry-run",
        value=st.session_state.get("bq_billing_project", ""),
        placeholder="blank = your gcloud ADC default",
        help="Never hardcoded — this is the project the dry-run job bills against, "
             "not necessarily the project of the tables being queried.",
    )
    st.session_state["bq_gb_threshold"] = st.number_input(
        "Warn if estimated scan exceeds (GB)",
        min_value=0.1,
        value=st.session_state.get("bq_gb_threshold", 10.0),
        step=1.0,
    )
    if not BQ_AVAILABLE:
        st.caption("⚠️ google-cloud-bigquery not installed")
    if not NTILE_FIX_AVAILABLE:
        st.caption("⚠️ apply_ntile_tiebreak.py not found — tiebreak option disabled")
    if not BULK_AVAILABLE:
        st.caption("⚠️ Batch tab disabled — see that tab for why")

st.title("Kredivo L1 → L2 Migration Helper")
st.caption(
    "Paste a legacy Mode query, get a DRAFT migration + a list of everything the "
    "tool is NOT sure about. This drafts, it does not validate — always run the "
    "reconciliation query yourself before touching the live Mode report."
)

tab_input, tab_recon, tab_batch, tab_mappings = st.tabs(
    ["Input", "Recon", "Batch", "Known Mappings"]
)

# ---------------------------------------------------------------------------
# TAB 1: Input — single-query migrate()
# ---------------------------------------------------------------------------
with tab_input:
    col_in, col_out = st.columns(2)

    with col_in:
        st.subheader("Legacy query (paste from Mode)")
        query_id = st.text_input("query_id (for your tracking sheet)", key="input_query_id")
        raw_sql = st.text_area("Legacy SQL", height=400, key="input_raw_sql",
                                placeholder="SELECT ... FROM `kre-data-warehouse-prod-6615...`")
        run_btn = st.button("Migrate", type="primary", key="input_run")

    if run_btn:
        if not raw_sql.strip():
            st.session_state["input_result"] = None
            with col_out:
                st.error("Paste a query first.")
        else:
            with st.spinner("Migrating..."):
                complexity = classify_complexity(raw_sql)
                new_sql, warnings = migrate(raw_sql)
            st.session_state["input_result"] = {
                "query_id": query_id,
                "old_sql": raw_sql,
                "new_sql": new_sql,
                "complexity": complexity,
                "warnings": warnings,
            }

    with col_out:
        st.subheader("Draft output")
        result = st.session_state.get("input_result")
        if result:
            complexity = result["complexity"]
            if "SIMPLE" in complexity:
                st.success(f"Complexity: {complexity}")
            elif "MODERATE" in complexity:
                st.warning(f"Complexity: {complexity}")
            else:
                st.error(f"Complexity: {complexity}")

            if is_collection_sensitive(result["old_sql"], result["new_sql"]):
                st.error(
                    "⚠️ DPD/DNC/collection-adjacent query (keyword match, so double-check it's not "
                    "a false positive). Get sign-off before marking Done, and generate the row-level "
                    "EXCEPT DISTINCT check in the Recon tab in addition to the aggregate recon."
                )

            st.markdown("**Draft migrated SQL** (verify before pasting into Mode):")
            st.code(result["new_sql"], language="sql")
            st.download_button(
                "Download draft .sql", result["new_sql"],
                file_name=f"{result['query_id'] or 'draft'}.sql", key="input_download",
            )

            st.markdown("**Warnings — read every single one:**")
            if result["warnings"]:
                for w in result["warnings"]:
                    st.markdown(f"- ⚠️ {w}")
            else:
                st.info("No warnings raised — still run reconciliation before trusting this.")

            if st.button("Send old + new SQL to Recon tab →", key="send_to_recon"):
                st.session_state["recon_old_sql_seed"] = result["old_sql"]
                st.session_state["recon_new_sql_seed"] = result["new_sql"]
                st.success("Sent — open the Recon tab.")

            st.divider()
            render_dry_run_button(result["new_sql"], key_prefix="input")
        else:
            st.caption("Output will appear here after you click Migrate.")

# ---------------------------------------------------------------------------
# TAB 2: Recon — standalone old_sql/new_sql -> recon SQL, same as the CLI's
# generate_key_metric_cols.py + generate_recon.py pair, decoupled from the
# Input tab so you can also recon a HAND-EDITED draft, not just a fresh one.
# ---------------------------------------------------------------------------
with tab_recon:
    st.caption(
        "Auto-generates the old_logic/new_logic FULL OUTER JOIN comparison query. "
        "This does NOT execute anything against production — copy the result into "
        "the BQ console yourself, per team discipline."
    )

    old_sql = st.text_area(
        "Old (legacy) SQL", value=st.session_state.get("recon_old_sql_seed", ""),
        height=180, key="recon_old_sql",
    )
    new_sql = st.text_area(
        "New (migrated) SQL", value=st.session_state.get("recon_new_sql_seed", ""),
        height=180, key="recon_new_sql",
    )

    if st.button("Auto-detect key/metric columns", key="recon_classify_btn"):
        if not old_sql.strip():
            st.error("Paste the old SQL first.")
        else:
            try:
                keys, metrics, flagged = classify_columns(old_sql)
                st.session_state["recon_keys_seed"] = "\n".join(keys)
                st.session_state["recon_metrics_seed"] = "\n".join(metrics)
                st.session_state["recon_flagged"] = flagged
            except Exception as e:
                st.error(
                    f"Could not parse this SQL for column classification ({e}). If this came "
                    f"straight from Mode, it may still contain Liquid templating ({{% %}} / "
                    f"{{{{ param }}}}) — run it through the Input tab's Migrate step first, which "
                    f"strips that, or clean it up manually before pasting here."
                )

    col_k, col_m = st.columns(2)
    with col_k:
        keys_text = st.text_area(
            "Key columns (one per line)",
            value=st.session_state.get("recon_keys_seed", ""), height=120, key="recon_keys_input",
        )
    with col_m:
        metrics_text = st.text_area(
            "Metric columns (one per line)",
            value=st.session_state.get("recon_metrics_seed", ""), height=120, key="recon_metrics_input",
        )

    flagged = st.session_state.get("recon_flagged", [])
    if flagged:
        st.warning(
            "Flagged for manual review — NOT included above, decide and add them yourself:\n"
            + "\n".join(f"- {f}" for f in flagged)
        )

    fix_ntile = False
    if NTILE_FIX_AVAILABLE:
        fix_ntile = st.checkbox(
            "Apply confirmed NTILE tiebreak fix to new_sql only (apply_ntile_tiebreak.py)",
            key="recon_fix_ntile",
        )
    else:
        st.caption("apply_ntile_tiebreak.py not found next to app.py — tiebreak option disabled.")

    if st.button("Generate reconciliation SQL", type="primary", key="recon_generate_btn"):
        keys = [k.strip() for k in keys_text.splitlines() if k.strip()]
        metrics = [m.strip() for m in metrics_text.splitlines() if m.strip()]
        if not old_sql.strip() or not new_sql.strip():
            st.error("Need both old and new SQL.")
        elif not keys:
            st.error("Need at least one key column — click auto-detect or add one manually.")
        elif not metrics:
            # Matches bulk_process.py's own behavior: it deliberately SKIPS calling
            # generate_recon_sql with zero metrics rather than let the join template
            # break (empty metric_select leaves a dangling comma before FROM).
            st.warning(
                "No metric columns — generate_recon.py's template breaks with zero metrics "
                "(dangling comma before FROM), same as the CLI. Add at least one metric column."
            )
        else:
            working_new_sql = new_sql
            n_fixed = 0
            if fix_ntile:
                working_new_sql, n_fixed = apply_ntile_tiebreak(working_new_sql)
            try:
                recon_sql = generate_recon_sql(old_sql, working_new_sql, keys, metrics)
            except Exception as e:
                st.error(f"generate_recon_sql failed: {e}")
            else:
                if fix_ntile:
                    st.info(f"Applied tiebreak to {n_fixed} window(s) in new_sql only (old_sql untouched).")
                st.session_state["recon_result_sql"] = recon_sql
                st.session_state["recon_result_old_sql"] = old_sql
                st.session_state["recon_result_new_sql"] = working_new_sql
                st.session_state["recon_is_sensitive"] = is_collection_sensitive(old_sql, working_new_sql)

    recon_sql = st.session_state.get("recon_result_sql")
    if recon_sql:
        st.markdown("**Reconciliation query:**")
        st.code(recon_sql, language="sql")
        st.download_button("Download recon .sql", recon_sql, file_name="recon.sql", key="recon_download")

        if st.session_state.get("recon_is_sensitive"):
            st.error(
                "⚠️ DPD/DNC/collection-adjacent query — run the row-level check below "
                "BOTH WAYS in addition to the aggregate recon above before marking Done."
            )
            o = st.session_state["recon_result_old_sql"].strip().rstrip(";")
            n = st.session_state["recon_result_new_sql"].strip().rstrip(";")
            except_sql = (
                f"-- Row-level check, direction 1 (rows in OLD but not in NEW)\n"
                f"SELECT * FROM (\n{o}\n)\nEXCEPT DISTINCT\nSELECT * FROM (\n{n}\n);\n\n"
                f"-- Row-level check, direction 2 (rows in NEW but not in OLD)\n"
                f"SELECT * FROM (\n{n}\n)\nEXCEPT DISTINCT\nSELECT * FROM (\n{o}\n);"
            )
            st.code(except_sql, language="sql")
            st.download_button("Download EXCEPT DISTINCT check .sql", except_sql,
                                file_name="row_level_check.sql", key="recon_except_download")

        st.divider()
        render_dry_run_button(recon_sql, key_prefix="recon")

# ---------------------------------------------------------------------------
# TAB 3: Batch — wraps bulk_process.process_row() over an uploaded CSV.
# Needs auto_recon.py (bulk_process.py's own hard import) -- not provided, so
# this tab disables itself with a clear reason instead of crashing app startup.
# ---------------------------------------------------------------------------
with tab_batch:
    if not BULK_AVAILABLE:
        st.error(
            f"bulk_process.py could not be imported: {BULK_IMPORT_ERROR}\n\n"
            f"It needs auto_recon.py in this same folder (source not yet provided). "
            f"Add it to enable this tab, or run bulk_process.py from the CLI for now."
        )
    else:
        st.caption(
            "Upload a CSV with columns: query_id, raw_sql (mode_url optional) — same shape "
            "as scratch/fetched_queries_*.csv. Output matches bulk_output_*.csv."
        )
        uploaded = st.file_uploader("CSV", type="csv", key="batch_csv")
        use_llm = st.checkbox(
            "Use LLM fallback for unresolved key/metric columns (needs auto_recon_llm.py + "
            "DEEPSEEK_API_KEY — silently no-ops if either is missing, same as the CLI)",
            key="batch_use_llm",
        )

        if uploaded and st.button("Run batch", type="primary", key="batch_run_btn"):
            content = uploaded.getvalue().decode("utf-8")
            reader = csv_mod.DictReader(io.StringIO(content))
            fieldnames_in = set(reader.fieldnames or [])
            missing = {"query_id", "raw_sql"} - fieldnames_in
            if missing:
                st.error(f"CSV is missing required column(s): {sorted(missing)}")
            else:
                rows = list(reader)
                results = []
                progress = st.progress(0, text=f"0/{len(rows)}")
                for i, row in enumerate(rows):
                    try:
                        results.append(process_row(row["query_id"], row["raw_sql"], use_llm))
                    except Exception as e:
                        results.append({
                            "query_id": row.get("query_id", ""),
                            "complexity": f"ERROR: {e}",
                            "draft_migrated_sql": "",
                            "warnings": str(e),
                            "auto_detected_keys": "",
                            "auto_detected_metrics": "",
                            "used_llm_fallback": False,
                            "recon_query": "",
                        })
                    progress.progress((i + 1) / len(rows), text=f"{i + 1}/{len(rows)}")
                st.session_state["batch_results"] = results

        results = st.session_state.get("batch_results")
        if results:
            skipped = sum(1 for r in results if str(r.get("recon_query", "")).startswith("-- SKIPPED"))
            llm_used = sum(1 for r in results if r.get("used_llm_fallback"))
            st.write(
                f"{len(results)} processed · {len(results) - skipped} recon auto-generated "
                f"({llm_used} via LLM fallback — review these) · {skipped} skipped (needs manual recon)"
            )
            st.dataframe(results, use_container_width=True, hide_index=True)

            fieldnames = ["query_id", "complexity", "draft_migrated_sql", "warnings",
                          "auto_detected_keys", "auto_detected_metrics", "used_llm_fallback", "recon_query"]
            buf = io.StringIO()
            writer = csv_mod.DictWriter(buf, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
            st.download_button(
                "Download bulk_output.csv", buf.getvalue(),
                file_name="bulk_output.csv", mime="text/csv", key="batch_download",
            )
            st.caption(
                "No per-row BQ dry-run here on purpose — 100+ dry-run calls in one batch risks "
                "quota/rate limits and would slow this down a lot. Use the Recon tab per query, "
                "or script bq_dry_run.estimate_bytes() yourself for the whole batch if you want it."
            )

# ---------------------------------------------------------------------------
# TAB 4: Known Mappings — read (all sections) + write (TABLE_MAP, optionally
# paired with one MANDATORY_FILTERS entry). Every write is append-only,
# backed up, and validated by re-import with automatic rollback — see
# mapping_writer.py's docstring for exactly why each of those exists.
# ---------------------------------------------------------------------------
with tab_mappings:
    st.caption(f"Source of truth: `{MAPPING_CONFIG_PATH}`. If you edit it by hand elsewhere, "
               f"restart the app to pick up the change.")

    st.subheader("Table map")
    st.dataframe(
        [{"old": k, "new": v["new"], "source_type": v["source_type"]}
         for k, v in mapping_config.TABLE_MAP.items()],
        use_container_width=True, hide_index=True,
    )
    st.subheader("Mandatory consolidation filters")
    st.dataframe(
        [{"new_table": k[0], "source_type": k[1], "filter": v}
         for k, v in mapping_config.MANDATORY_FILTERS.items()],
        use_container_width=True, hide_index=True,
    )
    st.subheader("Column map (auto-renamed)")
    st.dataframe(
        [{"old": k, "new": v} for k, v in mapping_config.COLUMN_MAP.items()],
        use_container_width=True, hide_index=True,
    )
    st.subheader("Always-flagged columns (never auto-renamed)")
    st.dataframe(
        [{"column": k, "why": v} for k, v in mapping_config.ALWAYS_FLAG_COLUMNS.items()],
        use_container_width=True, hide_index=True,
    )

    st.divider()
    st.subheader("Add a confirmed mapping")

    if not ENABLE_MAPPING_WRITE:
        st.info(
            "Mapping edits are made on the maintainer's local copy, not here — this hosted "
            "instance is read + Input/Recon/Batch only. Found a mapping that needs adding? "
            "Flag it to Ammar directly rather than waiting on this form."
        )
    else:
        st.warning(
            "This writes directly to mapping_config.py — the project's single source of truth, "
            "imported by every other script in this folder. A timestamped backup is made first, "
            "and the write is rolled back automatically if the file fails to re-import afterward. "
            "None of that protects against adding a mapping you haven't actually verified — that "
            "part is still on you."
        )

        with st.form("add_mapping_form"):
            old_key = st.text_input("Old table key (e.g. l2alpha.some_table)")
            new_table = st.text_input("New table (e.g. dwh.fact_something)")
            source_type = st.text_input("source_type (leave blank for None)")
            add_filter = st.checkbox("This table also needs a MANDATORY_FILTERS entry")
            filter_sql = st.text_input("Filter SQL (e.g. current_flag = 1)")
            note = st.text_input("Note (who/when/why confirmed — goes into the file as a comment)")
            confirm = st.checkbox("I've verified this mapping myself (schema + sample data) — not guessing")
            submitted = st.form_submit_button("Add to mapping_config.py")

        if submitted:
            if not (old_key.strip() and new_table.strip() and note.strip()):
                st.error("Old table key, new table, and note are all required.")
            elif not confirm:
                st.error("Check the confirmation box — this file is the whole team's source of truth.")
            elif add_filter and not filter_sql.strip():
                st.error("You checked 'needs a MANDATORY_FILTERS entry' but left the filter SQL blank.")
            else:
                try:
                    backup = add_table_mapping(
                        MAPPING_CONFIG_PATH,
                        old_key.strip(), new_table.strip(), source_type.strip() or None, note.strip(),
                        mandatory_filter_sql=filter_sql.strip() if add_filter else None,
                    )
                    st.success(f"Added. Backup saved at `{backup}`. Commit this in git if the repo "
                               f"is version-controlled — right now that backup file is the only history.")
                    st.rerun()
                except MappingWriteError as e:
                    st.error(str(e))
