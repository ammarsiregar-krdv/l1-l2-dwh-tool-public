"""
Safe, append-only writer for mapping_config.py's TABLE_MAP (and optionally one
MANDATORY_FILTERS entry alongside it).

mapping_config.py is the project's single source of truth, imported at
runtime by migrate_query.py -- treat every write to it like production DML:
back up first, never touch an existing line, validate by re-importing before
declaring success, and roll back automatically if the import fails.

Why append-only, never regenerate:
    Large parts of this file's value are the inline comments (who confirmed
    what, when, why -- e.g. the dim_user_offline discriminator note, the SCD2
    current_flag rationale). Rebuilding TABLE_MAP from a Python dict and
    re-serializing it would silently delete every one of those comments.
    This writer only ever inserts new lines immediately before a dict's
    closing '}' -- every existing byte is left alone.

Why reload BOTH mapping_config and migrate_query:
    migrate_query.py does `from mapping_config import TABLE_MAP, ...` at ITS
    OWN import time -- that's a name binding, not a live reference. Reloading
    mapping_config alone updates mapping_config.TABLE_MAP, but migrate_query's
    already-bound TABLE_MAP name keeps pointing at the OLD dict object until
    migrate_query itself is reloaded too (which re-runs its `from
    mapping_config import ...` line against the now-fresh module). Skipping
    this step is a real, silent failure mode: the UI would claim success while
    migrate() keeps ignoring the new mapping until the process restarts.
"""
from __future__ import annotations

import importlib
import re
import shutil
import sys
import time
from pathlib import Path

TABLE_MAP_OPEN_RE = re.compile(r"^TABLE_MAP\s*=\s*\{\s*$", re.MULTILINE)
MANDATORY_FILTERS_OPEN_RE = re.compile(r"^MANDATORY_FILTERS\s*=\s*\{\s*$", re.MULTILINE)


class MappingWriteError(Exception):
    pass


def _find_dict_close_brace(text: str, open_match_end: int) -> int:
    """Given the index right after a dict literal's opening '{', walk forward
    tracking brace depth and return the index of the matching top-level '}'.

    Does not attempt to special-case '{'/'}' inside string literals or
    comments -- confirmed by inspection that no dict body in this file
    contains a literal brace inside a string or comment. If that ever
    changes, this will raise (run off the end of the file) rather than
    silently inserting in the wrong place.
    """
    depth = 1
    i = open_match_end
    n = len(text)
    while depth > 0:
        if i >= n:
            raise MappingWriteError(
                "Could not find the matching '}' -- file structure changed "
                "unexpectedly. Aborting rather than guessing where to insert."
            )
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
        i += 1
    return i - 1  # index of the matching closing '}'


def _pyquote(s: str) -> str:
    """Double-quoted Python string literal, matching this file's existing
    convention (repr() defaults to single quotes, which would both look
    inconsistent AND -- more importantly -- break the duplicate-key check
    below, since existing keys in the file are double-quoted)."""
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _key_already_present(text: str, start: int, end: int, key: str) -> bool:
    """Match the key as a dict key regardless of which quote style it was
    originally written with -- a naive single-format substring check missed
    every pre-existing double-quoted key in this file (confirmed by testing:
    it let 'dragon.visit_reports' silently pass as \"new\" even though it's
    already mapped). A duplicate key in a Python dict literal doesn't raise
    -- the later one silently wins -- so a missed duplicate here would have
    silently overridden a working mapping with no error and no warning."""
    body = text[start:end]
    pattern = re.compile(r"""(['"])""" + re.escape(key) + r"""\1\s*:""")
    return bool(pattern.search(body))


def _reload_chain() -> None:
    """Reload mapping_config, then migrate_query if it's already loaded, in
    that order. Raises whatever import/reload throws -- caller is responsible
    for rollback."""
    import mapping_config as _mc
    importlib.reload(_mc)
    if "migrate_query" in sys.modules:
        import migrate_query as _mq
        importlib.reload(_mq)


def add_table_mapping(
    mapping_config_path: str,
    old_table_key: str,
    new_table: str,
    source_type: str | None,
    note: str,
    mandatory_filter_sql: str | None = None,
) -> str:
    """Appends one TABLE_MAP entry (and, if given, one MANDATORY_FILTERS
    entry) just before each dict's closing brace. Returns the backup file
    path on success.

    Raises MappingWriteError, leaving the on-disk file untouched, if:
      - old_table_key already exists in TABLE_MAP (idempotency guarantee --
        clicking "Add" twice, or two people adding the same key, is a
        no-op-with-error, never a silent duplicate)
      - the edited file fails to import (auto-rolled-back before raising)
    """
    path = Path(mapping_config_path)
    text = path.read_text(encoding="utf-8")

    tm_open = TABLE_MAP_OPEN_RE.search(text)
    if not tm_open:
        raise MappingWriteError("Could not find 'TABLE_MAP = {' -- aborting before any write.")
    tm_close = _find_dict_close_brace(text, tm_open.end())

    if _key_already_present(text, tm_open.end(), tm_close, old_table_key):
        raise MappingWriteError(
            f"{old_table_key!r} is already in TABLE_MAP -- refusing to add a duplicate "
            f"(a second dict key would silently win with no error, silently overriding the "
            f"existing mapping). Edit the existing entry by hand if it needs to change."
        )

    today = time.strftime("%Y-%m-%d")
    source_type_literal = _pyquote(source_type) if source_type else "None"
    new_table_literal = _pyquote(new_table)
    key_literal = _pyquote(old_table_key)

    table_map_entry = (
        f"\n    # ADDED {today} via Streamlit UI -- {note}\n"
        f"    {key_literal}: {{\"new\": {new_table_literal}, \"source_type\": {source_type_literal}}},\n"
    )

    new_text = text[:tm_close] + table_map_entry + text[tm_close:]

    if mandatory_filter_sql:
        mf_open = MANDATORY_FILTERS_OPEN_RE.search(new_text)
        if not mf_open:
            raise MappingWriteError("Could not find 'MANDATORY_FILTERS = {' -- aborting before any write.")
        mf_close = _find_dict_close_brace(new_text, mf_open.end())
        filter_key_literal = f"({new_table_literal}, {source_type_literal})"
        filter_entry = (
            f"\n    # ADDED {today} via Streamlit UI -- {note}\n"
            f"    {filter_key_literal}: {_pyquote(mandatory_filter_sql)},\n"
        )
        new_text = new_text[:mf_close] + filter_entry + new_text[mf_close:]

    backup_path = path.with_name(path.name + f".bak.{int(time.time())}")
    shutil.copy2(path, backup_path)
    path.write_text(new_text, encoding="utf-8")

    try:
        _reload_chain()
    except Exception as e:
        shutil.copy2(backup_path, path)
        try:
            _reload_chain()  # restore in-memory state to match the restored file
        except Exception:
            pass  # best-effort; the on-disk file is already back to the good copy
        raise MappingWriteError(
            f"Edited file failed to import ({type(e).__name__}: {e}) -- automatically "
            f"restored mapping_config.py from backup. Nothing was saved. Fix the input "
            f"and try again."
        ) from e

    return str(backup_path)
