"""
BigQuery dry-run cost estimator.

dry_run=True never executes the query and never returns result rows -- it's
free and only validates syntax + estimates bytes that WOULD be processed.
This exists to answer the audit's own documented gap: this project ran an
estimated 150-200GB/day of reconciliation queries with zero pre-flight cost
check. This does not reintroduce query EXECUTION into the app -- that stays
manual, per team discipline -- it only adds a cost estimate before you paste
the query into the console.

NOT TESTED AGAINST A LIVE BIGQUERY API -- this sandbox has no network path to
googleapis.com. Written directly against the documented google-cloud-bigquery
client interface, but run it once against a cheap known query yourself before
trusting it, same discipline the project already applies to llm_classify.py's
own untested-against-live-API DeepSeek call.

Never hardcodes a project ID -- billing_project is always supplied by the
caller (UI field / env var), defaulting to whatever `gcloud auth
application-default login` has set up if left blank.
"""
from __future__ import annotations


class DryRunError(Exception):
    pass


def estimate_bytes(sql: str, billing_project: str | None = None) -> dict:
    """Returns {'total_bytes_processed': int, 'total_bytes_gb': float,
    'referenced_tables': list[str]}.

    Raises DryRunError on an auth/connectivity failure, or on a genuine query
    error -- a query that fails dry-run would also fail for real, so that's
    itself useful signal, not just noise to swallow.
    """
    try:
        from google.cloud import bigquery
        from google.api_core.exceptions import GoogleAPIError
    except ImportError as e:
        raise DryRunError(
            "google-cloud-bigquery isn't installed. "
            "pip install google-cloud-bigquery --break-system-packages"
        ) from e

    try:
        client = bigquery.Client(project=billing_project) if billing_project else bigquery.Client()
        job_config = bigquery.QueryJobConfig(dry_run=True, use_query_cache=False)
        job = client.query(sql, job_config=job_config)
    except GoogleAPIError as e:
        raise DryRunError(f"BigQuery rejected this query at dry-run (it would also fail for real): {e}") from e
    except Exception as e:
        raise DryRunError(
            f"Could not reach BigQuery ({type(e).__name__}: {e}). Check "
            f"`gcloud auth application-default login` and that the billing "
            f"project (blank = your ADC default) is one you have query rights in."
        ) from e

    try:
        referenced = sorted(
            f"{t.project}.{t.dataset_id}.{t.table_id}" for t in (job.referenced_tables or [])
        )
    except Exception:
        referenced = []  # not fatal -- the byte estimate is the important number

    total_bytes = job.total_bytes_processed or 0
    return {
        "total_bytes_processed": total_bytes,
        "total_bytes_gb": round(total_bytes / (1024 ** 3), 3),
        "referenced_tables": referenced,
    }
