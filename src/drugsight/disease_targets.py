"""
Disease-to-Target Mapping via Open Targets and UniProt.

Given a rare disease EFO ID, retrieves associated protein targets from the
Open Targets Platform GraphQL API and resolves their Ensembl gene IDs to
UniProt accessions via the UniProt ID Mapping service.

Output conforms to the TargetRecord / TARGET_COLUMNS contract in schemas.py.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import pandas as pd

from drugsight.config import OPEN_TARGETS_URL, UNIPROT_API_URL
from drugsight.schemas import TARGET_COLUMNS

logger = logging.getLogger(__name__)

# ── GraphQL query ────────────────────────────────────────────────────────
_DISEASE_TARGETS_QUERY = """\
query DiseaseTargets($diseaseId: String!) {
    disease(efoId: $diseaseId) {
        name
        associatedTargets(page: {size: 50, index: 0}) {
            rows {
                target { id approvedSymbol approvedName }
                score
            }
        }
    }
}
"""

# ── Retry configuration ─────────────────────────────────────────────────
_MAX_RETRIES = 3
_BACKOFF_SECONDS = (1, 2, 4)


# ── Internal helpers ─────────────────────────────────────────────────────


def _get_session() -> "requests.Session":
    """Lazy-create a requests session with sensible defaults."""
    import requests

    session = requests.Session()
    session.headers.update(
        {
            "Accept": "application/json",
        }
    )
    return session


def _request_with_retry(
    session: "requests.Session",
    method: str,
    url: str,
    **kwargs: Any,
) -> "requests.Response":
    """Execute an HTTP request with exponential-backoff retry on failure.

    Retries up to ``_MAX_RETRIES`` times for connection errors and 5xx
    server responses.  4xx errors are raised immediately — they indicate
    a client-side problem that retrying will not fix.
    """
    import requests

    last_exc: Exception | None = None

    for attempt in range(_MAX_RETRIES):
        try:
            resp = session.request(method, url, **kwargs)

            # Raise immediately on client errors (4xx) — retrying won't help.
            if 400 <= resp.status_code < 500:
                resp.raise_for_status()

            # Retry on server errors (5xx).
            if resp.status_code >= 500:
                logger.warning(
                    "Server error %d from %s (attempt %d/%d)",
                    resp.status_code,
                    url,
                    attempt + 1,
                    _MAX_RETRIES,
                )
                if attempt < _MAX_RETRIES - 1:
                    time.sleep(_BACKOFF_SECONDS[attempt])
                    continue
                resp.raise_for_status()

            return resp  # noqa: TRY300 — happy path

        except requests.ConnectionError as exc:
            last_exc = exc
            logger.warning(
                "Connection error for %s (attempt %d/%d): %s",
                url,
                attempt + 1,
                _MAX_RETRIES,
                exc,
            )
            if attempt < _MAX_RETRIES - 1:
                time.sleep(_BACKOFF_SECONDS[attempt])

    # All retries exhausted.
    raise last_exc  # type: ignore[misc]


# ── Open Targets ─────────────────────────────────────────────────────────


def _fetch_targets(
    disease_id: str,
    min_score: float,
    session: "requests.Session",
) -> list[dict[str, Any]]:
    """Query Open Targets GraphQL for targets associated with *disease_id*.

    Returns a list of dicts, each with keys ``ensembl_id``, ``symbol``,
    ``name``, and ``association_score``.  Only rows whose score meets or
    exceeds *min_score* are included.
    """
    logger.info("Querying Open Targets for disease %s (min_score=%.2f)", disease_id, min_score)

    payload = {
        "query": _DISEASE_TARGETS_QUERY,
        "variables": {"diseaseId": disease_id},
    }

    resp = _request_with_retry(session, "POST", OPEN_TARGETS_URL, json=payload)
    body = resp.json()

    # Drill into the nested GraphQL response.
    disease_data = body.get("data", {}).get("disease")
    if disease_data is None:
        raise ValueError(f"No targets found for {disease_id}")

    disease_name = disease_data.get("name", disease_id)
    rows = (
        disease_data
        .get("associatedTargets", {})
        .get("rows", [])
    )

    if not rows:
        raise ValueError(f"No targets found for {disease_id}")

    logger.info(
        "Open Targets returned %d target(s) for '%s'", len(rows), disease_name
    )

    targets: list[dict[str, Any]] = []
    for row in rows:
        score = row.get("score", 0.0)
        if score < min_score:
            continue

        target_info = row.get("target", {})
        targets.append(
            {
                "ensembl_id": target_info.get("id", ""),
                "symbol": target_info.get("approvedSymbol", ""),
                "name": target_info.get("approvedName", ""),
                "association_score": float(score),
            }
        )

    if not targets:
        raise ValueError(
            f"No targets found for {disease_id} with score >= {min_score}"
        )

    logger.info(
        "%d target(s) pass the min_score filter (%.2f)", len(targets), min_score
    )
    return targets


# ── UniProt ID Mapping ───────────────────────────────────────────────────

_POLL_INTERVAL_SECONDS = 1.0
_POLL_MAX_ATTEMPTS = 30  # generous ceiling: 30 s for a small batch


def _map_ensembl_to_uniprot(
    ensembl_ids: list[str],
    session: "requests.Session",
) -> dict[str, str]:
    """Resolve a list of Ensembl gene IDs to UniProt accessions.

    Uses the UniProt ID Mapping REST service (submit / poll / fetch).
    Returns a dict mapping each Ensembl ID to its best UniProt accession.
    IDs that cannot be mapped are silently omitted.
    """
    if not ensembl_ids:
        return {}

    logger.info("Submitting %d Ensembl ID(s) to UniProt ID Mapping", len(ensembl_ids))

    # 1. Submit the mapping job.
    submit_url = f"{UNIPROT_API_URL}/idmapping/run"
    submit_resp = _request_with_retry(
        session,
        "POST",
        submit_url,
        data={
            "from": "Ensembl",
            "to": "UniProtKB",
            "ids": ",".join(ensembl_ids),
        },
    )
    job_id = submit_resp.json()["jobId"]
    logger.info("UniProt mapping job submitted: %s", job_id)

    # 2. Poll until the job completes.
    status_url = f"{UNIPROT_API_URL}/idmapping/status/{job_id}"
    results_url: str | None = None

    for poll_attempt in range(_POLL_MAX_ATTEMPTS):
        poll_resp = _request_with_retry(session, "GET", status_url, allow_redirects=False)

        # A 303 redirect means results are ready at the Location header.
        if poll_resp.status_code == 303:
            results_url = poll_resp.headers.get("Location", "")
            break

        poll_body = poll_resp.json()

        # Some API versions return the results inline once complete.
        if "results" in poll_body:
            return _parse_mapping_results(poll_body, ensembl_ids)

        # Still running.
        job_status = poll_body.get("jobStatus", "RUNNING")
        if job_status == "FINISHED":
            # Redirect didn't fire; build the results URL ourselves.
            results_url = f"{UNIPROT_API_URL}/idmapping/uniprotkb/results/{job_id}"
            break

        if job_status not in {"RUNNING", "NEW", "QUEUED"}:
            raise RuntimeError(
                f"UniProt ID mapping job {job_id} failed with status: {job_status}"
            )

        logger.debug(
            "Polling UniProt job %s — status: %s (attempt %d)",
            job_id,
            job_status,
            poll_attempt + 1,
        )
        time.sleep(_POLL_INTERVAL_SECONDS)
    else:
        raise TimeoutError(
            f"UniProt ID mapping job {job_id} did not complete within "
            f"{_POLL_MAX_ATTEMPTS * _POLL_INTERVAL_SECONDS:.0f}s"
        )

    # 3. Fetch the mapping results.
    assert results_url is not None
    logger.info("Fetching mapping results from %s", results_url)
    results_resp = _request_with_retry(session, "GET", results_url)
    return _parse_mapping_results(results_resp.json(), ensembl_ids)


def _parse_mapping_results(
    body: dict[str, Any],
    ensembl_ids: list[str],
) -> dict[str, str]:
    """Extract a { ensembl_id: uniprot_accession } dict from UniProt's response.

    When an Ensembl ID maps to multiple UniProt entries, the first
    (highest-confidence) Swiss-Prot accession is preferred.  If none is
    Swiss-Prot, the first TrEMBL accession is used.
    """
    mapping: dict[str, str] = {}

    for result in body.get("results", []):
        from_id = result.get("from", "")
        to_entry = result.get("to", {})

        # The accession lives at 'to.primaryAccession' in the UniProtKB
        # results schema, or directly under 'to' if the response is a
        # simple string mapping.
        if isinstance(to_entry, dict):
            accession = to_entry.get("primaryAccession", "")
        else:
            accession = str(to_entry)

        if not accession or not from_id:
            continue

        # Keep the first mapping per Ensembl ID (UniProt returns
        # reviewed/Swiss-Prot entries before unreviewed/TrEMBL).
        if from_id not in mapping:
            mapping[from_id] = accession

    mapped_count = sum(1 for eid in ensembl_ids if eid in mapping)
    logger.info(
        "UniProt mapping complete: %d/%d IDs resolved",
        mapped_count,
        len(ensembl_ids),
    )
    return mapping


# ── Public API ───────────────────────────────────────────────────────────


def get_disease_targets(
    disease_id: str,
    min_score: float = 0.5,
) -> pd.DataFrame:
    """Return protein targets for a rare disease, enriched with UniProt IDs.

    Parameters
    ----------
    disease_id:
        An EFO identifier (e.g. ``"EFO_0000337"`` for Huntington disease).
    min_score:
        Minimum Open Targets association score to include a target.
        Defaults to ``0.5``.

    Returns
    -------
    pd.DataFrame
        Columns match ``TARGET_COLUMNS``: *ensembl_id*, *symbol*, *name*,
        *association_score*, *uniprot_id*.  Sorted by *association_score*
        descending.

    Raises
    ------
    ValueError
        If Open Targets returns no targets or none meet *min_score*.
    """
    session = _get_session()

    try:
        # Step 1 — Retrieve targets from Open Targets.
        targets = _fetch_targets(disease_id, min_score, session)

        # Step 2 — Map Ensembl IDs to UniProt accessions.
        ensembl_ids = [t["ensembl_id"] for t in targets if t["ensembl_id"]]
        uniprot_map = _map_ensembl_to_uniprot(ensembl_ids, session)

        # Step 3 — Merge UniProt IDs into the target records.
        for target in targets:
            target["uniprot_id"] = uniprot_map.get(target["ensembl_id"], "")

        # Step 4 — Build DataFrame conforming to the schema contract.
        df = pd.DataFrame(targets, columns=TARGET_COLUMNS)
        df = df.sort_values("association_score", ascending=False).reset_index(drop=True)

        logger.info(
            "Returning %d target(s) for %s (%d with UniProt IDs)",
            len(df),
            disease_id,
            df["uniprot_id"].astype(bool).sum(),
        )
        return df

    finally:
        session.close()
