"""
AlphaFold Structure Retrieval — Module 2 of DrugSight.

Downloads AlphaFold-predicted protein structures (PDB files) and retrieves
per-residue pLDDT confidence scores for disease-linked proteins identified
by Module 1 (Open Targets).

All HTTP calls are lazy-loaded, retried with exponential backoff, and
resilient to missing predictions (404 -> None + logged warning).
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING

import requests

from drugsight.config import ALPHAFOLD_API_URL, STRUCTURES_DIR
from drugsight.schemas import AlphaFoldConfidence

if TYPE_CHECKING:
    from requests import Session

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────
_MODEL_TEMPLATE = "AF-{uniprot_id}-F1-model_v4.pdb"
_PDB_URL_TEMPLATE = f"{ALPHAFOLD_API_URL}/files/{{filename}}"
_PREDICTION_URL_TEMPLATE = f"{ALPHAFOLD_API_URL}/api/prediction/{{uniprot_id}}"

_MAX_RETRIES = 3
_BACKOFF_BASE = 1.0  # seconds; delays will be 1s, 2s, 4s

# ── Lazy session ─────────────────────────────────────────────────────────
_session: Session | None = None


def _get_session() -> Session:
    """Return a shared requests.Session, created on first call."""
    global _session
    if _session is None:
        _session = requests.Session()
        _session.headers.update(
            {"Accept": "application/json", "User-Agent": "DrugSight/0.1.0"}
        )
    return _session


# ── Retry helper ─────────────────────────────────────────────────────────

def _request_with_retry(
    method: str,
    url: str,
    *,
    max_retries: int = _MAX_RETRIES,
    backoff_base: float = _BACKOFF_BASE,
    **kwargs: object,
) -> requests.Response | None:
    """Execute an HTTP request with exponential-backoff retries.

    Returns the Response on success (2xx), ``None`` on a 404, and raises
    after exhausting retries for other failures.
    """
    session = _get_session()

    last_exc: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = session.request(method, url, **kwargs)  # type: ignore[arg-type]

            if resp.status_code == 404:
                return None

            resp.raise_for_status()
            return resp

        except requests.exceptions.HTTPError as exc:
            last_exc = exc
            # 4xx (other than 404) are not retryable
            if resp.status_code < 500:  # type: ignore[possibly-undefined]
                raise
        except requests.exceptions.RequestException as exc:
            last_exc = exc

        if attempt < max_retries:
            delay = backoff_base * (2 ** (attempt - 1))
            logger.warning(
                "Request to %s failed (attempt %d/%d), retrying in %.1fs: %s",
                url,
                attempt,
                max_retries,
                delay,
                last_exc,
            )
            time.sleep(delay)

    logger.error(
        "Request to %s failed after %d attempts: %s", url, max_retries, last_exc
    )
    raise requests.exceptions.RetryError(  # type: ignore[call-arg]
        f"Failed after {max_retries} retries: {last_exc}"
    ) from last_exc


# ── Public API ───────────────────────────────────────────────────────────

def fetch_structure(
    uniprot_id: str,
    output_dir: Path | None = None,
) -> Path | None:
    """Download an AlphaFold PDB structure for *uniprot_id*.

    Parameters
    ----------
    uniprot_id:
        UniProt accession (e.g. ``"P04637"``).
    output_dir:
        Directory to save the PDB file. Defaults to ``config.STRUCTURES_DIR``.

    Returns
    -------
    Path to the downloaded PDB file, or ``None`` if AlphaFold has no
    prediction for this protein.
    """
    if output_dir is None:
        output_dir = STRUCTURES_DIR

    filename = _MODEL_TEMPLATE.format(uniprot_id=uniprot_id)
    dest = output_dir / filename

    # Cache: skip download if file already exists
    if dest.exists():
        logger.debug("Structure already cached: %s", dest)
        return dest

    url = _PDB_URL_TEMPLATE.format(filename=filename)
    logger.info("Downloading AlphaFold structure for %s from %s", uniprot_id, url)

    resp = _request_with_retry("GET", url, stream=True)
    if resp is None:
        logger.warning(
            "No AlphaFold prediction found for %s (404)", uniprot_id
        )
        return None

    # Ensure the output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Stream to disk to handle large structures gracefully
    with open(dest, "wb") as fh:
        for chunk in resp.iter_content(chunk_size=65_536):
            fh.write(chunk)

    logger.info("Saved structure to %s", dest)
    return dest


def get_confidence(uniprot_id: str) -> AlphaFoldConfidence | None:
    """Retrieve pLDDT confidence metadata for *uniprot_id*.

    Parameters
    ----------
    uniprot_id:
        UniProt accession (e.g. ``"P04637"``).

    Returns
    -------
    An :class:`AlphaFoldConfidence` dict with ``uniprot_id``,
    ``avg_plddt``, ``model_url``, and ``version``; or ``None`` if AlphaFold
    has no prediction for this protein.
    """
    url = _PREDICTION_URL_TEMPLATE.format(uniprot_id=uniprot_id)
    logger.info("Fetching AlphaFold confidence for %s", uniprot_id)

    resp = _request_with_retry("GET", url)
    if resp is None:
        logger.warning(
            "No AlphaFold prediction metadata for %s (404)", uniprot_id
        )
        return None

    payload = resp.json()

    # The API returns a JSON array; the first element holds the prediction.
    if not isinstance(payload, list) or len(payload) == 0:
        logger.warning(
            "Unexpected empty response from AlphaFold API for %s", uniprot_id
        )
        return None

    entry = payload[0]

    confidence: AlphaFoldConfidence = {
        "uniprot_id": uniprot_id,
        "avg_plddt": float(entry["confidenceAvgLocalScore"]),
        "model_url": str(entry["pdbUrl"]),
        "version": int(entry["latestVersion"]),
    }

    logger.debug(
        "Confidence for %s: avg_plddt=%.1f, version=%d",
        uniprot_id,
        confidence["avg_plddt"],
        confidence["version"],
    )
    return confidence


def fetch_structures_batch(
    uniprot_ids: list[str],
    output_dir: Path | None = None,
    min_plddt: float = 70.0,
) -> tuple[list[Path], list[AlphaFoldConfidence]]:
    """Fetch AlphaFold structures for multiple proteins, filtered by confidence.

    Only proteins whose average pLDDT score meets *min_plddt* are included
    in the output.  Proteins that lack an AlphaFold prediction or fall below
    the threshold are logged and skipped.

    Parameters
    ----------
    uniprot_ids:
        List of UniProt accessions.
    output_dir:
        Directory to save PDB files. Defaults to ``config.STRUCTURES_DIR``.
    min_plddt:
        Minimum average pLDDT score to accept a structure (default 70.0).

    Returns
    -------
    A tuple of ``(pdb_paths, confidence_records)`` for the proteins that
    passed the confidence filter.
    """
    pdb_paths: list[Path] = []
    confidences: list[AlphaFoldConfidence] = []

    total = len(uniprot_ids)
    logger.info(
        "Batch fetch: %d protein(s), min_plddt=%.1f", total, min_plddt
    )

    for idx, uid in enumerate(uniprot_ids, start=1):
        logger.info("Processing %s (%d/%d)", uid, idx, total)

        # --- Confidence check first (cheaper than downloading the PDB) ---
        conf = get_confidence(uid)
        if conf is None:
            logger.warning("Skipping %s — no AlphaFold prediction available", uid)
            continue

        if conf["avg_plddt"] < min_plddt:
            logger.warning(
                "Skipping %s — avg pLDDT %.1f is below threshold %.1f",
                uid,
                conf["avg_plddt"],
                min_plddt,
            )
            continue

        # --- Download structure ------------------------------------------------
        path = fetch_structure(uid, output_dir=output_dir)
        if path is None:
            logger.warning(
                "Skipping %s — structure download failed despite valid metadata",
                uid,
            )
            continue

        pdb_paths.append(path)
        confidences.append(conf)

    logger.info(
        "Batch complete: %d/%d structures retrieved (pLDDT >= %.1f)",
        len(pdb_paths),
        total,
        min_plddt,
    )
    return pdb_paths, confidences
