"""
DrugSight configuration — paths and API endpoints.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
STRUCTURES_DIR = PROJECT_ROOT / "structures"
CONFORMERS_DIR = PROJECT_ROOT / "conformers"
RESULTS_DIR = PROJECT_ROOT / "results"

# API endpoints
OPEN_TARGETS_URL = "https://api.platform.opentargets.org/api/v4/graphql"
ALPHAFOLD_API_URL = "https://alphafold.ebi.ac.uk"
UNIPROT_API_URL = "https://rest.uniprot.org"
