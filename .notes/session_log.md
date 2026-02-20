# DrugSight Session Log

## 2026-02-19 — Project Kickoff
- Created project scaffolding with interface-first architecture
- Decomposed into 17 beads across 5 layers (peak parallelism: 8)
- `schemas.py` defines all inter-module data contracts
- Demis Hassabis context: at hotel, project must impress

### Layer 0 — Foundation (completed)
- pyproject.toml, schemas.py, config.py, conftest.py, .gitignore
- 4 sample data files (20 FDA drugs, 5 HD targets, 50 docking results, 100 training cases)
- Git init + first commit

### Layer 1 — Core Modules (8 parallel agents, all completed)
- disease_targets.py — Open Targets GraphQL + UniProt ID mapping
- alphafold_client.py — PDB download + pLDDT confidence filtering
- drug_library.py — RDKit descriptors + 3D conformers
- docking_engine.py — AutoDock Vina subprocess wrapper
- ml_scorer.py — LightGBM LambdaMART + SHAP + WeightedSumScorer fallback
- app.py — Streamlit dashboard (992 lines, py3Dmol, Plotly, PDF export)
- Unit tests for modules 1-4 (18 tests)
- Unit tests for ML scorer (25 tests)

### Layer 2 — Integration (completed)
- pipeline.py — end-to-end orchestrator with demo mode
- huntingtons_case_study.ipynb — 26-cell notebook with Plotly visualizations
- test_pipeline.py (14 tests) + test_integration.py (14 tests)

### Layer 3 — Polish (completed)
- __main__.py — CLI entry point (`python -m drugsight --demo`)
- .streamlit/config.toml + requirements.txt + packages.txt
- README.md — comprehensive docs with architecture diagram

### Layer 4 — Smoke Test (completed)
- 67 passed, 5 skipped (RDKit not installed)
- CLI demo mode runs end-to-end, outputs ranked_candidates.csv
- Fixed pyproject.toml build-backend (hatchling.build, not hatchling.backends)

### Corrections
| Issue | Fix |
|-------|-----|
| pyproject.toml `build-backend = "hatchling.backends"` | Changed to `"hatchling.build"` |
| `test_scores_are_valid` failed (negative LambdaMART scores) | Changed assertion to `nunique() > 1` |
| `test_batch_dock_handles_failures` (wrong binary path) | Used `side_effect` to return correct path per binary name |

### All 17 beads closed, `bd sync` complete.
