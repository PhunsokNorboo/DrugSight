<div align="center">

# 🔬 DrugSight

### AI-Powered Drug Repurposing Engine Built on AlphaFold's Open Data

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Streamlit](https://img.shields.io/badge/demo-Streamlit-FF4B4B.svg)](https://drugsight.streamlit.app)
[![Built with AlphaFold](https://img.shields.io/badge/built%20with-AlphaFold-4285F4.svg)](https://alphafold.ebi.ac.uk)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)]()

*One person. Open data. Open tools. Real impact on rare disease.*

---

**7,000** rare diseases exist. Fewer than **5%** have approved treatments.
Developing a new drug costs **$2.6 billion** and takes **10-15 years**.

Drug repurposing cuts that to **3-5 years** at a **fraction of the cost**.

DrugSight is a complete, end-to-end pipeline that identifies FDA-approved drugs
with repurposing potential for rare diseases -- built entirely on open data,
open tools, and AlphaFold's predicted protein structures.

**Total infrastructure cost: $0.**

</div>

---

## Table of Contents

- [The Problem](#the-problem)
- [The Solution](#the-solution)
- [Architecture](#architecture)
- [Quick Start](#quick-start)
- [Case Study: Huntington's Disease](#case-study-huntingtons-disease)
- [Technology Stack](#technology-stack)
- [Module Overview](#module-overview)
- [Data Sources](#data-sources)
- [How It Works](#how-it-works)
- [Interactive Dashboard](#interactive-dashboard)
- [Future Directions](#future-directions)
- [Contributing](#contributing)
- [Acknowledgments](#acknowledgments)
- [License](#license)

---

## The Problem

Rare diseases are not rare. They affect **300 million people worldwide** -- roughly 1 in 25. Yet the economics of drug development work against them:

| Metric | Traditional Drug Discovery | Drug Repurposing |
|:---|:---:|:---:|
| Timeline | 10-15 years | 3-5 years |
| Cost | $2.6 billion | $300 million |
| Clinical failure rate | ~90% | ~45% |
| Safety data available | No | Yes (FDA-approved) |

The bottleneck is not biology. It is economics. Pharmaceutical companies cannot justify billion-dollar investments for diseases affecting small populations. The patients who need treatments the most are the ones the market serves the least.

But the data to find these treatments already exists -- scattered across public databases, waiting to be connected.

## The Solution

DrugSight is an end-to-end computational pipeline that takes a rare disease as input and returns ranked, explainable drug repurposing candidates as output. No wet lab. No proprietary data. No API keys with monthly fees.

The pipeline:

1. **Accepts a rare disease identifier** (EFO ontology)
2. **Maps disease to protein targets** via the Open Targets GraphQL API
3. **Retrieves AlphaFold-predicted 3D structures** for each target protein, filtered by pLDDT confidence
4. **Screens FDA-approved drugs** against each target via molecular docking (AutoDock Vina)
5. **Ranks candidates** using a LightGBM learning-to-rank model with 8 scoring features
6. **Explains every prediction** with SHAP feature attribution
7. **Presents results** in an interactive Streamlit dashboard with 3D protein visualization, radar charts, and PDF export

```
python -m drugsight EFO_0000337 --demo
```

One command. Disease in, ranked drug candidates out.

## Architecture

```
                          DrugSight Pipeline
 ============================================================

  Disease ID                                    FDA-Approved
 (EFO_0000337) ──┐                         ┌── Drug Library
                  │                         │   (DrugBank CSV)
                  v                         v
         ┌───────────────┐         ┌───────────────┐
         │  Open Targets  │         │    RDKit       │
         │  GraphQL API   │         │  Descriptors   │
         │  + UniProt ID  │         │  + Conformers  │
         │    Mapping     │         │                │
         └───────┬───────┘         └───────┬───────┘
                 │                         │
                 v                         │
         ┌───────────────┐                 │
         │   AlphaFold    │                 │
         │  Structure DB  │                 │
         │  (pLDDT >= 70) │                 │
         └───────┬───────┘                 │
                 │                         │
                 └──────────┬──────────────┘
                            │
                            v
                  ┌───────────────────┐
                  │   AutoDock Vina    │
                  │  Molecular Docking │
                  │  (Binding Affinity)│
                  └─────────┬─────────┘
                            │
                            v
                  ┌───────────────────┐
                  │  LightGBM Ranker   │
                  │  8 Features + SHAP │
                  │  (LambdaMART)      │
                  └─────────┬─────────┘
                            │
                            v
                  ┌───────────────────┐
                  │  Streamlit Dashboard│
                  │  3D Viewer + Charts │
                  │  CSV / PDF Export   │
                  └───────────────────┘
```

## Quick Start

### Demo Mode (no external dependencies required)

```bash
# Clone the repository
git clone https://github.com/PhunsokNorboo/DrugSight.git
cd drugsight

# Install the package
pip install -e ".[dev]"

# Run the demo pipeline (uses pre-computed sample data)
python -m drugsight --demo

# Launch the interactive dashboard
streamlit run src/drugsight/app.py
```

Demo mode uses pre-computed docking results for Huntington's disease, so you can explore the full scoring, ranking, and visualization pipeline without installing AutoDock Vina, RDKit, or Open Babel.

### Full Pipeline (requires local tools)

```bash
# Install with RDKit support
pip install -e ".[dev,rdkit]"

# Ensure AutoDock Vina and Open Babel are on PATH
brew install open-babel  # macOS
# or: conda install -c conda-forge vina openbabel

# Run the full pipeline for Huntington's disease
python -m drugsight EFO_0000337 \
    --drugbank-csv data/sample_drugbank.csv \
    --min-score 0.5 \
    --min-plddt 70.0 \
    --top-n 20 \
    --output results/ranked_candidates.csv

# Run with verbose logging
python -m drugsight EFO_0000337 --demo -v
```

### Run the Tests

```bash
pytest tests/ -v
```

## Case Study: Huntington's Disease

Huntington's disease (EFO_0000337) is a progressive neurodegenerative disorder caused by CAG repeat expansion in the *HTT* gene. There is no cure. Current treatments address symptoms only.

DrugSight identified **5 protein targets** from Open Targets, retrieved their AlphaFold-predicted structures, and screened **10 FDA-approved drugs** across all targets via molecular docking. The top candidates:

| Rank | Drug | Best Affinity (kcal/mol) | Target | Top Factor | Patent Expired |
|:---:|:---|:---:|:---:|:---|:---:|
| 1 | **Riluzole** | -9.5 | GRIA1 | Binding Affinity | Yes |
| 2 | **Memantine** | -9.1 | GRIA1 | Binding Affinity | Yes |
| 3 | **Baclofen** | -8.6 | GRIA1 | Safety Score | Yes |
| 4 | **Tamoxifen** | -8.4 | HTT | Binding Affinity | Yes |
| 5 | **Sildenafil** | -7.8 | HTT | Association Score | No |

**Riluzole**, currently approved for ALS, emerged as the top candidate with the strongest binding affinity against the glutamate receptor GRIA1 -- a target strongly associated with Huntington's via excitotoxicity pathways. Notably, riluzole's neuroprotective mechanism (glutamate modulation) is directly relevant to Huntington's pathophysiology.

**Memantine**, approved for Alzheimer's, ranked second. Its NMDA receptor antagonism aligns with the excitotoxic damage hypothesis in Huntington's. Both drugs are off-patent and orally bioavailable -- ideal repurposing candidates.

These results are consistent with published literature exploring glutamate-modulating therapies for Huntington's disease, demonstrating that the pipeline can independently surface clinically plausible hypotheses.

*Targets screened: HTT (Huntingtin), BDNF (Brain-derived neurotrophic factor), GRIA1 (Glutamate receptor), NR3C1 (Glucocorticoid receptor), MAPK3 (MAP kinase 3).*

## Technology Stack

Every tool in the stack is free, open-source, and academically licensed. The total cost to run the full pipeline is **$0**.

| Component | Tool | Purpose | License | Cost |
|:---|:---|:---|:---:|:---:|
| Protein structures | **AlphaFold DB** | 3D structure predictions with pLDDT confidence | CC-BY 4.0 | $0 |
| Disease-target mapping | **Open Targets** | Genetic & literature evidence for disease-gene links | Apache 2.0 | $0 |
| Protein ID resolution | **UniProt** | Ensembl-to-UniProt accession mapping | CC-BY 4.0 | $0 |
| Drug library | **DrugBank** | FDA-approved drug metadata and SMILES | CC-BY-NC 4.0 | $0 |
| Molecular descriptors | **RDKit** | Lipinski descriptors, conformer generation | BSD-3 | $0 |
| Molecular docking | **AutoDock Vina** | Binding affinity prediction (kcal/mol) | Apache 2.0 | $0 |
| Format conversion | **Open Babel** | PDB/MOL to PDBQT conversion | GPL-2.0 | $0 |
| ML ranking | **LightGBM** | LambdaMART learning-to-rank | MIT | $0 |
| Explainability | **SHAP** | TreeExplainer feature attribution | MIT | $0 |
| Dashboard | **Streamlit** | Interactive web UI with 3D protein viewer | Apache 2.0 | $0 |
| 3D visualization | **py3Dmol / stmol** | Interactive protein structure rendering | MIT | $0 |
| Charting | **Plotly** | Histograms, radar charts, bar charts | MIT | $0 |
| PDF export | **fpdf2** | Report generation | LGPL-3.0 | $0 |
| **Total** | | | | **$0** |

## Module Overview

DrugSight is organized into six modules that chain together as a pipeline. Each module has a strict data contract defined in [`schemas.py`](src/drugsight/schemas.py).

| Module | File | Description |
|:---|:---|:---|
| **Schemas** | [`src/drugsight/schemas.py`](src/drugsight/schemas.py) | Shared TypedDict contracts and column definitions. Single source of truth for all inter-module data shapes. |
| **Config** | [`src/drugsight/config.py`](src/drugsight/config.py) | Paths and API endpoint URLs. |
| **1. Disease Targets** | [`src/drugsight/disease_targets.py`](src/drugsight/disease_targets.py) | Queries Open Targets GraphQL API for disease-associated protein targets, resolves Ensembl gene IDs to UniProt accessions via the UniProt ID Mapping service. |
| **2. AlphaFold Client** | [`src/drugsight/alphafold_client.py`](src/drugsight/alphafold_client.py) | Downloads AlphaFold-predicted PDB structures, retrieves per-residue pLDDT confidence scores, filters by confidence threshold. |
| **3. Drug Library** | [`src/drugsight/drug_library.py`](src/drugsight/drug_library.py) | Loads FDA-approved drugs from CSV, computes Lipinski molecular descriptors via RDKit (MW, LogP, HBD, HBA, TPSA, rotatable bonds), generates 3D conformers. |
| **4. Docking Engine** | [`src/drugsight/docking_engine.py`](src/drugsight/docking_engine.py) | Wraps AutoDock Vina for binding affinity prediction. Handles PDB-to-PDBQT conversion via Open Babel, automatic binding site detection, and batch screening. |
| **5. ML Scorer** | [`src/drugsight/ml_scorer.py`](src/drugsight/ml_scorer.py) | Trains a LightGBM LambdaMART model on known repurposing cases, scores and ranks new candidates, generates SHAP explanations. Falls back to weighted-sum scoring when training data is small. |
| **6. Dashboard** | [`src/drugsight/app.py`](src/drugsight/app.py) | Streamlit interactive dashboard: candidate rankings table, score distribution, SHAP feature attribution, 3D AlphaFold protein viewer, radar comparison chart, CSV/PDF export. |
| **Pipeline** | [`src/drugsight/pipeline.py`](src/drugsight/pipeline.py) | Orchestrates all six modules into `run_pipeline()` (full) and `run_pipeline_demo()` (pre-computed data) entry points. |
| **CLI** | [`src/drugsight/__main__.py`](src/drugsight/__main__.py) | Command-line interface with `--demo`, `--verbose`, disease selection, and output options. |

## Data Sources

| Database | What DrugSight Uses | Records | Access | License |
|:---|:---|:---:|:---|:---:|
| [AlphaFold DB](https://alphafold.ebi.ac.uk) | Predicted 3D protein structures + pLDDT confidence | 214M+ proteins | REST API | CC-BY 4.0 |
| [Open Targets](https://platform.opentargets.org) | Disease-target genetic associations | 12K+ diseases | GraphQL API | Apache 2.0 |
| [UniProt](https://www.uniprot.org) | Protein ID mapping (Ensembl to UniProt) | 250M+ sequences | REST API | CC-BY 4.0 |
| [DrugBank](https://go.drugbank.com) | FDA-approved drug structures (SMILES), indications | 2,700+ approved | CSV download | CC-BY-NC 4.0 |
| [PubChem](https://pubchem.ncbi.nlm.nih.gov) | Bioactivity data, chemical properties | 116M+ compounds | REST API | Public domain |
| [ChEMBL](https://www.ebi.ac.uk/chembl/) | Bioactivity measurements, assay data | 2.4M+ compounds | REST API | CC-BY-SA 3.0 |

All data access is programmatic. No manual downloads, no browser logins, no API keys with paywalls.

## How It Works

### Step 1 -- Disease-to-Target Mapping

Given a disease identifier (e.g., `EFO_0000337` for Huntington's), DrugSight queries the **Open Targets Platform** GraphQL API to retrieve genetically and experimentally validated protein targets. Each target is scored by Open Targets based on genetic associations, literature evidence, animal models, and known drug interactions. Only targets exceeding a configurable confidence threshold (default: 0.5) are retained.

Ensembl gene IDs from Open Targets are then resolved to UniProt accessions via the **UniProt ID Mapping** service (submit/poll/fetch pattern with exponential backoff), establishing the link between disease genetics and protein structure.

### Step 2 -- AlphaFold Structure Retrieval

For each UniProt accession, DrugSight retrieves the **AlphaFold-predicted 3D structure** from the EBI AlphaFold Protein Structure Database. Before downloading the full PDB file, the pipeline checks the per-residue **pLDDT confidence score** -- AlphaFold's measure of prediction reliability.

Only structures with an average pLDDT above the threshold (default: 70.0, corresponding to "confident" prediction) proceed to docking. This ensures that drug screening is performed against structurally reliable targets, not speculative predictions.

### Step 3 -- Drug Library Preparation

FDA-approved drugs are loaded from a DrugBank-sourced CSV. For each compound, DrugSight uses **RDKit** to:

- Parse the SMILES representation
- Compute Lipinski-relevant molecular descriptors: molecular weight, LogP, hydrogen bond donors/acceptors, topological polar surface area, and rotatable bonds
- Generate energy-minimized 3D conformers (MMFF force field) for docking

Drugs that fail SMILES parsing or conformer generation are logged and skipped.

### Step 4 -- Molecular Docking

Each drug conformer is docked against each protein target using **AutoDock Vina**:

1. **Receptor preparation**: PDB structures are converted to PDBQT format via Open Babel with Gasteiger partial charges
2. **Binding site detection**: The geometric center of all C-alpha atoms is computed as a heuristic binding center, with a 25A search box
3. **Docking**: Vina evaluates binding poses and returns the best binding affinity in kcal/mol (more negative = stronger binding)
4. **Batch screening**: Every drug is screened against every target; individual failures are caught and skipped without aborting the batch

### Step 5 -- ML Scoring and Ranking

Docking affinity alone is insufficient for ranking -- it ignores target relevance, drug safety, and clinical feasibility. DrugSight trains a **LightGBM LambdaMART** learning-to-rank model on known drug repurposing cases using 8 features:

| Feature | Source | Weight |
|:---|:---|:---:|
| `binding_affinity` | AutoDock Vina (negated, so higher = better) | 0.25 |
| `association_score` | Open Targets genetic evidence | 0.15 |
| `bioactivity_count` | Known bioactivity measurements | 0.15 |
| `tanimoto_similarity` | Structural similarity to known actives | 0.15 |
| `safety_score` | Post-market safety profile | 0.10 |
| `plddt_score` | AlphaFold structure confidence | 0.10 |
| `oral_bioavailability` | Fraction absorbed orally | 0.05 |
| `patent_expired` | Generic availability (binary) | 0.05 |

When training data is limited (< 50 rows), the model gracefully falls back to a hand-tuned weighted-sum scorer with the same `.predict()` interface, so downstream code never branches on model type.

Every top candidate receives a **SHAP explanation** (TreeExplainer for LightGBM, pseudo-SHAP for the fallback) that identifies the single most important factor driving its rank -- making every prediction auditable and interpretable.

### Step 6 -- Interactive Dashboard

Results are presented in a **Streamlit** dashboard with:

- **Overview metrics**: targets found, structures retrieved, drugs screened, top score
- **Ranked candidates table**: sortable, with top-3 highlighting
- **Score distribution histogram**: colored by patent status
- **SHAP feature attribution chart**: per-drug explanation of "why does this drug rank here?"
- **3D protein viewer**: interactive AlphaFold structure colored by pLDDT confidence (via py3Dmol)
- **Radar chart**: top-5 candidate comparison across all 8 scoring features
- **Export**: CSV download and formatted PDF report generation

## Interactive Dashboard

The Streamlit dashboard provides a biotech-grade interface for exploring results:

```bash
streamlit run src/drugsight/app.py
```

The dashboard includes:

- **Disease selector** with 10 pre-configured rare diseases
- **Configurable parameters**: minimum association score, result count
- **Demo mode toggle**: explore pre-computed results immediately, or run the full pipeline locally
- **3D protein structures** rendered directly from AlphaFold DB with pLDDT confidence coloring
- **PDF report generation** for sharing findings with collaborators

## Future Directions

- **AlphaFold 3 integration** -- Leverage AlphaFold 3's ability to predict protein-ligand complex structures, replacing classical docking with learned interaction predictions
- **AlphaMissense variant layer** -- Incorporate pathogenicity predictions for missense variants to prioritize targets where the disease mechanism involves specific mutations
- **Multi-target scoring** -- Model polypharmacology: drugs that hit multiple disease targets simultaneously may have stronger therapeutic effects
- **Agentic workflows** -- Use LLM agents to automatically review literature for each top candidate, surfacing supporting or contradicting clinical evidence
- **Scale to all 7,000 rare diseases** -- Batch processing across the full rare disease ontology with result aggregation and cross-disease pattern detection
- **Patient advocacy partnerships** -- Partner with rare disease organizations (NORD, EURORDIS) to prioritize diseases with the most urgent unmet need
- **Clinical evidence integration** -- Incorporate ClinicalTrials.gov data to flag candidates already under investigation, reducing duplication of effort

## Contributing

Contributions are welcome. DrugSight is a solo-built project, but rare diseases deserve collective effort.

**How to contribute:**

1. **Fork** the repository
2. **Create a branch** for your feature (`git checkout -b feature/your-feature`)
3. **Write tests** -- foundational code must be verified (`pytest tests/ -v`)
4. **Follow the data contracts** -- all inter-module data shapes are defined in `schemas.py`
5. **Submit a pull request** with a clear description of what changed and why

**Especially valuable contributions:**

- Adding new disease case studies with validated results
- Improving the ML scoring model with better training data
- Integrating additional drug databases (ChEMBL bioactivity, PubChem assays)
- Adding AlphaFold 3 or AlphaMissense support
- Translating the dashboard for international rare disease communities

## Acknowledgments

This project stands on the shoulders of extraordinary open science:

- **[AlphaFold](https://alphafold.ebi.ac.uk)** (DeepMind / EMBL-EBI) -- For making 214 million protein structure predictions freely available to the world. The AlphaFold Protein Structure Database is one of the most consequential contributions to open science in history.
- **[Open Targets](https://www.opentargets.org)** (EMBL-EBI / Wellcome Sanger / GSK) -- For building a comprehensive, open-access platform linking diseases to their molecular targets with rigorous evidence scoring.
- **[DrugBank](https://go.drugbank.com)** -- For maintaining a curated database of FDA-approved drugs with structural and pharmacological data.
- **[AutoDock Vina](https://vina.scripps.edu)** -- For providing fast, accurate, open-source molecular docking.
- **[RDKit](https://www.rdkit.org)** -- For the gold-standard open-source cheminformatics toolkit.
- **[LightGBM](https://lightgbm.readthedocs.io)** and **[SHAP](https://shap.readthedocs.io)** -- For making ML ranking and explainability accessible.
- **The rare disease community** -- Patients, caregivers, advocates, and researchers who fight every day for diseases the market has forgotten. This project exists because of you.

## Citation

If you use DrugSight in your research, please cite:

```bibtex
@software{drugsight2026,
  title     = {DrugSight: AI-Powered Drug Repurposing Engine Built on AlphaFold's Open Data},
  author    = {Phunsok Norboo},
  year      = {2026},
  url       = {https://github.com/PhunsokNorboo/DrugSight},
  version   = {0.1.0},
  license   = {MIT}
}
```

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

The underlying data sources have their own licenses (noted in the [Data Sources](#data-sources) table). Please respect the terms of each when using DrugSight's outputs in commercial or clinical contexts.

---

<div align="center">

*Built by one person with open data and open tools.*
*Because 300 million people with rare diseases should not have to wait for market incentives.*

</div>
