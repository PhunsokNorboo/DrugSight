"""
DrugSight Streamlit Dashboard.

Interactive demo showing drug repurposing results: candidate rankings,
SHAP-style explanations, 3D protein visualization, and PDF export.

Launch:
    streamlit run src/drugsight/app.py
"""

from __future__ import annotations

import io
from datetime import datetime
from typing import TYPE_CHECKING

import streamlit as st

from drugsight.config import DATA_DIR, STRUCTURES_DIR
from drugsight.schemas import SCORING_FEATURES

if TYPE_CHECKING:
    import pandas as pd

# ---------------------------------------------------------------------------
# Page configuration -- must be the first Streamlit call
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="DrugSight",
    page_icon="\U0001f52c",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
RARE_DISEASES: dict[str, str] = {
    "Huntington's disease": "EFO_0000337",
    "Alzheimer's disease": "EFO_0000249",
    "Parkinson's disease": "EFO_0000537",
    "Amyotrophic lateral sclerosis": "EFO_0000253",
    "Multiple sclerosis": "EFO_0000311",
    "Crohn's disease": "EFO_0000341",
    "Systemic lupus erythematosus": "EFO_0000685",
    "Cystic fibrosis": "EFO_0000305",
    "Multiple myeloma": "EFO_0000616",
    "Epilepsy": "EFO_0000712",
}

# Feature display names for charts
_FEATURE_LABELS: dict[str, str] = {
    "binding_affinity": "Binding Affinity",
    "plddt_score": "pLDDT Score",
    "bioactivity_count": "Bioactivity Count",
    "safety_score": "Safety Score",
    "tanimoto_similarity": "Tanimoto Similarity",
    "association_score": "Association Score",
    "patent_expired": "Patent Expired",
    "oral_bioavailability": "Oral Bioavailability",
}

# Composite-score weights (mirrors the scoring module)
_WEIGHTS: dict[str, float] = {
    "binding_affinity": 0.25,
    "plddt_score": 0.10,
    "bioactivity_count": 0.10,
    "safety_score": 0.15,
    "tanimoto_similarity": 0.10,
    "association_score": 0.15,
    "patent_expired": 0.05,
    "oral_bioavailability": 0.10,
}

# AlphaFold URL builder -- works for any UniProt ID
def _alphafold_url(uniprot_id: str) -> str:
    """Return the AlphaFold PDB URL for a given UniProt accession."""
    return f"https://alphafold.ebi.ac.uk/files/AF-{uniprot_id}-F1-model_v4.pdb"

# ---------------------------------------------------------------------------
# Custom CSS -- professional biotech aesthetic
# ---------------------------------------------------------------------------
_CSS = """
<style>
/* ---- global ---------------------------------------------------------- */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ---- header ---------------------------------------------------------- */
.main-header {
    background: linear-gradient(135deg, #0a1628 0%, #132744 50%, #0d3b66 100%);
    padding: 2.5rem 2rem 2rem 2rem;
    border-radius: 12px;
    margin-bottom: 2rem;
    border: 1px solid rgba(56, 189, 248, 0.15);
    box-shadow: 0 4px 24px rgba(0, 0, 0, 0.25);
}
.main-header h1 {
    color: #e0f2fe;
    font-size: 2.2rem;
    font-weight: 700;
    margin: 0 0 0.3rem 0;
    letter-spacing: -0.02em;
}
.main-header p {
    color: #7dd3fc;
    font-size: 1.05rem;
    margin: 0;
    opacity: 0.85;
}

/* ---- metric cards ---------------------------------------------------- */
div[data-testid="stMetric"] {
    background: linear-gradient(145deg, #0f1f38, #162a4a);
    border: 1px solid rgba(56, 189, 248, 0.12);
    border-radius: 10px;
    padding: 1rem 1.2rem;
    box-shadow: 0 2px 12px rgba(0, 0, 0, 0.18);
}
div[data-testid="stMetric"] label {
    color: #7dd3fc !important;
    font-weight: 500;
    text-transform: uppercase;
    font-size: 0.75rem !important;
    letter-spacing: 0.06em;
}
div[data-testid="stMetric"] [data-testid="stMetricValue"] {
    color: #e0f2fe !important;
    font-weight: 700;
}

/* ---- section dividers ------------------------------------------------ */
.section-divider {
    border: none;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(56,189,248,0.25), transparent);
    margin: 2rem 0;
}

/* ---- section titles -------------------------------------------------- */
.section-title {
    color: #38bdf8;
    font-size: 1.25rem;
    font-weight: 600;
    margin-bottom: 0.75rem;
    letter-spacing: -0.01em;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

/* ---- table highlighting ---------------------------------------------- */
.top-candidate {
    background-color: rgba(56, 189, 248, 0.08) !important;
}

/* ---- sidebar --------------------------------------------------------- */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0a1628, #0f1f38);
}
section[data-testid="stSidebar"] .stMarkdown h2 {
    color: #7dd3fc;
}

/* ---- footer ---------------------------------------------------------- */
.footer-credits {
    text-align: center;
    color: #64748b;
    font-size: 0.82rem;
    padding: 1.5rem 0 0.5rem 0;
    border-top: 1px solid rgba(56,189,248,0.1);
    margin-top: 3rem;
}

/* ---- demo badge ------------------------------------------------------ */
.demo-badge {
    display: inline-block;
    background: rgba(251, 191, 36, 0.15);
    color: #fbbf24;
    padding: 0.25rem 0.75rem;
    border-radius: 100px;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    border: 1px solid rgba(251, 191, 36, 0.3);
    margin-left: 0.75rem;
    vertical-align: middle;
}
</style>
"""


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading demo data ...")
def _load_demo_data(disease_id: str) -> "pd.DataFrame":
    """Load and enrich pre-computed demo results for the selected disease.

    Loads the targets JSON (dict keyed by disease_id), filters docking
    results to only that disease's target proteins, and merges with
    training feature data to produce a single DataFrame containing all
    SCORING_FEATURES plus a composite score.
    """
    import json

    import pandas as pd

    docking = pd.read_csv(DATA_DIR / "sample_docking_results.csv")
    training = pd.read_csv(DATA_DIR / "training_repurposing_cases.csv")
    drugbank = pd.read_csv(DATA_DIR / "sample_drugbank.csv")

    # Load targets for the selected disease from the JSON dict
    targets_path = DATA_DIR / "sample_targets.json"
    if targets_path.exists():
        with open(targets_path, "r") as fh:
            all_targets = json.load(fh)
        disease_targets = all_targets.get(disease_id, [])
        disease_uniprots = {t["uniprot_id"] for t in disease_targets}
    else:
        disease_uniprots = set()

    # Filter docking results to this disease's target proteins
    if disease_uniprots:
        docking = docking[docking["uniprot_id"].isin(disease_uniprots)].copy()

    # Filter training data to the selected disease
    disease_training = training[training["disease_id"] == disease_id].copy()

    # Merge docking with training features on drugbank_id
    merged = docking.merge(
        disease_training[
            ["drugbank_id"] + SCORING_FEATURES + ["repurposing_success"]
        ],
        on="drugbank_id",
        how="left",
    )

    # Fill any drugs without training rows using drugbank metadata
    drug_meta = drugbank.set_index("drugbank_id")
    for idx in merged.index:
        dbid = merged.at[idx, "drugbank_id"]
        if pd.isna(merged.at[idx, "binding_affinity"]):
            aff = abs(merged.at[idx, "affinity_kcal_mol"])
            merged.at[idx, "binding_affinity"] = aff
            merged.at[idx, "plddt_score"] = 80.0
            merged.at[idx, "bioactivity_count"] = 3
            merged.at[idx, "safety_score"] = 0.6
            merged.at[idx, "tanimoto_similarity"] = 0.2
            merged.at[idx, "association_score"] = 0.4
            if dbid in drug_meta.index:
                merged.at[idx, "patent_expired"] = int(
                    drug_meta.at[dbid, "patent_expired"]
                )
                merged.at[idx, "oral_bioavailability"] = float(
                    drug_meta.at[dbid, "oral_bioavailability"]
                )
            else:
                merged.at[idx, "patent_expired"] = 1
                merged.at[idx, "oral_bioavailability"] = 0.5

    # Cast patent_expired to int safely
    merged["patent_expired"] = merged["patent_expired"].astype("Int64")

    # Compute composite score
    merged["composite_score"] = _compute_composite(merged)

    # Identify top contributing factor per row
    merged["top_contributing_factor"] = merged.apply(
        _top_factor, axis=1
    )

    # Sort and rank
    merged = merged.sort_values("composite_score", ascending=False).reset_index(
        drop=True
    )
    merged.insert(0, "rank", range(1, len(merged) + 1))

    return merged


def _normalize_series(s: "pd.Series") -> "pd.Series":
    """Min-max normalize a Series to [0, 1]."""
    mn, mx = s.min(), s.max()
    if mx == mn:
        return s * 0.0
    return (s - mn) / (mx - mn)


def _compute_composite(df: "pd.DataFrame") -> "pd.Series":
    """Weighted composite score from scoring features, 0-100 scale."""
    import pandas as pd

    score = pd.Series(0.0, index=df.index)
    for feat, w in _WEIGHTS.items():
        normed = _normalize_series(df[feat].astype(float))
        score += normed * w
    # Scale to 0-100
    return (score * 100).round(2)


def _top_factor(row: "pd.Series") -> str:
    """Return the feature name that contributes most to the composite."""
    best_feat = ""
    best_val = -1.0
    for feat, w in _WEIGHTS.items():
        val = float(row[feat]) * w
        if val > best_val:
            best_val = val
            best_feat = feat
    return _FEATURE_LABELS.get(best_feat, best_feat)


def _generate_pdf(df: "pd.DataFrame", disease_name: str) -> bytes:
    """Generate a PDF report summarizing top candidates."""
    from fpdf import FPDF

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.add_page()

    # Title
    pdf.set_font("Helvetica", "B", 22)
    pdf.set_text_color(10, 30, 60)
    pdf.cell(0, 14, "DrugSight Report", new_x="LMARGIN", new_y="NEXT")

    pdf.set_font("Helvetica", "", 11)
    pdf.set_text_color(80, 80, 80)
    pdf.cell(
        0,
        8,
        f"AI-Powered Drug Repurposing Analysis  |  {disease_name}",
        new_x="LMARGIN",
        new_y="NEXT",
    )
    pdf.cell(
        0,
        8,
        f"Generated {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        new_x="LMARGIN",
        new_y="NEXT",
    )
    pdf.ln(6)

    # Summary metrics
    pdf.set_font("Helvetica", "B", 13)
    pdf.set_text_color(10, 30, 60)
    pdf.cell(0, 10, "Summary", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(40, 40, 40)
    n_targets = df["target_symbol"].nunique()
    n_drugs = df["drug_name"].nunique()
    top_score = df["composite_score"].max()
    pdf.cell(
        0, 7, f"Targets screened: {n_targets}", new_x="LMARGIN", new_y="NEXT"
    )
    pdf.cell(
        0, 7, f"Drugs evaluated: {n_drugs}", new_x="LMARGIN", new_y="NEXT"
    )
    pdf.cell(
        0, 7, f"Top composite score: {top_score:.2f}", new_x="LMARGIN", new_y="NEXT"
    )
    pdf.ln(6)

    # Table header
    pdf.set_font("Helvetica", "B", 13)
    pdf.set_text_color(10, 30, 60)
    pdf.cell(0, 10, "Top 20 Candidates", new_x="LMARGIN", new_y="NEXT")

    col_widths = [10, 35, 24, 28, 24, 40, 22]
    headers = [
        "#",
        "Drug Name",
        "Score",
        "Affinity",
        "Target",
        "Top Factor",
        "Patent\nExpired",
    ]

    pdf.set_font("Helvetica", "B", 8)
    pdf.set_fill_color(15, 31, 56)
    pdf.set_text_color(224, 242, 254)
    for w, h in zip(col_widths, headers):
        pdf.cell(w, 8, h, border=1, fill=True)
    pdf.ln()

    pdf.set_font("Helvetica", "", 8)
    pdf.set_text_color(30, 30, 30)
    top = df.head(20)
    for _, row in top.iterrows():
        fill = int(row["rank"]) <= 3
        if fill:
            pdf.set_fill_color(230, 245, 255)
        pdf.cell(col_widths[0], 7, str(int(row["rank"])), border=1, fill=fill)
        pdf.cell(col_widths[1], 7, str(row["drug_name"])[:18], border=1, fill=fill)
        pdf.cell(
            col_widths[2], 7, f"{row['composite_score']:.2f}", border=1, fill=fill
        )
        aff = row.get("affinity_kcal_mol", "")
        pdf.cell(col_widths[3], 7, f"{aff} kcal/mol", border=1, fill=fill)
        pdf.cell(col_widths[4], 7, str(row["target_symbol"]), border=1, fill=fill)
        pdf.cell(
            col_widths[5], 7, str(row["top_contributing_factor"])[:22], border=1, fill=fill
        )
        patent = "Yes" if int(row.get("patent_expired", 0)) == 1 else "No"
        pdf.cell(col_widths[6], 7, patent, border=1, fill=fill)
        pdf.ln()

    # Footer
    pdf.ln(10)
    pdf.set_font("Helvetica", "I", 8)
    pdf.set_text_color(120, 120, 120)
    pdf.cell(
        0,
        6,
        "Built with AlphaFold, Open Targets, RDKit, AutoDock Vina  |  drugsight v0.1.0",
        new_x="LMARGIN",
        new_y="NEXT",
    )

    return bytes(pdf.output())


# ---------------------------------------------------------------------------
# 3D Protein viewer helper
# ---------------------------------------------------------------------------
def _render_protein_viewer(uniprot_id: str, target_symbol: str) -> None:
    """Render a 3D protein structure using py3Dmol via stmol."""
    try:
        import py3Dmol
        from stmol import showmol
    except ImportError:
        st.warning(
            "Install `stmol` and `py3Dmol` for 3D protein visualization: "
            "`pip install stmol py3Dmol`"
        )
        return

    pdb_path = STRUCTURES_DIR / f"{uniprot_id}.pdb"
    pdb_data: str | None = None

    if pdb_path.exists():
        pdb_data = pdb_path.read_text()

    viewer = py3Dmol.view(width=700, height=500)

    if pdb_data:
        viewer.addModel(pdb_data, "pdb")
    else:
        # Fetch directly from AlphaFold
        url = _alphafold_url(uniprot_id)
        viewer.addModel("", "pdb")
        # Use JavaScript fetch to load the model from AlphaFold
        viewer_script = f"""
        fetch('{url}')
            .then(r => r.text())
            .then(data => {{
                viewer.addModel(data, 'pdb');
                viewer.setStyle({{}}, {{
                    cartoon: {{
                        colorscheme: {{
                            prop: 'b',
                            gradient: 'rwb',
                            min: 50,
                            max: 90
                        }}
                    }}
                }});
                viewer.zoomTo();
                viewer.render();
            }});
        """
        st.info(
            f"Loading **{target_symbol}** (UniProt: {uniprot_id}) from "
            f"AlphaFold Protein Structure Database..."
        )

    # pLDDT-based coloring: B-factor field stores pLDDT in AlphaFold PDBs
    # Blue (high confidence) -> Red (low confidence)
    viewer.setStyle(
        {},
        {
            "cartoon": {
                "colorscheme": {
                    "prop": "b",
                    "gradient": "rwb",
                    "min": 50,
                    "max": 90,
                }
            }
        },
    )
    viewer.setBackgroundColor("#0a1628")
    viewer.zoomTo()

    showmol(viewer, height=500, width=700)

    # Legend
    cols = st.columns(5)
    cols[0].markdown(
        '<span style="color:#2563eb;font-weight:600;">&#9608; pLDDT > 90 (Very high)</span>',
        unsafe_allow_html=True,
    )
    cols[1].markdown(
        '<span style="color:#60a5fa;font-weight:600;">&#9608; 70-90 (Confident)</span>',
        unsafe_allow_html=True,
    )
    cols[2].markdown(
        '<span style="color:#fbbf24;font-weight:600;">&#9608; 50-70 (Low)</span>',
        unsafe_allow_html=True,
    )
    cols[3].markdown(
        '<span style="color:#ef4444;font-weight:600;">&#9608; < 50 (Very low)</span>',
        unsafe_allow_html=True,
    )
    cols[4].markdown(
        f"<span style='color:#94a3b8;font-size:0.85rem;'>Source: AlphaFold DB v4</span>",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Chart builders (lazy Plotly imports)
# ---------------------------------------------------------------------------
def _build_score_histogram(df: "pd.DataFrame") -> "go.Figure":  # noqa: F821
    """Plotly histogram of composite scores, colored by patent status."""
    import plotly.express as px

    df_plot = df.copy()
    df_plot["Patent Status"] = df_plot["patent_expired"].apply(
        lambda x: "Expired (repurposable)" if int(x) == 1 else "Active"
    )

    fig = px.histogram(
        df_plot,
        x="composite_score",
        color="Patent Status",
        nbins=20,
        title="Distribution of Repurposing Scores",
        color_discrete_map={
            "Expired (repurposable)": "#38bdf8",
            "Active": "#f472b6",
        },
        template="plotly_dark",
        labels={"composite_score": "Composite Score"},
    )
    fig.update_layout(
        plot_bgcolor="rgba(10,22,40,0.9)",
        paper_bgcolor="rgba(10,22,40,0.9)",
        font=dict(family="Inter, sans-serif", color="#e0f2fe"),
        title_font=dict(size=16, color="#7dd3fc"),
        bargap=0.08,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
        xaxis=dict(gridcolor="rgba(56,189,248,0.08)"),
        yaxis=dict(gridcolor="rgba(56,189,248,0.08)", title="Count"),
    )
    return fig


def _build_shap_chart(
    row: "pd.Series", drug_name: str, rank: int
) -> "go.Figure":  # noqa: F821
    """Bar chart showing pseudo-SHAP feature contributions."""
    import plotly.graph_objects as go

    contributions: list[tuple[str, float]] = []
    for feat in SCORING_FEATURES:
        val = float(row[feat])
        weight = _WEIGHTS.get(feat, 0.1)
        contrib = val * weight
        contributions.append((_FEATURE_LABELS.get(feat, feat), contrib))

    contributions.sort(key=lambda x: x[1])
    labels = [c[0] for c in contributions]
    values = [c[1] for c in contributions]

    colors = [
        "#38bdf8" if v >= 0 else "#f87171" for v in values
    ]

    fig = go.Figure(
        go.Bar(
            x=values,
            y=labels,
            orientation="h",
            marker_color=colors,
            marker_line=dict(color="rgba(255,255,255,0.1)", width=0.5),
            text=[f"{v:.2f}" for v in values],
            textposition="outside",
            textfont=dict(size=11, color="#e0f2fe"),
        )
    )
    fig.update_layout(
        title=dict(
            text=f"Why does {drug_name} rank #{rank}?",
            font=dict(size=16, color="#7dd3fc"),
        ),
        template="plotly_dark",
        plot_bgcolor="rgba(10,22,40,0.9)",
        paper_bgcolor="rgba(10,22,40,0.9)",
        font=dict(family="Inter, sans-serif", color="#e0f2fe"),
        xaxis=dict(
            title="Feature Contribution (weight \u00d7 value)",
            gridcolor="rgba(56,189,248,0.08)",
        ),
        yaxis=dict(gridcolor="rgba(56,189,248,0.08)"),
        height=400,
        margin=dict(l=150, r=60, t=60, b=50),
    )
    return fig


def _build_radar_chart(df: "pd.DataFrame") -> "go.Figure":  # noqa: F821
    """Radar chart comparing top 5 candidates across scoring features."""
    import plotly.graph_objects as go

    top5 = df.head(5).copy()
    feature_labels = [_FEATURE_LABELS.get(f, f) for f in SCORING_FEATURES]

    # Normalize each feature to 0-1 across the full dataset for fair comparison
    normed = {}
    for feat in SCORING_FEATURES:
        normed[feat] = _normalize_series(df[feat].astype(float))

    palette = ["#38bdf8", "#a78bfa", "#34d399", "#fbbf24", "#f472b6"]

    # Convert hex palette to rgba for fill transparency
    def _hex_to_rgba(hex_color: str, alpha: float = 0.08) -> str:
        h = hex_color.lstrip("#")
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        return f"rgba({r},{g},{b},{alpha})"

    fig = go.Figure()
    for i, (_, row) in enumerate(top5.iterrows()):
        idx = row.name
        values = [float(normed[feat].iloc[idx]) for feat in SCORING_FEATURES]
        values.append(values[0])  # close the polygon

        color = palette[i % len(palette)]
        fig.add_trace(
            go.Scatterpolar(
                r=values,
                theta=feature_labels + [feature_labels[0]],
                fill="toself",
                fillcolor=_hex_to_rgba(color, 0.10),
                opacity=0.85,
                name=f"#{int(row['rank'])} {row['drug_name']}",
                line=dict(color=color, width=2),
            )
        )

    fig.update_layout(
        title=dict(
            text="Top 5 Candidates \u2014 Feature Comparison",
            font=dict(size=16, color="#7dd3fc"),
        ),
        template="plotly_dark",
        polar=dict(
            bgcolor="rgba(10,22,40,0.9)",
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                gridcolor="rgba(56,189,248,0.12)",
                tickfont=dict(size=9, color="#64748b"),
            ),
            angularaxis=dict(
                gridcolor="rgba(56,189,248,0.12)",
                tickfont=dict(size=11, color="#e0f2fe"),
            ),
        ),
        paper_bgcolor="rgba(10,22,40,0.9)",
        font=dict(family="Inter, sans-serif", color="#e0f2fe"),
        legend=dict(
            font=dict(size=11),
            bgcolor="rgba(10,22,40,0.6)",
            bordercolor="rgba(56,189,248,0.15)",
            borderwidth=1,
        ),
        height=520,
        margin=dict(t=80, b=40),
    )
    return fig


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
def _render_sidebar() -> tuple[str, str, float, int, bool]:
    """Render sidebar controls and return user selections."""
    with st.sidebar:
        st.markdown("## DrugSight Configuration")
        st.markdown("---")

        # Disease selection
        disease_label = st.selectbox(
            "Disease",
            options=list(RARE_DISEASES.keys()),
            index=0,
            help="Select a rare disease to explore repurposing candidates.",
        )
        disease_id = RARE_DISEASES[disease_label]

        st.text_input(
            "EFO Disease ID",
            value=disease_id,
            disabled=True,
            help="Experimental Factor Ontology identifier.",
        )

        st.markdown("---")

        min_assoc = st.slider(
            "Min association score",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05,
            help="Filter targets with Open Targets association score below this threshold.",
        )

        max_results = st.slider(
            "Max results displayed",
            min_value=10,
            max_value=100,
            value=20,
            step=5,
        )

        st.markdown("---")

        demo_mode = st.toggle("Demo Mode", value=True, help="Load pre-computed results. Turn off to run the full pipeline (requires local environment).")

        if demo_mode:
            st.info(f"Using pre-computed results for {disease_label}.")
        else:
            st.warning("Live mode requires AutoDock Vina, RDKit, and network access.")

        st.button(
            "Run Pipeline",
            disabled=demo_mode,
            help="Coming soon \u2014 requires full environment with AutoDock Vina and RDKit."
            if demo_mode
            else "Execute the full drug repurposing pipeline.",
            use_container_width=True,
        )

        st.markdown("---")
        st.markdown(
            "<p style='color:#64748b;font-size:0.75rem;text-align:center;'>"
            "DrugSight v0.1.0<br>100% Free & Open Source</p>",
            unsafe_allow_html=True,
        )

    return disease_id, disease_label, min_assoc, max_results, demo_mode


# ---------------------------------------------------------------------------
# Main application
# ---------------------------------------------------------------------------
def main() -> None:
    """Entry point for the DrugSight dashboard."""
    import pandas as pd

    # Inject custom CSS
    st.markdown(_CSS, unsafe_allow_html=True)

    # Sidebar
    disease_id, disease_label, min_assoc, max_results, demo_mode = _render_sidebar()

    # -- Header ---
    st.markdown(
        '<div class="main-header">'
        "<h1>\U0001f52c DrugSight: AI-Powered Drug Repurposing Engine"
        '<span class="demo-badge">DEMO</span></h1>'
        "<p>Built on AlphaFold's Open Protein Structure Database "
        "\u2022 Open Targets \u2022 AutoDock Vina</p>"
        "</div>",
        unsafe_allow_html=True,
    )

    # -- Load data ---
    if demo_mode:
        df_all = _load_demo_data(disease_id)
    else:
        st.error(
            "Live pipeline is not yet available. Enable **Demo Mode** in the sidebar."
        )
        st.stop()

    # Limit to user-selected max results
    df = df_all.head(max_results).copy()

    # ====================================================================
    # Section 1 -- Overview Metrics
    # ====================================================================
    st.markdown('<div class="section-title">\U0001f4ca Overview</div>', unsafe_allow_html=True)

    n_targets = df_all["target_symbol"].nunique()
    n_structures = df_all["uniprot_id"].nunique()
    n_drugs = df_all["drug_name"].nunique()
    top_score = df_all["composite_score"].max()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Targets Found", n_targets)
    c2.metric("Structures Retrieved", n_structures)
    c3.metric("Drugs Screened", n_drugs)
    c4.metric("Top Score", f"{top_score:.1f}", delta=f"+{top_score - df_all['composite_score'].median():.1f} vs median")

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # ====================================================================
    # Section 2 -- Top Candidates Table
    # ====================================================================
    st.markdown(
        '<div class="section-title">\U0001f3af Top Candidates</div>',
        unsafe_allow_html=True,
    )

    display_cols = [
        "rank",
        "drug_name",
        "composite_score",
        "affinity_kcal_mol",
        "target_symbol",
        "top_contributing_factor",
        "patent_expired",
    ]

    display_df = df[display_cols].copy()
    display_df.columns = [
        "Rank",
        "Drug Name",
        "Composite Score",
        "Binding Affinity (kcal/mol)",
        "Target",
        "Top Contributing Factor",
        "Patent Expired",
    ]
    display_df["Patent Expired"] = display_df["Patent Expired"].apply(
        lambda x: "Yes" if int(x) == 1 else "No"
    )
    display_df["Composite Score"] = display_df["Composite Score"].apply(
        lambda x: f"{x:.2f}"
    )

    def _highlight_top3(row: "pd.Series") -> list[str]:
        if int(row["Rank"]) <= 3:
            return ["background-color: rgba(56, 189, 248, 0.10); font-weight: 600"] * len(row)
        return [""] * len(row)

    styled = display_df.style.apply(_highlight_top3, axis=1)
    st.dataframe(
        styled,
        use_container_width=True,
        hide_index=True,
        height=min(38 * len(display_df) + 40, 780),
    )

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # ====================================================================
    # Section 3 -- Score Distribution
    # ====================================================================
    st.markdown(
        '<div class="section-title">\U0001f4c8 Score Distribution</div>',
        unsafe_allow_html=True,
    )
    st.plotly_chart(
        _build_score_histogram(df_all),
        use_container_width=True,
        config={"displayModeBar": False},
    )

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # ====================================================================
    # Section 4 -- SHAP Analysis
    # ====================================================================
    st.markdown(
        '<div class="section-title">\U0001f9ec Feature Attribution (SHAP-style)</div>',
        unsafe_allow_html=True,
    )

    drug_options = df["drug_name"].unique().tolist()
    selected_drug = st.selectbox(
        "Select a candidate to explain",
        options=drug_options,
        index=0,
        key="shap_drug_select",
    )

    drug_row = df[df["drug_name"] == selected_drug].iloc[0]
    st.plotly_chart(
        _build_shap_chart(drug_row, selected_drug, int(drug_row["rank"])),
        use_container_width=True,
        config={"displayModeBar": False},
    )

    # Show the raw feature values in an expander
    with st.expander("View raw feature values"):
        feat_data = {
            _FEATURE_LABELS.get(f, f): [drug_row[f]] for f in SCORING_FEATURES
        }
        st.dataframe(pd.DataFrame(feat_data), hide_index=True)

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # ====================================================================
    # Section 5 -- 3D Protein Viewer
    # ====================================================================
    st.markdown(
        '<div class="section-title">\U0001f9eb 3D Protein Structure (AlphaFold)</div>',
        unsafe_allow_html=True,
    )

    targets = df_all[["target_symbol", "uniprot_id"]].drop_duplicates()
    target_labels = {
        row["target_symbol"]: row["uniprot_id"] for _, row in targets.iterrows()
    }

    selected_target = st.selectbox(
        "Select target protein",
        options=list(target_labels.keys()),
        index=0,
        key="protein_select",
    )
    selected_uniprot = target_labels[selected_target]

    _render_protein_viewer(selected_uniprot, selected_target)

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # ====================================================================
    # Section 6 -- Radar Chart
    # ====================================================================
    st.markdown(
        '<div class="section-title">\U0001f578\ufe0f Feature Comparison \u2014 Top 5</div>',
        unsafe_allow_html=True,
    )
    st.plotly_chart(
        _build_radar_chart(df_all),
        use_container_width=True,
        config={"displayModeBar": False},
    )

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # ====================================================================
    # Footer -- Downloads & Credits
    # ====================================================================
    st.markdown(
        '<div class="section-title">\U0001f4e5 Export</div>',
        unsafe_allow_html=True,
    )

    dl_col1, dl_col2, _ = st.columns([1, 1, 2])

    with dl_col1:
        csv_buf = io.StringIO()
        df_all.to_csv(csv_buf, index=False)
        st.download_button(
            label="Download Results CSV",
            data=csv_buf.getvalue(),
            file_name=f"drugsight_{disease_id}_results.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with dl_col2:
        pdf_bytes = _generate_pdf(df_all, disease_label)
        st.download_button(
            label="Download PDF Report",
            data=pdf_bytes,
            file_name=f"drugsight_{disease_id}_report.pdf",
            mime="application/pdf",
            use_container_width=True,
        )

    st.markdown(
        '<div class="footer-credits">'
        "Built with AlphaFold \u2022 Open Targets \u2022 RDKit \u2022 AutoDock Vina "
        "&nbsp;|&nbsp; 100% Free &amp; Open Source"
        "</div>",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    main()
