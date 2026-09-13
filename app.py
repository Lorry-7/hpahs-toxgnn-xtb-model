"""Minimal, publication-style Streamlit interface for ToxGNN-xTB inference."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from prediction_engine import (  # noqa: E402
    LOCKED_PERFORMANCE,
    REQUIRED_XTB,
    ToxGNNEngine,
    clean_molecule_table,
    read_table,
)


st.set_page_config(
    page_title="ToxGNN-xTB Prediction Platform",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    :root {
        --navy: #17324d;
        --blue: #1d6f8a;
        --teal: #36a69a;
        --mint: #eaf7f4;
        --coral: #e97862;
        --ink: #233545;
        --muted: #637789;
        --line: #dce8ed;
    }
    .stApp {
        background: linear-gradient(180deg, #f3faf9 0, #ffffff 260px);
        color: var(--ink);
    }
    .block-container {
        max-width: 1240px;
        padding-top: 2.1rem;
        padding-bottom: 3.5rem;
    }
    .hero {
        position: relative;
        overflow: hidden;
        border: 1px solid rgba(43, 128, 145, .16);
        border-radius: 24px;
        padding: 2.15rem 2.3rem 1.95rem;
        background:
            radial-gradient(circle at 88% 18%, rgba(54, 166, 154, .20), transparent 28%),
            linear-gradient(125deg, #ffffff 0%, #f2fbf9 60%, #edf7fb 100%);
        box-shadow: 0 18px 48px rgba(22, 62, 79, .08);
        margin-bottom: 1.4rem;
    }
    .eyebrow {
        color: var(--blue);
        font-size: .76rem;
        font-weight: 750;
        letter-spacing: .15em;
        text-transform: uppercase;
        margin-bottom: .55rem;
    }
    .hero h1 {
        color: var(--navy);
        font-size: clamp(2.05rem, 4vw, 3.25rem);
        line-height: 1.05;
        letter-spacing: -.035em;
        margin: 0 0 .7rem;
    }
    .hero p {
        max-width: 760px;
        color: var(--muted);
        font-size: 1.02rem;
        line-height: 1.65;
        margin: 0;
    }
    .section-title {
        color: var(--navy);
        font-size: 1.22rem;
        font-weight: 760;
        margin: .45rem 0 .2rem;
    }
    .section-note {
        color: var(--muted);
        font-size: .92rem;
        margin-bottom: .8rem;
    }
    div[data-testid="stMetric"] {
        background: rgba(255,255,255,.92);
        border: 1px solid var(--line);
        border-radius: 15px;
        padding: .9rem 1rem;
        box-shadow: 0 7px 22px rgba(23, 50, 77, .045);
    }
    div[data-testid="stMetricLabel"] {color: var(--muted);}
    div[data-testid="stMetricValue"] {color: var(--navy);}
    div[data-testid="stFileUploader"] {
        background: #fbfefe;
        border: 1px solid var(--line);
        border-radius: 16px;
        padding: .7rem .85rem .2rem;
    }
    div[data-testid="stTextArea"] textarea {
        border-color: var(--line);
        border-radius: 13px;
        background: #fbfefe;
    }
    .stButton > button[kind="primary"] {
        min-height: 3.15rem;
        border: 0;
        border-radius: 13px;
        color: #fff;
        font-weight: 740;
        letter-spacing: .01em;
        background: linear-gradient(100deg, #176c86, #35a496);
        box-shadow: 0 10px 24px rgba(29, 111, 138, .22);
    }
    .stButton > button[kind="primary"]:hover {
        color: #fff;
        transform: translateY(-1px);
        box-shadow: 0 13px 28px rgba(29, 111, 138, .28);
    }
    .stDownloadButton > button {
        border-radius: 11px;
        border-color: #b9d4da;
        color: var(--blue);
        font-weight: 650;
    }
    div[data-testid="stDataFrame"] {
        border: 1px solid var(--line);
        border-radius: 14px;
        overflow: hidden;
    }
    .method-strip {
        margin-top: 1.25rem;
        border-top: 1px solid var(--line);
        padding-top: 1rem;
        color: var(--muted);
        font-size: .82rem;
        line-height: 1.65;
    }
    .chip {
        display: inline-block;
        margin: .15rem .28rem .15rem 0;
        padding: .22rem .55rem;
        border-radius: 999px;
        background: var(--mint);
        color: #246f69;
        border: 1px solid #cfe9e4;
        font-size: .76rem;
        font-weight: 650;
    }
    footer {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)


def model_paths() -> dict[str, Path]:
    return {
        "source": PROJECT_ROOT / "data/processed/source_domain_tox.csv",
        "xtb": PROJECT_ROOT / "data/features/xtb_descriptors.parquet",
        "checkpoint": PROJECT_ROOT
        / "artifacts/fresh_runs/final_locked_r2_0p84384_seed3407/stage2_lc50_adapter/best_loss.ckpt",
        "bundle": APP_DIR / "models/final_toxgnn_xtb_bundle.joblib",
    }


@st.cache_resource(show_spinner=False)
def get_engine() -> ToxGNNEngine:
    paths = model_paths()
    return ToxGNNEngine(paths["source"], paths["xtb"], paths["checkpoint"])


def manual_table(text: str) -> pd.DataFrame:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return pd.DataFrame(
        {
            "compound_id": [f"manual_{index:04d}" for index in range(1, len(lines) + 1)],
            "smiles": lines,
        }
    )


def collect_input(uploaded_file, manual_smiles: str) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    if uploaded_file is not None:
        uploaded_file.seek(0)
        uploaded_frame = read_table(uploaded_file)
        lookup = {str(column).strip().lower(): column for column in uploaded_frame.columns}
        for alias in ["smiles", "canonical_smiles", "structure", "mol_smiles"]:
            if alias in lookup:
                uploaded_frame = uploaded_frame.rename(columns={lookup[alias]: "smiles"})
                break
        if "compound_id" not in uploaded_frame.columns:
            for alias in ["compound_id", "mol_id", "id", "name", "cas"]:
                if alias in lookup:
                    uploaded_frame = uploaded_frame.rename(columns={lookup[alias]: "compound_id"})
                    break
        frames.append(uploaded_frame)
    if manual_smiles.strip():
        frames.append(manual_table(manual_smiles))
    if not frames:
        raise ValueError("Upload a molecule table or enter at least one SMILES string.")
    return pd.concat(frames, ignore_index=True, sort=False)


def run_prediction(raw: pd.DataFrame, xtb_executable: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    clean, audit = clean_molecule_table(raw)
    if clean.empty:
        raise ValueError("No valid unique molecular structure remained after cleaning.")
    engine = get_engine()
    prepared = engine.prepare(clean)
    prepared = engine.compute_missing_xtb(prepared, executable=xtb_executable)
    failures = prepared.loc[~prepared["xtb_complete"]]
    if not failures.empty:
        failed_ids = ", ".join(failures["compound_id"].astype(str).head(6))
        suffix = " ..." if len(failures) > 6 else ""
        raise RuntimeError(
            f"xTB descriptors could not be completed for {len(failures)} compound(s): "
            f"{failed_ids}{suffix}. Review the xTB executable and calculation output."
        )
    result = engine.predict(prepared, ensemble_size=1)
    return result, audit, prepared


paths = model_paths()
assets_ready = all(path.exists() for path in paths.values())

st.markdown(
    """
    <div class="hero">
      <div class="eyebrow">Environmental toxicology prediction</div>
      <h1>ToxGNN-xTB</h1>
      <p>One-step HPAH toxicity prediction from molecular structure, combining an
      attention-enhanced graph representation with quantum-chemical information.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

metric_cols = st.columns(4)
metric_cols[0].metric("Training R2", f"{LOCKED_PERFORMANCE['train_r2']:.3f}")
metric_cols[1].metric("Test R2", f"{LOCKED_PERFORMANCE['test_r2']:.3f}")
metric_cols[2].metric("Test MAE", f"{LOCKED_PERFORMANCE['test_mae']:.3f}")
metric_cols[3].metric("Test RMSE", f"{LOCKED_PERFORMANCE['test_rmse']:.3f}")

with st.sidebar:
    st.header("Model configuration")
    st.caption("The application loads the locked model and calculates missing quantum descriptors automatically.")
    if assets_ready:
        st.success("All model assets are available.")
    else:
        st.error("One or more model assets are missing.")
        for name, path in paths.items():
            if not path.exists():
                st.caption(f"Missing {name}: {path}")
    xtb_executable = st.text_input(
        "xTB executable",
        value="xtb",
        help="Enter xtb when it is available on PATH, or provide the full executable path.",
    )
    st.divider()
    st.caption("Locked output scale: pLC50")
    st.caption("GFN2-xTB / ALPB water / single-point protocol")

st.markdown('<div class="section-title">Molecular input</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-note">Upload a CSV or Excel file containing a SMILES column, or paste one SMILES string per line.</div>',
    unsafe_allow_html=True,
)

input_left, input_right = st.columns([1.03, 1], gap="large")
with input_left:
    uploaded = st.file_uploader(
        "Molecule table",
        type=["csv", "xlsx", "xls"],
        help="Accepted structure-column names include smiles, canonical_smiles, and structure.",
    )
    if uploaded is not None:
        st.caption(f"Selected file: {uploaded.name}")
with input_right:
    manual_smiles = st.text_area(
        "SMILES input",
        placeholder="c1ccc2ccccc2c1\nClc1cccc2ccccc12",
        height=142,
        help="Enter one molecular structure per line.",
    )

run_clicked = st.button(
    "Run ToxGNN-xTB prediction",
    type="primary",
    use_container_width=True,
    disabled=not assets_ready,
)

if run_clicked:
    try:
        raw_table = collect_input(uploaded, manual_smiles)
        with st.spinner("Validating structures, calculating xTB descriptors, and generating predictions..."):
            results, audit, prepared = run_prediction(raw_table, xtb_executable)
        st.session_state["results"] = results
        st.session_state["audit"] = audit
        st.session_state["prepared"] = prepared
        st.success(f"Prediction completed for {len(results)} compound(s).")
    except Exception as exc:
        st.error(str(exc))

if "results" in st.session_state:
    results = st.session_state["results"]
    audit = st.session_state["audit"]
    prepared = st.session_state["prepared"]

    st.markdown('<div class="section-title">Prediction results</div>', unsafe_allow_html=True)
    summary_cols = st.columns(4)
    summary_cols[0].metric("Predicted compounds", f"{len(results)}")
    summary_cols[1].metric("Mean predicted pLC50", f"{results['pred_pLC50'].mean():.3f}")
    summary_cols[2].metric("Within applicability domain", f"{(results['AD_overall'] == 'inside').sum()} / {len(results)}")
    summary_cols[3].metric("Rejected input records", f"{(audit['status'] == 'rejected').sum()}")

    primary_columns = [
        "compound_id",
        "canonical_smiles",
        "pred_pLC50",
        "pred_pLC50_gnn_branch",
        "pred_pLC50_xtb_branch",
        "AD_overall",
    ]
    if "LC50_mg_L" in results.columns:
        primary_columns.insert(3, "LC50_mg_L")
    formatters = {
        "pred_pLC50": "{:.4f}",
        "pred_pLC50_gnn_branch": "{:.4f}",
        "pred_pLC50_xtb_branch": "{:.4f}",
    }
    if "LC50_mg_L" in primary_columns:
        formatters["LC50_mg_L"] = "{:.6g}"
    st.dataframe(
        results[primary_columns].style.format(formatters, na_rep="-"),
        use_container_width=True,
        hide_index=True,
    )

    download_left, download_right = st.columns(2)
    with download_left:
        st.download_button(
            "Download predictions",
            data=results.to_csv(index=False).encode("utf-8-sig"),
            file_name="toxgnn_xtb_predictions.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with download_right:
        st.download_button(
            "Download cleaning audit",
            data=audit.to_csv(index=False).encode("utf-8-sig"),
            file_name="toxgnn_input_audit.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with st.expander("Applicability-domain and calculation details"):
        detail_columns = [
            "compound_id",
            "nearest_tanimoto",
            "embedding_distance",
            "xtb_mahalanobis",
            "AD_structural",
            "AD_embedding",
            "AD_xtb",
            "AD_overall",
        ]
        st.dataframe(results[detail_columns], use_container_width=True, hide_index=True)
        status_columns = [c for c in ["compound_id", "xtb_status", "xtb_message", *REQUIRED_XTB] if c in prepared.columns]
        st.dataframe(prepared[status_columns], use_container_width=True, hide_index=True)

    with st.expander("Input-cleaning record"):
        st.dataframe(audit, use_container_width=True, hide_index=True)

st.markdown(
    '<div class="method-strip"><strong>Locked inference configuration</strong><br>'
    'GNN weight: 0.5955 &nbsp;|&nbsp; xTB weight: 0.4045 &nbsp;|&nbsp; calibration: ridge3<br>'
    + "".join(f'<span class="chip">{name}</span>' for name in REQUIRED_XTB)
    + '<br>Predictions outside the applicability domain require expert review and experimental confirmation.</div>',
    unsafe_allow_html=True,
)
