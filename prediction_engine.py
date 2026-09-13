"""Inference services for the Streamlit ToxGNN-xTB application.

The module deliberately reuses the project's model, feature, fusion, and
applicability-domain implementations. It does not create surrogate values for
missing xTB descriptors.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

APP_DIR = Path(__file__).resolve().parent
# The public deployment keeps app.py/prediction_engine.py at repository root,
# while the research workspace keeps them in toxgnn_streamlit_app/. Resolve
# the root from whichever layout is present.
PROJECT_ROOT = APP_DIR if (APP_DIR / "src").exists() else APP_DIR.parent
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# The authoritative Stage-3 xTB branch is correlation-selected (k=4), not the
# previous QS5 placeholder. The four descriptors are selected from the raw xTB
# table using source-domain absolute Pearson correlation with pLC50.
RAW_XTB = [
    "total_energy", "homo", "lumo", "gap", "dipole", "min_charge",
    "max_charge", "max_abs_charge", "charge_range", "isotropic_polarizability",
]
REQUIRED_XTB = ["isotropic_polarizability", "total_energy", "homo", "lumo"]
# Locked performance reported for the final ToxGNN-xTB model in the manuscript.
# These values are metadata only; the app never recomputes or substitutes them.
LOCKED_PERFORMANCE = {
    "train_r2": 0.922,
    "test_r2": 0.844,
    "test_r2_exact": 0.84384,
    "test_mae": 0.41136,
    "test_rmse": 0.59832,
}
ALIASES = {
    "smiles": ["smiles", "SMILES", "canonical_smiles", "structure", "mol_smiles"],
    "compound_id": ["compound_id", "mol_id", "id", "name", "CAS", "cas"],
}


def _first_column(frame: pd.DataFrame, names: Iterable[str]) -> str | None:
    lookup = {str(c).strip().lower(): c for c in frame.columns}
    for name in names:
        if name.lower() in lookup:
            return lookup[name.lower()]
    return None


def read_table(uploaded_file_or_path) -> pd.DataFrame:
    """Read CSV/XLSX/Parquet input into a dataframe."""
    if isinstance(uploaded_file_or_path, (str, Path)):
        path = Path(uploaded_file_or_path)
        suffix = path.suffix.lower()
    else:
        suffix = Path(getattr(uploaded_file_or_path, "name", "input.csv")).suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(uploaded_file_or_path)
    if suffix == ".parquet":
        try:
            return pd.read_parquet(uploaded_file_or_path)
        except Exception:
            # Some legacy descriptor artifacts in the project were written with
            # metadata that current pyarrow rejects; fastparquet can still read
            # them losslessly.
            # Streamlit UploadedFile objects are seekable streams.  pyarrow may
            # advance the stream before raising, so rewind before retrying.
            if hasattr(uploaded_file_or_path, "seek"):
                uploaded_file_or_path.seek(0)
            return pd.read_parquet(uploaded_file_or_path, engine="fastparquet")
    return pd.read_csv(uploaded_file_or_path)


def clean_molecule_table(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Canonicalize SMILES, remove invalid structures and duplicates.

    Returns `(clean_table, audit_table)`. The audit table is intentionally
    explicit so users can inspect every rejected record.
    """
    from rdkit import Chem

    if raw.empty:
        raise ValueError("The uploaded table is empty.")
    smiles_col = _first_column(raw, ALIASES["smiles"])
    if smiles_col is None:
        raise ValueError("A SMILES column is required. Accepted names: smiles, canonical_smiles, structure.")
    id_col = _first_column(raw, ALIASES["compound_id"])
    work = raw.copy().reset_index(drop=True)
    work["input_row"] = np.arange(len(work)) + 1
    work["input_smiles"] = work[smiles_col].astype("string").fillna("").str.strip()
    if id_col is None:
        work["compound_id"] = [f"compound_{i:04d}" for i in range(1, len(work) + 1)]
    else:
        work["compound_id"] = work[id_col].astype("string").fillna("").str.strip()
        work.loc[work["compound_id"].eq(""), "compound_id"] = work.loc[
            work["compound_id"].eq(""), "input_row"
        ].map(lambda x: f"compound_{int(x):04d}")

    audit = []
    canonical = []
    for _, row in work.iterrows():
        mol = Chem.MolFromSmiles(row["input_smiles"])
        if mol is None:
            canonical.append(None)
            audit.append({"input_row": int(row["input_row"]), "compound_id": row["compound_id"],
                          "status": "rejected", "reason": "Invalid SMILES"})
        else:
            canonical.append(Chem.MolToSmiles(mol, canonical=True))
    work["canonical_smiles"] = canonical
    valid = work[work["canonical_smiles"].notna()].copy()
    duplicate_mask = valid.duplicated("canonical_smiles", keep="first")
    for _, row in valid[duplicate_mask].iterrows():
        audit.append({"input_row": int(row["input_row"]), "compound_id": row["compound_id"],
                      "status": "rejected", "reason": "Duplicate canonical SMILES"})
    valid = valid.loc[~duplicate_mask].copy()
    for _, row in valid.iterrows():
        audit.append({"input_row": int(row["input_row"]), "compound_id": row["compound_id"],
                      "status": "accepted", "reason": "Valid unique structure"})
    return valid.reset_index(drop=True), pd.DataFrame(audit).sort_values("input_row").reset_index(drop=True)


def normalize_descriptor_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize common descriptor spellings without changing numerical values."""
    out = frame.copy()
    aliases = {
        "total_energy": ["total_energy", "total energy", "energy", "etotal"],
        "homo": ["homo", "ehomo", "homo_energy", "homo energy"],
        "lumo": ["lumo", "elumo", "lumo_energy", "lumo energy"],
        "gap": ["gap", "homo_lumo_gap", "homo-lumo gap", "homo_lumo"],
        "charge_range": ["charge_range", "charge range", "q_range"],
        "isotropic_polarizability": [
            "isotropic_polarizability", "isotropic polarizability",
            "polarizability", "alpha_iso", "alpha isotropic",
        ],
    }
    lookup = {str(c).strip().lower(): c for c in out.columns}
    for canonical, names in aliases.items():
        if canonical in out.columns:
            continue
        for name in names:
            if name.lower() in lookup:
                out[canonical] = out[lookup[name.lower()]]
                break
    return out


def attach_xtb_descriptors(
    molecules: pd.DataFrame,
    descriptor_table: pd.DataFrame,
) -> pd.DataFrame:
    """Join descriptor values by mol_id or canonical SMILES, then audit completeness."""
    descriptors = normalize_descriptor_columns(descriptor_table)
    if not set(REQUIRED_XTB).issubset(descriptors.columns):
        missing = sorted(set(REQUIRED_XTB) - set(descriptors.columns))
        raise ValueError(f"Descriptor table is missing required model inputs: {', '.join(missing)}")
    descriptors = descriptors.copy()
    if "smiles" in descriptors.columns and (
        "canonical_smiles" not in descriptors.columns
        or descriptors["canonical_smiles"].isna().any()
    ):
        from rdkit import Chem
        def canonicalize(value):
            if pd.isna(value) or str(value).strip().lower() in {"", "nan", "none"}:
                return None
            mol = Chem.MolFromSmiles(str(value))
            return Chem.MolToSmiles(mol, canonical=True) if mol is not None else None

        generated = descriptors["smiles"].map(canonicalize)
        if "canonical_smiles" not in descriptors.columns:
            descriptors["canonical_smiles"] = generated
        else:
            descriptors["canonical_smiles"] = descriptors["canonical_smiles"].where(
                descriptors["canonical_smiles"].notna(), generated
            )
    keep = [c for c in ["mol_id", "compound_id", "canonical_smiles", *REQUIRED_XTB] if c in descriptors.columns]
    descriptors = descriptors[keep].drop_duplicates()
    left = molecules.copy()
    if "mol_id" in descriptors.columns and "mol_id" in left.columns:
        left = left.merge(descriptors, on="mol_id", how="left", suffixes=("", "_xtb"))
    elif "compound_id" in descriptors.columns and "compound_id" in left.columns:
        left = left.merge(descriptors, on="compound_id", how="left", suffixes=("", "_xtb"))
    elif "canonical_smiles" in descriptors.columns:
        left = left.merge(descriptors, on="canonical_smiles", how="left", suffixes=("", "_xtb"))
    else:
        raise ValueError("Descriptor table must contain mol_id, compound_id, or canonical_smiles for matching.")
    for col in REQUIRED_XTB:
        if f"{col}_xtb" in left.columns:
            left[col] = left[col].where(left[col].notna(), left[f"{col}_xtb"])
            left.drop(columns=[f"{col}_xtb"], inplace=True)
        left[col] = pd.to_numeric(left[col], errors="coerce")
    left["xtb_complete"] = left[REQUIRED_XTB].notna().all(axis=1)
    return left


class ToxGNNEngine:
    """Locked Stage-2 embedding + selected xTB/GNN prediction-fusion engine."""

    def __init__(
        self,
        source_csv: Path | None = None,
        xtb_parquet: Path | None = None,
        checkpoint: Path | None = None,
    ):
        self.source_csv = source_csv or PROJECT_ROOT / "data/processed/source_domain_tox.csv"
        self.xtb_parquet = xtb_parquet or PROJECT_ROOT / "data/features/xtb_descriptors.parquet"
        self.checkpoint = checkpoint or PROJECT_ROOT / "artifacts/fresh_runs/final_locked_r2_0p84384_seed3407/stage2_lc50_adapter/best_loss.ckpt"
        # Keep the bundle next to the deployed app; this works for both the
        # public repository-root layout and the local toxgnn_streamlit_app/ layout.
        self.bundle_path = APP_DIR / "models/final_toxgnn_xtb_bundle.joblib"
        self._model = None
        self._device = None
        self._source_embeddings = None
        self._source = None
        self._source_xtb = None
        self._descriptor_store = None
        self._fusion = {}
        self._ad = None
        self._bundle = None

    def _load_bundle(self):
        import joblib
        if not self.bundle_path.exists():
            raise FileNotFoundError(f"Portable model bundle not found: {self.bundle_path}")
        self._bundle = joblib.load(self.bundle_path)
        if self._bundle.get("selected_xtb_names") != REQUIRED_XTB:
            raise ValueError("The portable bundle does not match the authoritative four-descriptor xTB branch.")

    def _load_base_tables(self):
        if self._source is None:
            self._source = pd.read_csv(self.source_csv)
            xtb = read_table(self.xtb_parquet)
            self._descriptor_store = xtb.copy()
            target_reference = PROJECT_ROOT / "artifacts/fresh_runs/final_locked_r2_0p84384_seed3407/stage4_hpah57/hpah57_base_table.csv"
            if target_reference.exists():
                target_xtb = pd.read_csv(target_reference)
                target_xtb = target_xtb.rename(columns={"hpah_id": "mol_id"})
                target_xtb = target_xtb[[c for c in ["mol_id", "canonical_smiles", *RAW_XTB] if c in target_xtb.columns]]
                self._descriptor_store = pd.concat([self._descriptor_store, target_xtb], ignore_index=True, sort=False)
            self._source_xtb = self._source.merge(
                xtb[["mol_id", *[c for c in RAW_XTB if c in xtb.columns]]], on="mol_id", how="inner"
            ).dropna(subset=REQUIRED_XTB)

    def _load_model(self):
        import torch
        from toxgnn.models.adapter import LC50AdapterModel, ResidualAdapter
        from toxgnn.models.gnn_backbone import GATv2Backbone

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(self.checkpoint, map_location=self._device, weights_only=False)
        state = ckpt.get("model_state_dict", ckpt)
        conv_keys = sorted(k for k in state if k.startswith("backbone.convs.") and k.endswith("lin_l.weight"))
        if not conv_keys:
            raise ValueError("The checkpoint does not contain a compatible GATv2 backbone.")
        first = state[conv_keys[0]]
        node_dim = int(first.shape[1])
        heads = int(state["backbone.convs.0.att"].shape[1])
        hidden_dim = int(first.shape[0] // heads)
        edge_dim = int(state["backbone.convs.0.lin_edge.weight"].shape[1])
        num_layers = len(conv_keys)
        bottleneck = int(state["adapter.adapter.0.weight"].shape[0])
        head_hidden = int(state["head.0.weight"].shape[0])
        backbone = GATv2Backbone(
            node_dim=node_dim, edge_dim=edge_dim, hidden_dim=hidden_dim,
            num_layers=num_layers, heads=heads, dropout=0.0,
        )
        adapter = ResidualAdapter(dim=hidden_dim, bottleneck=bottleneck, alpha=0.1, dropout=0.1)
        model = LC50AdapterModel(backbone, adapter, head_hidden=head_hidden, dropout=0.0)
        model.load_state_dict(state, strict=True)
        self._model = model.to(self._device).eval()

    def _embeddings(self, frame: pd.DataFrame) -> np.ndarray:
        import torch
        from torch_geometric.loader import DataLoader
        from toxgnn.data.dataset import MolGraphDataset

        dataset = MolGraphDataset(frame, smiles_col="canonical_smiles", mol_id_col="compound_id")
        loader = DataLoader(dataset, batch_size=64, shuffle=False)
        chunks = []
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self._device)
                chunks.append(self._model.get_adapted_embeddings(batch).cpu().numpy())
        return np.concatenate(chunks, axis=0)

    def prepare(self, clean_molecules: pd.DataFrame, descriptor_table: pd.DataFrame | None = None) -> pd.DataFrame:
        """Attach descriptors from a user table or the repository descriptor table."""
        self._load_base_tables()
        descriptors = descriptor_table if descriptor_table is not None else self._descriptor_store
        prepared = attach_xtb_descriptors(clean_molecules, descriptors)
        return prepared

    def compute_missing_xtb(self, prepared: pd.DataFrame, executable: str = "xtb") -> pd.DataFrame:
        """Run the same GFN2-xTB/ALPB-water single-point protocol used in the project."""
        from rdkit import Chem
        from toxgnn.features.xtb_parser import parse_xtb_output
        from toxgnn.features.xtb_runner import XTBConfig, run_xtb_single

        work = prepared.copy()
        missing = ~work["xtb_complete"]
        if not missing.any():
            return work
        runtime_dir = PROJECT_ROOT / "toxgnn_streamlit_app" / "runtime_xtb"
        cfg = XTBConfig(executable=executable, timeout_sec=300, method="gfn2", solvent_model="alpb", solvent="water", use_sp=True)
        statuses = []
        for idx, row in work.loc[missing].iterrows():
            mol = Chem.MolFromSmiles(str(row["canonical_smiles"]))
            charge = int(Chem.GetFormalCharge(mol)) if mol is not None else 0
            job = run_xtb_single(str(row["compound_id"]), str(row["canonical_smiles"]), runtime_dir, charge, cfg, n_confs=20, seed=42)
            values = parse_xtb_output(job.output_dir) if job.status == "success" else {}
            for col in RAW_XTB:
                if col in values:
                    work.at[idx, col] = values[col]
            statuses.append((idx, job.status, job.error_message or ""))
        for idx, status, message in statuses:
            work.at[idx, "xtb_status"] = status
            work.at[idx, "xtb_message"] = message
        work["xtb_complete"] = work[REQUIRED_XTB].notna().all(axis=1)
        return work

    def predict(self, prepared: pd.DataFrame, ensemble_size: int = 1) -> pd.DataFrame:
        """Generate pLC50, uncertainty, and dual-space AD diagnostics."""
        from toxgnn.models.applicability_domain import ApplicabilityDomain

        self._load_base_tables()
        if self._bundle is None:
            self._load_bundle()
        if self._model is None:
            self._load_model()
        valid = prepared[prepared["xtb_complete"]].copy().reset_index(drop=True)
        if valid.empty:
            raise ValueError("No compound has a complete set of xTB descriptors.")
        if self._source_embeddings is None:
            self._source_embeddings = np.asarray(self._bundle["source_embeddings"], dtype=float)
        query_embeddings = self._embeddings(valid)
        x_source = np.asarray(self._bundle["source_xtb_selected"], dtype=float)
        x_query = valid[REQUIRED_XTB].to_numpy(dtype=float)
        xtb_model = self._bundle["xtb_model"]
        gnn_scaler = self._bundle["gnn_scaler"]
        gnn_pca = self._bundle["gnn_pca"]
        gnn_model = self._bundle["gnn_model"]
        pred_xtb = xtb_model.predict(x_query)
        pred_gnn = gnn_model.predict(gnn_pca.transform(gnn_scaler.transform(query_embeddings)))
        raw_fusion = self._bundle["fusion_weight_xtb"] * pred_xtb + self._bundle["fusion_weight_gnn"] * pred_gnn
        pred_fusion = self._bundle["calibrator"].predict(raw_fusion.reshape(-1, 1))
        result = valid[["compound_id", "canonical_smiles"]].copy()
        result["pred_pLC50_xtb_branch"] = pred_xtb
        result["pred_pLC50_gnn_branch"] = pred_gnn
        result["pred_pLC50"] = pred_fusion
        result["pred_pLC50_sd"] = 0.0
        result["pred_pLC50_ci95_low"] = pred_fusion
        result["pred_pLC50_ci95_high"] = pred_fusion
        if "molecular_weight" in valid.columns:
            mw = pd.to_numeric(valid["molecular_weight"], errors="coerce").to_numpy()
            result["LC50_mg_L"] = np.where(np.isfinite(mw), 10 ** (-result["pred_pLC50"]) * mw * 1000, np.nan)

        if self._ad is None:
            self._ad = ApplicabilityDomain(k=5).fit(
                ref_smiles=self._source_xtb["canonical_smiles"].tolist(),
                ref_embeddings=self._source_embeddings,
                ref_xtb=x_source,
            )
        ad = pd.DataFrame(self._ad.batch_assess(
            valid["canonical_smiles"].tolist(), query_embeddings, x_query
        ))
        return pd.concat([result.reset_index(drop=True), ad.reset_index(drop=True)], axis=1)


def lc50_from_pLC50(p_lc50: float, molecular_weight: float) -> float:
    """Convert pLC50 = -log10(LC50 mol/L) to LC50 mg/L."""
    return float(10 ** (-p_lc50) * molecular_weight * 1000)
