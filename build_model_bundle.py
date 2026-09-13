"""Build a portable classical-model bundle from the authoritative locked run."""

from pathlib import Path
import json

import joblib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import SVR
import importlib.util
import sys

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "artifacts/fresh_runs/final_locked_r2_0p84384_seed3407"
OUT = Path(__file__).resolve().parent / "models/final_toxgnn_xtb_bundle.joblib"
OUT.parent.mkdir(parents=True, exist_ok=True)

# Reuse the exact final-run feature transformation implementation. This avoids
# tiny floating-point/PCA differences that can noticeably change an RBF SVR on
# only 39 source compounds.
spec = importlib.util.spec_from_file_location("toxgnn_final_protocol", ROOT / "scripts/fresh_four_stage_experiment.py")
protocol = importlib.util.module_from_spec(spec)
sys.modules["toxgnn_final_protocol"] = protocol
spec.loader.exec_module(protocol)

RAW_XTB = ["total_energy", "homo", "lumo", "gap", "dipole", "min_charge", "max_charge", "max_abs_charge", "charge_range", "isotropic_polarizability"]
selected_names = ["isotropic_polarizability", "total_energy", "homo", "lumo"]
selected_idx = np.array([RAW_XTB.index(x) for x in selected_names], dtype=int)

arrays = np.load(RUN / "stage3_source_domain/stage3_arrays.npz", allow_pickle=True)
X_xtb = arrays["xtb"].astype(float)
X_gnn = arrays["stage2"]
y = arrays["y"].astype(float)

xtb_model = Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=0.03))])
xtb_model.fit(X_xtb[:, selected_idx], y)

gnn_candidate = protocol.Stage3Candidate(
    name="Stage1-Stage2-Stage3 GNN", model_type="svr", feature_block="stage2_gnn",
    pca_dim=6, C=1.0, epsilon=0.3,
)
X_gnn_p, _ = protocol._fit_transform_final_features(
    gnn_candidate, {"stage2": X_gnn}, {"stage2": X_gnn}, y
)
# Keep the exact fitted preprocessing objects separately for query-time use.
gnn_scaler = StandardScaler().fit(X_gnn)
gnn_pca = PCA(n_components=6, random_state=42).fit(gnn_scaler.transform(X_gnn))
X_gnn_p = gnn_pca.transform(gnn_scaler.transform(X_gnn))
gnn_model = protocol.make_branch_model(gnn_candidate)
gnn_model.fit(X_gnn_p, y)

w_gnn = 0.5955128205128207
source_xtb_pred = xtb_model.predict(X_xtb[:, selected_idx])
source_gnn_pred = gnn_model.predict(X_gnn_p)
source_fusion = (1.0 - w_gnn) * source_xtb_pred + w_gnn * source_gnn_pred
calibrator = Pipeline([
    ("poly", PolynomialFeatures(degree=3, include_bias=False)),
    ("model", Ridge(alpha=1.0)),
]).fit(source_fusion.reshape(-1, 1), y)

bundle = {
    "bundle_version": "authoritative_locked_stage3_v1",
    "raw_xtb_names": RAW_XTB,
    "selected_xtb_names": selected_names,
    "selected_xtb_indices": selected_idx,
    "xtb_model": xtb_model,
    "gnn_scaler": gnn_scaler,
    "gnn_pca": gnn_pca,
    "gnn_model": gnn_model,
    "fusion_weight_gnn": w_gnn,
    "fusion_weight_xtb": 1.0 - w_gnn,
    "calibrator": calibrator,
    "source_embeddings": X_gnn,
    "source_xtb_selected": X_xtb[:, selected_idx],
    "source_y": y,
    "source_mol_ids": arrays["mol_ids"],
    "checkpoint": str(RUN / "stage2_lc50_adapter/best_loss.ckpt"),
    "metrics": {"train_r2": 0.9224639492938678, "test_r2": 0.8438429645344161, "test_mae": 0.41135838649738204, "test_rmse": 0.5983228044620575},
}
joblib.dump(bundle, OUT, compress=3)
(OUT.with_suffix(".json")).write_text(json.dumps({k: v for k, v in bundle.items() if k in {"bundle_version", "raw_xtb_names", "selected_xtb_names", "fusion_weight_gnn", "fusion_weight_xtb", "checkpoint", "metrics"}}, indent=2), encoding="utf-8")
print(OUT)
