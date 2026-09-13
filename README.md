# ToxGNN-xTB Prediction Platform

This Streamlit application provides a research-oriented interface for the
authoritative locked ToxGNN-xTB workflow in this repository. It accepts molecular
SMILES, performs RDKit validation and deduplication, automatically calculates
missing xTB descriptors with the project GFN2-xTB/ALPB-water single-point
protocol, generates pLC50 predictions, reports applicability-domain diagnostics,
and exports a complete result table.

## Run

From the project root:

```powershell
python -m pip install -r toxgnn_streamlit_app/requirements.txt
streamlit run toxgnn_streamlit_app/app.py
```

For new structures, install the GFN2-xTB executable separately and make sure
`xtb` is available on the system `PATH` (or enter its full path in the sidebar).

The default application paths point to the locked checkpoint, portable model
bundle, and source-domain files already stored in this project. The final xTB
branch uses four correlation-selected descriptors: `isotropic_polarizability`,
`total_energy`, `homo`, and `lumo`. The application never fabricates missing
quantum-chemical descriptors. It runs xTB for new structures and reports any
calculation failure explicitly.

The interface reports the manuscript-locked performance metadata (training
R2 = 0.922464; test R2 = 0.843843, MAE = 0.411358, RMSE = 0.598323). These
values are not recomputed from user input.

## Input format

Upload CSV or XLSX with at least one column named `smiles` (aliases accepted:
`SMILES`, `canonical_smiles`, `structure`). Optional columns are `compound_id`,
`mol_id`, and `molecular_weight`. A separate descriptor upload may contain
`mol_id` or `canonical_smiles` plus the four model descriptor columns.

The platform reports pLC50 on the model scale. When molecular weight is supplied,
it additionally reports the corresponding LC50 in mg/L using
`LC50(mg/L) = 10**(-pLC50) * MW * 1000`.

## Scientific safeguards

- Invalid SMILES and duplicate canonical structures are shown in the cleaning log.
- Missing xTB descriptors are not imputed for user compounds.
- Applicability-domain flags are based on structural similarity, graph-embedding
  distance, and xTB Mahalanobis distance relative to the source domain.
- Results outside the dual-space support should be treated as extrapolative and
  require expert review.
