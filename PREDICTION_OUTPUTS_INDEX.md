# ToxGNN-xTB prediction-output index

The following files are the reference outputs generated with the locked
ToxGNN-xTB model. They correspond to the 57 target HPAHs and should be used
for checking the Streamlit inference results.

```text
toxgnn_streamlit_app/
├─ reference_predictions/
│  ├─ hpah57_predictions.csv       # 57 predictions, branch outputs, fusion pLC50, LC50
│  ├─ hpah57_base_table.csv         # target structures and raw GFN2-xTB descriptors
│  ├─ dual_space_ad_target_hpah57.csv # dual-space AD diagnostics
│  └─ README.txt
└─ models/
   ├─ final_toxgnn_xtb_bundle.joblib # portable inference bundle
   └─ final_toxgnn_xtb_bundle.json   # bundle metadata and locked metrics
```

Locked model metrics: training R2 = 0.922464; test R2 = 0.843843;
MAE = 0.411358; RMSE = 0.598323.

The four xTB inputs used by the final branch are
`isotropic_polarizability`, `total_energy`, `homo`, and `lumo`.
