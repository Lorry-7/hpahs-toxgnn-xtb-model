"""QS5 fusion model: Optimal low-dim GNN + xTB fusion via residual-correlated feature selection."""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold, LeaveOneOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

SVR_PARAM_GRID = [
    {"svr__C": C, "svr__epsilon": eps, "svr__gamma": gamma}
    for C in [0.1, 1.0, 10.0, 100.0]
    for eps in [0.01, 0.05, 0.10, 0.20]
    for gamma in ["scale", 0.001, 0.01, 0.10, 1.0]
]

FUSION_WEIGHTS = np.round(np.linspace(0.0, 1.0, 11), 2)


def select_gnn_dims_by_residual_corr(
    X_gnn: np.ndarray,
    residuals: np.ndarray,
    n_dims: int = 5,
) -> np.ndarray:
    """Select top N GNN embedding dimensions by correlation with xTB residuals."""
    corrs = np.array([
        np.corrcoef(X_gnn[:, i], residuals)[0, 1]
        for i in range(X_gnn.shape[1])
    ])
    # Replace NaN with 0 (constant features)
    corrs = np.nan_to_num(corrs, nan=0.0)
    top_idx = np.argsort(np.abs(corrs))[::-1][:n_dims]
    return top_idx


@dataclass
class QS5FitResult:
    """Result of QS5 fusion model fitting."""

    graph_model: Pipeline
    xtb_model: Pipeline
    weight_graph: float
    graph_params: dict
    xtb_params: dict
    inner_mae: float


def _make_svr(params: dict) -> Pipeline:
    """Create SVR pipeline with given parameters."""
    pipe = Pipeline([("scaler", StandardScaler()), ("svr", SVR(kernel="rbf"))])
    pipe.set_params(**params)
    return pipe


def select_branch_model(
    X: np.ndarray,
    y: np.ndarray,
    seed: int,
    param_grid: list[dict] | None = None,
) -> tuple[Pipeline, dict, float]:
    """Select best SVR model via cross-validation."""
    if param_grid is None:
        param_grid = SVR_PARAM_GRID

    cv = KFold(n_splits=5, shuffle=True, random_state=seed) if len(y) >= 10 else LeaveOneOut()

    best_model, best_params, best_mae = None, None, float("inf")
    for params in param_grid:
        preds = np.zeros_like(y, dtype=float)
        for tr, va in cv.split(X, y):
            m = _make_svr(params)
            m.fit(X[tr], y[tr])
            preds[va] = m.predict(X[va])
        mae = mean_absolute_error(y, preds)
        if mae < best_mae:
            best_mae, best_params = mae, params

    best_model = _make_svr(best_params)
    best_model.fit(X, y)
    return best_model, best_params, best_mae


def fit_qs5_fusion(
    X_graph: np.ndarray,
    X_xtb: np.ndarray,
    y: np.ndarray,
    seed: int,
    param_grid: list[dict] | None = None,
    fusion_weights: np.ndarray | None = None,
) -> QS5FitResult:
    """Fit QS5 fusion model with LOOCV inner validation for weight selection."""
    if fusion_weights is None:
        fusion_weights = FUSION_WEIGHTS

    # Train branch models
    graph_model, graph_params, _ = select_branch_model(X_graph, y, seed, param_grid)
    xtb_model, xtb_params, _ = select_branch_model(X_xtb, y, seed + 101, param_grid)

    # Select fusion weight via inner CV
    cv = KFold(n_splits=5, shuffle=True, random_state=seed + 202) if len(y) >= 10 else LeaveOneOut()

    best_w, best_mae = 0.5, float("inf")
    for w in fusion_weights:
        pred = np.zeros_like(y, dtype=float)
        for tr, va in cv.split(X_graph, y):
            gm = _make_svr(graph_params).fit(X_graph[tr], y[tr])
            xm = _make_svr(xtb_params).fit(X_xtb[tr], y[tr])
            pred[va] = w * gm.predict(X_graph[va]) + (1.0 - w) * xm.predict(X_xtb[va])
        mae = mean_absolute_error(y, pred)
        if mae < best_mae:
            best_w, best_mae = float(w), float(mae)

    return QS5FitResult(
        graph_model=graph_model,
        xtb_model=xtb_model,
        weight_graph=best_w,
        graph_params=graph_params,
        xtb_params=xtb_params,
        inner_mae=best_mae,
    )


def predict_qs5(
    model: QS5FitResult,
    X_graph: np.ndarray,
    X_xtb: np.ndarray,
) -> np.ndarray:
    """Generate QS5 fusion predictions."""
    pg = model.graph_model.predict(X_graph)
    px = model.xtb_model.predict(X_xtb)
    return model.weight_graph * pg + (1.0 - model.weight_graph) * px


def loocv_qs5(
    X_graph: np.ndarray,
    X_xtb: np.ndarray,
    y: np.ndarray,
    seed: int,
    param_grid: list[dict] | None = None,
) -> tuple[np.ndarray, dict]:
    """Run LOOCV for QS5 fusion model.

    Returns (predictions, metrics).
    """
    n = len(y)
    preds = np.zeros(n)
    weights = np.zeros(n)

    loo = LeaveOneOut()
    for train_idx, test_idx in loo.split(X_graph, y):
        # Fit on training fold
        result = fit_qs5_fusion(
            X_graph[train_idx],
            X_xtb[train_idx],
            y[train_idx],
            seed=seed,
            param_grid=param_grid,
        )
        # Predict on test fold
        preds[test_idx] = predict_qs5(
            result, X_graph[test_idx], X_xtb[test_idx]
        )
        weights[test_idx] = result.weight_graph

    # Calculate metrics
    from sklearn.metrics import r2_score
    mae = float(mean_absolute_error(y, preds))
    rmse = float(np.sqrt(np.mean((y - preds) ** 2)))
    r2 = float(r2_score(y, preds))

    metrics = {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "mean_weight_graph": float(np.mean(weights)),
    }

    return preds, metrics


@dataclass
class ResidualFusionFitResult:
    """Result of residual-correlation-based fusion fitting."""
    model: Pipeline
    selected_gnn_dims: np.ndarray
    xtb_params: dict
    inner_mae: float


def fit_residual_fusion(
    X_gnn: np.ndarray,
    X_xtb: np.ndarray,
    y: np.ndarray,
    seed: int,
    n_gnn_dims: int = 5,
    param_grid: list[dict] | None = None,
) -> ResidualFusionFitResult:
    """Fit residual-correlation-based fusion: select GNN dims that complement xTB."""
    if param_grid is None:
        param_grid = SVR_PARAM_GRID

    # Step 1: Get xTB LOOCV predictions to compute residuals
    loo = LeaveOneOut()
    xtb_preds = np.zeros(len(y))
    for train_idx, test_idx in loo.split(X_xtb, y):
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_xtb[train_idx])
        X_test = scaler.transform(X_xtb[test_idx])
        svr = SVR(kernel='rbf', C=10.0, epsilon=0.05, gamma='scale')
        svr.fit(X_train, y[train_idx])
        xtb_preds[test_idx] = svr.predict(X_test)
    residuals = y - xtb_preds

    # Step 2: Select top GNN dims by correlation with residuals
    selected_dims = select_gnn_dims_by_residual_corr(X_gnn, residuals, n_gnn_dims)

    # Step 3: Concatenate and train SVR
    X_combined = np.hstack([X_xtb, X_gnn[:, selected_dims]])

    cv = KFold(n_splits=5, shuffle=True, random_state=seed) if len(y) >= 10 else LeaveOneOut()

    best_model, best_params, best_mae = None, None, float("inf")
    for params in param_grid:
        preds_cv = np.zeros_like(y, dtype=float)
        for tr, va in cv.split(X_combined, y):
            m = _make_svr(params)
            m.fit(X_combined[tr], y[tr])
            preds_cv[va] = m.predict(X_combined[va])
        mae = mean_absolute_error(y, preds_cv)
        if mae < best_mae:
            best_mae, best_params = mae, params

    best_model = _make_svr(best_params)
    best_model.fit(X_combined, y)

    return ResidualFusionFitResult(
        model=best_model,
        selected_gnn_dims=selected_dims,
        xtb_params=best_params,
        inner_mae=best_mae,
    )


def predict_residual_fusion(
    model: ResidualFusionFitResult,
    X_gnn: np.ndarray,
    X_xtb: np.ndarray,
) -> np.ndarray:
    """Generate residual fusion predictions."""
    X_combined = np.hstack([X_xtb, X_gnn[:, model.selected_gnn_dims]])
    return model.model.predict(X_combined)


def loocv_residual_fusion(
    X_gnn: np.ndarray,
    X_xtb: np.ndarray,
    y: np.ndarray,
    seed: int,
    n_gnn_dims: int = 5,
    param_grid: list[dict] | None = None,
) -> tuple[np.ndarray, dict]:
    """Run LOOCV for residual-correlation-based fusion.

    For each fold:
    1. Compute xTB residuals on training set
    2. Select top GNN dims by residual correlation
    3. Train SVR on concatenated features
    """
    n = len(y)
    preds = np.zeros(n)
    loo = LeaveOneOut()

    for train_idx, test_idx in loo.split(X_gnn, y):
        # Step 1: xTB SVR on training fold (inner LOOCV for residuals)
        xtb_inner_preds = np.zeros(len(train_idx))
        inner_loo = LeaveOneOut()
        for inner_tr, inner_va in inner_loo.split(X_xtb[train_idx]):
            inner_data = X_xtb[train_idx]
            scaler = StandardScaler()
            X_inner_train = scaler.fit_transform(inner_data[inner_tr])
            X_inner_test = scaler.transform(inner_data[inner_va])
            svr = SVR(kernel='rbf', C=10.0, epsilon=0.05, gamma='scale')
            svr.fit(X_inner_train, y[train_idx][inner_tr])
            xtb_inner_preds[inner_va] = svr.predict(X_inner_test)
        residuals_train = y[train_idx] - xtb_inner_preds

        # Step 2: Select GNN dims by residual correlation
        selected_dims = select_gnn_dims_by_residual_corr(
            X_gnn[train_idx], residuals_train, n_gnn_dims
        )

        # Step 3: Concatenate and train
        X_train_combined = np.hstack([X_xtb[train_idx], X_gnn[train_idx][:, selected_dims]])
        X_test_combined = np.hstack([X_xtb[test_idx], X_gnn[test_idx][:, selected_dims]])

        # SVR with param search
        if param_grid is None:
            grid = SVR_PARAM_GRID
        else:
            grid = param_grid

        cv_inner = KFold(n_splits=5, shuffle=True, random_state=seed) if len(train_idx) >= 10 else LeaveOneOut()
        best_params = grid[0]
        best_mae_inner = float("inf")
        for params in grid:
            preds_inner = np.zeros(len(train_idx))
            for tr2, va2 in cv_inner.split(X_train_combined, y[train_idx]):
                m = _make_svr(params)
                m.fit(X_train_combined[tr2], y[train_idx][tr2])
                preds_inner[va2] = m.predict(X_train_combined[va2])
            mae_inner = mean_absolute_error(y[train_idx], preds_inner)
            if mae_inner < best_mae_inner:
                best_mae_inner = mae_inner
                best_params = params

        final_model = _make_svr(best_params)
        final_model.fit(X_train_combined, y[train_idx])
        preds[test_idx] = final_model.predict(X_test_combined)

    from sklearn.metrics import r2_score
    mae = float(mean_absolute_error(y, preds))
    rmse = float(np.sqrt(np.mean((y - preds) ** 2)))
    r2 = float(r2_score(y, preds))

    metrics = {"mae": mae, "rmse": rmse, "r2": r2}
    return preds, metrics
