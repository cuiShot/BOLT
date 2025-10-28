import numpy as np
import pandas as pd
import warnings
from copy import deepcopy

from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, average_precision_score, balanced_accuracy_score,
    confusion_matrix, brier_score_loss
)

"""
新增参数 weight_handling（默认 "as_feature"），可选：

"ignore"：完全不用权重（让元模型自己学系数）；

"as_feature"：把你给的权重做加权平均 w·p 形成一列额外的 meta 特征，和每个基模型的原始概率一起喂给元模型（推荐做法）；

"scale_features"：按权重逐列缩放 meta 特征（不推荐，会扰动标度）；

"multiply_proba"：把权重直接乘到概率（保留兼容，但不推荐）。

这样你的 weight_list 就用在了新增的先验特征上（默认生效），元模型既能看到每个模型的独立输出，也能看到你主观加权后的“综合信号”，再用它自己的系数去学习谁更重要。
"""


# ------------------------------
# Helper utilities (self-contained)
# ------------------------------

def _as_np(X):
    if isinstance(X, (pd.DataFrame, pd.Series)):
        return X.values
    return np.asarray(X)


def _slice_xy(X, y, idx):
    if isinstance(X, (pd.DataFrame, pd.Series)):
        Xs = X.iloc[idx]
    else:
        Xs = X[idx]
    if isinstance(y, (pd.Series, pd.DataFrame)):
        ys = y.iloc[idx]
    else:
        ys = y[idx]
    return Xs, ys


def _set_random_state_if_any(model, seed):
    if hasattr(model, 'set_params'):
        try:
            model.set_params(random_state=seed)
        except Exception:
            pass
    return model


def _fresh_model(base, seed=42):
    """Return a fresh model instance.
    - If `base` is callable, call it with `seed` if supported, otherwise without args.
    - Else deepcopy the provided estimator and set random_state if available.
    """
    if callable(base):
        try:
            return base(random_state=seed)
        except Exception:
            try:
                return base()
            except Exception as e:
                raise ValueError(f"Model factory failed to instantiate: {e}")
    m = deepcopy(base)
    return _set_random_state_if_any(m, seed)


def _get_proba(model, X):
    if hasattr(model, 'predict_proba'):
        proba = model.predict_proba(X)
        if proba.ndim == 2 and proba.shape[1] >= 2:
            return proba[:, 1]
        return proba.ravel()
    if hasattr(model, 'decision_function'):
        z = model.decision_function(X)
        z = np.clip(z, -50, 50)
        return 1.0 / (1.0 + np.exp(-z))
    pred = model.predict(X)
    return _as_np(pred).astype(float)


def _safe_logit(p, eps=1e-6):
    p = np.asarray(p, dtype=float)
    p = np.clip(p, eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


def _clean_xy(X, y, drop_x_na=False, name="train"):
    if isinstance(X, pd.DataFrame):
        if drop_x_na:
            mask = ~X.isna().any(axis=1)
            if mask.sum() < len(mask):
                warnings.warn(f"[{name}] Dropped {len(mask) - mask.sum()} rows due to NaNs in X.")
            X = X.loc[mask]
            if isinstance(y, (pd.Series, pd.DataFrame)):
                y = y.loc[mask]
            else:
                y = _as_np(y)[mask]
        Xv = X.values
    else:
        Xv = _as_np(X)
        if drop_x_na:
            mask = ~np.isnan(Xv).any(axis=1)
            if mask.sum() < len(mask):
                warnings.warn(f"[{name}] Dropped {len(mask) - mask.sum()} rows due to NaNs in X.")
            Xv = Xv[mask]
            y = _as_np(y)[mask]
    yv = _as_np(y).ravel()
    return Xv, yv


def _encode_binary_y(y):
    y = _as_np(y).ravel()
    uniq = np.unique(y[~pd.isna(y)])
    if set(uniq.tolist()) <= {0, 1}:
        return y.astype(int)
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    if len(le.classes_) > 2:
        warnings.warn(f"Detected {len(le.classes_)} classes; proceeding with multi-class labels (0..K-1).")
    return y_enc.astype(int)


def _metric_from_name(name):
    name = name.lower()
    if name in ("acc", "accuracy"):
        return lambda yt, yp: accuracy_score(yt, yp)
    if name in ("bal_acc", "balanced_accuracy"):
        return lambda yt, yp: balanced_accuracy_score(yt, yp)
    if name in ("f1",):
        return lambda yt, yp: f1_score(yt, yp, zero_division=0)
    if name in ("precision", "prec"):
        return lambda yt, yp: precision_score(yt, yp, zero_division=0)
    if name in ("recall", "tpr", "sensitivity"):
        return lambda yt, yp: recall_score(yt, yp, zero_division=0)
    if name in ("youden", "j", "youden_j"):
        def youden(yt, yp):
            cm = confusion_matrix(yt, yp, labels=[0,1])
            tn, fp, fn, tp = cm.ravel()
            tpr = tp / (tp + fn) if (tp + fn) else 0.0
            tnr = tn / (tn + fp) if (tn + fp) else 0.0
            return tpr + tnr - 1.0
        return youden
    raise ValueError(f"Unsupported threshold metric: {name}")


def _best_threshold_via_oof(y_true, y_score, metric_name="accuracy", grid=None):
    if grid is None:
        grid = np.linspace(0.01, 0.99, 99)
    metric = _metric_from_name(metric_name)
    best_t, best_v = 0.5, -np.inf
    for t in grid:
        yp = (y_score >= t).astype(int)
        v = metric(y_true, yp)
        if v > best_v:
            best_v, best_t = v, t
    return float(best_t), float(best_v)


# ------------------------------
# Main function (improved Stacking with weight handling and bugfix)
# ------------------------------

def stacking_ensemble_v2(
    base_models,
    X_train, y_train,
    X_test, y_test,
    weight_list=None,
    weight_handling="as_feature",   # "ignore" | "as_feature" | "scale_features" | "multiply_proba"
    n_folds=5,
    meta_cv_repeats=3,
    meta_C_grid=(0.01, 0.03, 0.1, 0.3, 1, 3, 10),
    use_logit=False,
    scale_meta=True,
    class_weight=None,
    optimize_metric="accuracy",
    tune_threshold=True,
    threshold_metric="accuracy",
    random_state=42,
    verbose=False,
    drop_x_na_in_train=False,
    drop_x_na_in_test=False,
):
    """
    Stacking (LogisticRegression meta) with flexible use of `weight_list`.
    This version fixes a bug where an `enumerate` around CV split yielded 1D arrays passed to the pipeline.
    """

    # -- Clean & encode labels --
    X_train, y_train = _clean_xy(X_train, y_train, drop_x_na=drop_x_na_in_train, name="train")
    X_test,  y_test  = _clean_xy(X_test,  y_test,  drop_x_na=drop_x_na_in_test,  name="test")
    y_train = _encode_binary_y(y_train)
    y_test  = _encode_binary_y(y_test)
    if len(np.unique(y_train)) < 2:
        raise ValueError("y_train contains a single class; cannot train.")

    n_models = len(base_models)

    # Normalize weights if provided
    weights_used = None
    if weight_list is not None:
        w = np.asarray(weight_list, dtype=float)
        if len(w) != n_models:
            raise ValueError("Length of weight_list must equal number of base_models.")
        if not np.isfinite(w).all() or (w < 0).any():
            raise ValueError("weight_list must be non-negative and finite.")
        s = w.sum()
        weights_used = (w / s) if s > 0 else np.ones_like(w) / len(w)
        if verbose:
            print(f"[Info] weights (normalized): {np.round(weights_used, 4)}; handling = {weight_handling}")
    else:
        weight_handling = "ignore"

    # ---------------- Step 1: OOF meta features ----------------
    train_meta = np.zeros((len(y_train), n_models), dtype=float)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    for m_idx, base in enumerate(base_models):
        if verbose:
            print(f"\n[Base-{m_idx+1}/{n_models}] Building OOF predictions ...")
        for fold_idx, (tr_idx, va_idx) in enumerate(skf.split(_as_np(X_train), _as_np(y_train))):
            X_tr, y_tr = _slice_xy(X_train, y_train, tr_idx)
            X_va, _    = _slice_xy(X_train, y_train, va_idx)

            seed = (random_state + 1009 * (m_idx + 1) + 31 * (fold_idx + 1)) & 0x7fffffff
            model_f = _fresh_model(base, seed=seed)
            model_f.fit(X_tr, y_tr)
            proba_va = _get_proba(model_f, X_va)
            train_meta[va_idx, m_idx] = proba_va

            if verbose:
                print(f"  fold {fold_idx+1}/{n_folds} done.")

    if np.isnan(train_meta).any() or ~np.isfinite(train_meta).all():
        if verbose:
            print("[Warn] train_meta has NaN/inf; replacing (NaN->0.5, inf->0).")
        train_meta = np.nan_to_num(train_meta, nan=0.5, posinf=0.0, neginf=0.0)

    # ---------------- Step 2: Train base models on full train -> test_meta ----------------
    test_meta = np.zeros((len(y_test), n_models), dtype=float)
    base_models_fitted = []
    for m_idx, base in enumerate(base_models):
        seed = (random_state + 2027 * (m_idx + 1)) & 0x7fffffff
        model_full = _fresh_model(base, seed=seed)
        model_full.fit(X_train, y_train)
        base_models_fitted.append(model_full)
        proba_te = _get_proba(model_full, X_test)
        test_meta[:, m_idx] = proba_te

    if np.isnan(test_meta).any() or ~np.isfinite(test_meta).all():
        if verbose:
            print("[Warn] test_meta has NaN/inf; replacing (NaN->0.5, inf->0).")
        test_meta = np.nan_to_num(test_meta, nan=0.5, posinf=0.0, neginf=0.0)

    # ---------------- Step 3: Apply weight handling on meta features ----------------
    weight_handling_used = weight_handling
    if weights_used is not None:
        if weight_handling == "as_feature":
            tr_wavg = (train_meta * weights_used.reshape(1, -1)).sum(axis=1, keepdims=True)
            te_wavg = (test_meta  * weights_used.reshape(1, -1)).sum(axis=1, keepdims=True)
            train_meta = np.hstack([train_meta, tr_wavg])
            test_meta  = np.hstack([test_meta,  te_wavg])
        elif weight_handling == "scale_features" or weight_handling == "multiply_proba":
            if weight_handling == "multiply_proba":
                warnings.warn("weight_handling='multiply_proba' distorts calibration; prefer 'as_feature'.")
            train_meta = train_meta * weights_used.reshape(1, -1)
            test_meta  = test_meta  * weights_used.reshape(1, -1)
        elif weight_handling == "ignore":
            pass
        else:
            warnings.warn(f"Unknown weight_handling='{weight_handling}', falling back to 'ignore'.")
            weight_handling_used = "ignore"

    # ---------------- Step 4: Meta model with GridSearch ----------------
    steps = []
    if use_logit:
        steps.append(("logit", FunctionTransformer(_safe_logit, validate=False)))
    if scale_meta:
        steps.append(("scaler", StandardScaler()))
    steps.append(("clf", LogisticRegression(
        solver="lbfgs", max_iter=10000, class_weight=class_weight, random_state=random_state
    )))
    pipe = Pipeline(steps)

    scoring = optimize_metric
    param_grid = {"clf__C": list(meta_C_grid)}
    meta_cv = RepeatedStratifiedKFold(n_splits=n_folds, n_repeats=meta_cv_repeats, random_state=random_state)
    gscv = GridSearchCV(pipe, param_grid=param_grid, scoring=scoring, cv=meta_cv, n_jobs=-1, refit=True, verbose=0)
    gscv.fit(train_meta, y_train)

    best_meta = gscv.best_estimator_
    if verbose:
        print(f"\n[Meta] Best params: {gscv.best_params_}, CV score ({optimize_metric}): {gscv.best_score_:.4f}")

    # ---------------- Step 5: Threshold tuning on OOF ----------------
    skf2 = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    oof_meta = np.zeros_like(y_train, dtype=float)
    for tr_idx, va_idx in skf2.split(train_meta, y_train):  # <-- FIX: no enumerate here
        m = clone(best_meta)
        m.fit(train_meta[tr_idx], y_train[tr_idx])
        oof_meta[va_idx] = m.predict_proba(train_meta[va_idx])[:, 1]

    if tune_threshold:
        best_t, best_val = _best_threshold_via_oof(y_train, oof_meta, metric_name=threshold_metric)
    else:
        best_t, best_val = 0.5, np.nan

    if verbose:
        print(f"[Meta] Chosen threshold (by {threshold_metric}): t* = {best_t:.3f}  (CV={best_val:.4f})")

    # ---------------- Step 6: Final evaluation on test ----------------
    final_proba = best_meta.predict_proba(test_meta)[:, 1]
    final_pred  = (final_proba >= best_t).astype(int)

    metrics = {}
    try:
        metrics["roc_auc"] = roc_auc_score(y_test, final_proba)
    except Exception:
        metrics["roc_auc"] = np.nan
    try:
        metrics["pr_auc"] = average_precision_score(y_test, final_proba)
    except Exception:
        metrics["pr_auc"] = np.nan

    metrics.update({
        "accuracy": accuracy_score(y_test, final_pred),
        "balanced_accuracy": balanced_accuracy_score(y_test, final_pred),
        "f1": f1_score(y_test, final_pred, zero_division=0),
        "precision": precision_score(y_test, final_pred, zero_division=0),
        "recall": recall_score(y_test, final_pred, zero_division=0),
        "brier": brier_score_loss(y_test, final_proba),
    })

    cm = confusion_matrix(y_test, final_pred, labels=[0, 1])

    per_model_coef = None
    try:
        clf = best_meta.named_steps.get("clf", None)
        if clf is not None and hasattr(clf, "coef_"):
            per_model_coef = clf.coef_.ravel()
    except Exception:
        per_model_coef = None

    return {
        "meta_model": best_meta,
        "stacking_metrics": metrics,
        "train_meta": train_meta,
        "test_meta": test_meta,
        "chosen_threshold": best_t,
        "weights_used": weights_used,
        "weight_handling_used": weight_handling_used,
        "meta_cv_best_params": gscv.best_params_,
        "meta_cv_best_score": gscv.best_score_,
        "confusion_matrix": cm,
        "per_model_coef": per_model_coef,
        "base_models_fitted": base_models_fitted,
    }
