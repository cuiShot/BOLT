# -*- coding: utf-8 -*-
"""
Multi-class Ensemble Utilities
==============================
参考二分类版 `ensemble_utils.py` 的实现思路（权重软投票 + Stacking）
提供**多分类**场景下的两个入口函数：

- soft_voting_with_weights: 基于 LLM 权重的多分类软投票集成
- stacking_ensemble        : 多分类两层 Stacking（OOF 生成 meta 特征 + LogisticRegression(mn) 元模型）

仅保留上述两个对外函数；其它函数为内部工具。
"""
from __future__ import annotations

import copy
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional

from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score

# ============================ 内部工具 ============================ #

def _as_np(X):
    try:
        return X.values if hasattr(X, "values") else np.asarray(X)
    except Exception:
        return np.asarray(X)


def _slice_xy(X, y, idx):
    if hasattr(X, "iloc"):
        Xs = X.iloc[idx]
    else:
        Xs = X[idx]
    if hasattr(y, "iloc"):
        ys = y.iloc[idx]
    else:
        ys = y[idx]
    return Xs, ys


def _safe_sigmoid(z):
    z = np.clip(z, -50, 50)
    return 1.0 / (1.0 + np.exp(-z))


def _softmax(Z):
    Z = np.asarray(Z, dtype=float)
    Z = Z - np.max(Z, axis=1, keepdims=True)
    EZ = np.exp(Z)
    return EZ / np.clip(EZ.sum(axis=1, keepdims=True), 1e-12, None)


def _project_weights_nonneg(weights: np.ndarray) -> np.ndarray:
    w = np.asarray(weights, dtype=float)
    w[~np.isfinite(w)] = 0.0
    w[w < 0] = 0.0
    s = w.sum()
    if s <= 0:
        return np.ones_like(w) / (len(w) if len(w) > 0 else 1)
    return w / s


def _clean_xy(X, y, drop_x_na=False, name="train"):
    y_arr = _as_np(y).ravel()
    y_mask = np.isfinite(y_arr)
    if hasattr(pd, "isna"):
        y_mask = y_mask & (~pd.isna(y_arr))
    if drop_x_na:
        X_arr = _as_np(X)
        y_mask = y_mask & ~np.isnan(X_arr).any(axis=1)
    if hasattr(X, "loc") and hasattr(y, "loc"):
        X2 = X.loc[y_mask]
        y2 = y.loc[y_mask]
    else:
        X2 = _as_np(X)[y_mask]
        y2 = _as_np(y)[y_mask]
    return X2, y2


def _collect_global_classes(models: List, *ys) -> Tuple[np.ndarray, Dict]:
    classes = set()
    for y in ys:
        if y is not None:
            classes.update(list(np.unique(_as_np(y).ravel())))
    for m in models:
        if hasattr(m, "classes_"):
            try:
                classes.update(list(m.classes_))
            except Exception:
                pass
    classes = np.array(sorted(list(classes)))
    cls_index = {c: i for i, c in enumerate(classes)}
    return classes, cls_index


def _align_proba_to_global(model, X, classes: np.ndarray, cls_index: Dict, label_smoothing: float = 1e-3) -> np.ndarray:
    """将模型对 X 的输出对齐到全局类别空间，返回 (M, C) 概率矩阵。"""
    M, C = len(X), len(classes)
    P = np.zeros((M, C), dtype=float)

    # 先取硬预测以备兜底
    if hasattr(model, "predict"):
        y_pred = model.predict(X)
        y_pred_idx = np.vectorize(lambda c: cls_index[c])(y_pred)
    else:
        y_pred_idx = None

    if hasattr(model, "predict_proba"):
        p_local = np.asarray(model.predict_proba(X))
        # 一维：当作正类概率
        if p_local.ndim == 1:
            if hasattr(model, "classes_") and len(getattr(model, "classes_")) == 2:
                neg, pos = model.classes_[0], model.classes_[1]
                neg_i, pos_i = cls_index[neg], cls_index[pos]
                ppos = np.clip(p_local, 0.0, 1.0)
                P[:, pos_i] = ppos
                P[:, neg_i] = 1.0 - ppos
            else:
                # 异常：无 classes_ 或多类只给一列；把该概率给预测类，其它均匀分
                if y_pred_idx is None:
                    raise ValueError("Model provides 1D proba but has no predict/classes_.")
                maxp = np.clip(p_local, 0.0, 1.0)
                P[np.arange(M), y_pred_idx] = maxp
                remain = 1.0 - maxp
                if C > 1:
                    P += (remain / (C - 1))[:, None]
                    P[np.arange(M), y_pred_idx] -= remain / (C - 1)
        else:  # 二维 (M, C_m)
            if hasattr(model, "classes_"):
                for j, cls in enumerate(model.classes_):
                    P[:, cls_index[cls]] = p_local[:, j]
            else:
                # 无 classes_：将最大列映射到预测类，其余均匀
                if y_pred_idx is None:
                    raise ValueError("Model has proba but no classes_/predict; cannot align.")
                maxp = p_local.max(axis=1)
                P[np.arange(M), y_pred_idx] = maxp
                remain = 1.0 - maxp
                if C > 1:
                    P += (remain / (C - 1))[:, None]
                    P[np.arange(M), y_pred_idx] -= remain / (C - 1)
    elif hasattr(model, "decision_function"):
        df = np.asarray(model.decision_function(X))
        if df.ndim == 1:
            p1 = _safe_sigmoid(df)
            if hasattr(model, "classes_") and len(getattr(model, "classes_")) == 2:
                neg, pos = model.classes_[0], model.classes_[1]
                neg_i, pos_i = cls_index[neg], cls_index[pos]
                P[:, pos_i] = p1
                P[:, neg_i] = 1.0 - p1
            else:
                if y_pred_idx is None:
                    raise ValueError("decision_function 1D but cannot infer classes.")
                P[np.arange(M), y_pred_idx] = p1
                remain = 1.0 - p1
                if C > 1:
                    P += (remain / (C - 1))[:, None]
                    P[np.arange(M), y_pred_idx] -= remain / (C - 1)
        else:
            P = _softmax(df)
            # 若有 classes_，需写入相应列
            if hasattr(model, "classes_"):
                P2 = np.zeros((M, C), dtype=float)
                for j, cls in enumerate(model.classes_):
                    P2[:, cls_index[cls]] = P[:, j]
                P = P2
            else:
                # 无 classes_：按预测类对齐最大概率
                if y_pred_idx is None:
                    raise ValueError("decision_function multi-d but no classes_/predict.")
                maxp = P.max(axis=1)
                P = np.zeros((M, C), dtype=float)
                P[np.arange(M), y_pred_idx] = maxp
                remain = 1.0 - maxp
                if C > 1:
                    P += (remain / (C - 1))[:, None]
                    P[np.arange(M), y_pred_idx] -= remain / (C - 1)
    else:
        # 没有概率/分数：label smoothing one-hot
        eps = float(label_smoothing) if C > 1 else 0.0
        P[:] = eps
        if y_pred_idx is None:
            # 没有 predict 也无法对齐，只能平均
            P[:] = 1.0 / max(C, 1)
        else:
            P[np.arange(M), y_pred_idx] = 1.0 - (C - 1) * eps if C > 1 else 1.0

    # 数值保稳
    P = np.clip(P, 1e-12, 1.0)
    P /= P.sum(axis=1, keepdims=True)
    return P


def _metrics_multiclass(y_true_idx: np.ndarray, proba: np.ndarray) -> Dict[str, float]:
    y_pred_idx = np.argmax(proba, axis=1)
    out: Dict[str, float] = {}
    out["accuracy"] = accuracy_score(y_true_idx, y_pred_idx)
    out["f1"] = f1_score(y_true_idx, y_pred_idx, average="macro")
    out["precision"] = precision_score(y_true_idx, y_pred_idx, average="macro", zero_division=0)
    out["recall"] = recall_score(y_true_idx, y_pred_idx, average="macro")
    try:
        out["auc"] = roc_auc_score(y_true_idx, proba, multi_class="ovr", average="macro")
    except Exception:
        out["auc"] = np.nan
    return out


def _fresh_model(base, seed=None):
    m = base() if callable(base) else copy.deepcopy(base)
    if seed is not None:
        for attr in ("random_state", "seed"):
            if hasattr(m, attr):
                try:
                    setattr(m, attr, seed)
                except Exception:
                    pass
    return m


def _safe_logit_matrix(P: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    P = np.clip(P, eps, 1 - eps)
    return np.log(P / (1 - P))

# ===================== 1) 软投票（多分类） ===================== #

def soft_voting_with_weights(
    base_models: List,
    llm_weights: List[float],
    X_test,
    y_test,
    label_smoothing: float = 1e-3,
    verbose: bool = False,
) -> Dict:
    """
    多分类软投票（概率加权平均）：
      1) 统一**全局类别空间**；
      2) 收集各模型对齐后的 p(y|x)；
      3) 按 LLM 权重加权平均得到 ensemble 概率；
      4) 计算 macro 指标与 OVR AUC。

    返回：{ 'metrics', 'preds', 'proba', 'classes', 'weights_used' }
    """
    # 权重
    w = _project_weights_nonneg(np.asarray(llm_weights, dtype=float))

    # 全局类别
    classes, cls_index = _collect_global_classes(base_models, y_test)
    y_test_idx = np.vectorize(lambda c: cls_index[c])(np.asarray(y_test))

    # 收集概率
    P_list = []
    for m in base_models:
        P = _align_proba_to_global(m, X_test, classes, cls_index, label_smoothing=label_smoothing)
        P_list.append(P)

    # 加权平均
    P_stack = np.stack(P_list, axis=0)            # (N, M, C)
    proba_ens = np.tensordot(w, P_stack, axes=(0, 0))  # (M, C)
    proba_ens = np.clip(proba_ens, 1e-12, 1.0)
    proba_ens /= proba_ens.sum(axis=1, keepdims=True)

    preds_idx = np.argmax(proba_ens, axis=1)
    preds = classes[preds_idx]

    metrics = _metrics_multiclass(y_test_idx, proba_ens)

    if verbose:
        print("\n✅ 多分类软投票评估结果：")
        for k, v in metrics.items():
            if v is not None:
                print(f"  {k}: {v:.4f}")

    return {
        "voting_metrics": metrics,
        "preds": preds,
        "proba": proba_ens,
        "classes": classes,
        "weights_used": w,
    }

# ===================== 2) Stacking（多分类） ===================== #

def stacking_ensemble(
    base_models: List,                  # 模型实例或工厂函数；若为实例将通过 deepcopy 克隆
    X_train,
    y_train,
    X_test,
    y_test,
    weight_list: List[float],           # 与 base_models 对齐的权重
    n_folds: int = 5,
    meta_cv_repeats: int = 3,
    meta_C_grid: Tuple = (0.03, 0.1, 0.3, 1, 3, 10),
    use_logit: bool = True,             # 对 meta 特征做按列 logit 变换（避免概率边界效应）
    scale_meta: bool = True,            # 是否标准化 meta 特征
    class_weight: Optional[str] = "balanced",
    random_state: int = 42,
    verbose: bool = False,
    drop_x_na_in_train: bool = False,
    drop_x_na_in_test: bool = False,
    label_smoothing: float = 1e-3,
) -> Dict:
    """
    多分类两层 Stacking：
      - 第1层：对每个基模型做 K 折 OOF，收集**对齐到全局类别**的概率作为 meta 特征；每个模型贡献 C 列；
      - 第2层：在 train_meta 上用 RepeatedStratifiedKFold + GridSearchCV 搜 LogisticRegression(mn) 的 C；
      - 推断：全量重训基模型，生成 test_meta → 预测概率，计算 macro 指标与 OVR AUC。

    返回：{ 'meta_model', 'stacking_metrics', 'train_meta', 'test_meta', 'weights_used', 'classes', 'meta_cv_best_params', 'meta_cv_best_score' }
    """
    if weight_list is None or len(weight_list) != len(base_models):
        raise ValueError("weight_list 必须提供，且长度与 base_models 一致。")

    # 清洗 & 类别
    X_train, y_train = _clean_xy(X_train, y_train, drop_x_na=drop_x_na_in_train, name="train")
    X_test,  y_test  = _clean_xy(X_test,  y_test,  drop_x_na=drop_x_na_in_test,  name="test")

    classes, cls_index = _collect_global_classes(base_models, y_train, y_test)
    C = len(classes)
    y_train_idx = np.vectorize(lambda c: cls_index[c])(np.asarray(y_train))
    y_test_idx  = np.vectorize(lambda c: cls_index[c])(np.asarray(y_test))

    # 权重归一化
    n_models = len(base_models)
    weights = _project_weights_nonneg(np.asarray(weight_list, dtype=float))

    if verbose:
        print(f"[Info] Using {n_models} base models; normalized weights: {np.round(weights, 4)}; classes={list(classes)}")

    # ---------- Step 1: OOF → train_meta  ----------
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    train_meta = np.zeros((len(y_train_idx), n_models * C), dtype=float)

    for m_idx, base in enumerate(base_models):
        if verbose:
            print(f"\n[Base-{m_idx+1}/{n_models}] Building OOF probabilities ...")
        for fold_idx, (tr_idx, va_idx) in enumerate(skf.split(_as_np(X_train), y_train_idx)):
            X_tr, y_tr = _slice_xy(X_train, y_train, tr_idx)
            X_va, _    = _slice_xy(X_train, y_train, va_idx)

            seed = (random_state + 1009 * (m_idx + 1) + 31 * (fold_idx + 1)) & 0x7fffffff
            model_f = _fresh_model(base, seed=seed)
            model_f.fit(X_tr, y_tr)
            P_va = _align_proba_to_global(model_f, X_va, classes, cls_index, label_smoothing)
            # 写入该模型对应的 C 列
            start = m_idx * C
            train_meta[va_idx, start : start + C] = P_va * weights[m_idx]
            if verbose:
                print(f"  fold {fold_idx+1}/{n_folds} done.")

    # 数值兜底
    if np.isnan(train_meta).any() or ~np.isfinite(train_meta).all():
        if verbose:
            print("[Warn] train_meta 出现 NaN/inf，已用 1/C/0 替换（NaN->1/C，inf->0）。")
        train_meta = np.nan_to_num(train_meta, nan=1.0 / max(C, 1), posinf=0.0, neginf=0.0)

    # ---------- Step 2: 全量基模型 → test_meta ----------
    test_meta = np.zeros((len(y_test_idx), n_models * C), dtype=float)
    for m_idx, base in enumerate(base_models):
        seed = (random_state + 2027 * (m_idx + 1)) & 0x7fffffff
        model_full = _fresh_model(base, seed=seed)
        model_full.fit(X_train, y_train)
        P_te = _align_proba_to_global(model_full, X_test, classes, cls_index, label_smoothing)
        start = m_idx * C
        test_meta[:, start : start + C] = P_te * weights[m_idx]

    if np.isnan(test_meta).any() or ~np.isfinite(test_meta).all():
        if verbose:
            print("[Warn] test_meta 出现 NaN/inf，已用 1/C/0 替换（NaN->1/C，inf->0）。")
        test_meta = np.nan_to_num(test_meta, nan=1.0 / max(C, 1), posinf=0.0, neginf=0.0)

    # ---------- Step 3: 元模型（多项逻辑回归） ----------
    steps = []
    if use_logit:
        steps.append(("logit", FunctionTransformer(_safe_logit_matrix, validate=False)))
    if scale_meta:
        steps.append(("scaler", StandardScaler()))
    steps.append(("clf", LogisticRegression(
        multi_class="multinomial",
        solver="lbfgs",
        max_iter=10000,
        class_weight=class_weight,
        random_state=random_state,
    )))
    pipe = Pipeline(steps)

    param_grid = {"clf__C": list(meta_C_grid)}
    meta_cv = RepeatedStratifiedKFold(n_splits=n_folds, n_repeats=meta_cv_repeats, random_state=random_state)
    # 使用 sklearn 内置多分类 AUC 评分器（OVR）
    gscv = GridSearchCV(
        pipe,
        param_grid=param_grid,
        scoring="roc_auc_ovr",
        cv=meta_cv,
        n_jobs=-1,
        refit=True,
        verbose=0,
    )
    gscv.fit(train_meta, y_train_idx)
    best_meta = gscv.best_estimator_
    if verbose:
        print(f"\n[Meta] Best C: {gscv.best_params_['clf__C']}, CV AUC(ovr): {gscv.best_score_:.4f}")

    # ---------- Step 4: 测试评估 ----------
    proba_te = best_meta.predict_proba(test_meta)  # (M, C)
    preds_idx = np.argmax(proba_te, axis=1)

    metrics = _metrics_multiclass(y_test_idx, proba_te)

    if verbose:
        print("\n✅ Stacking 测试集评估：")
        for k, v in metrics.items():
            if v is not None:
                print(f"  {k}: {v:.4f}")

    return {
        "meta_model": best_meta,
        "stacking_metrics": metrics,
        "train_meta": train_meta,
        "test_meta": test_meta,
        "weights_used": weights,
        "classes": classes,
        "meta_cv_best_params": gscv.best_params_,
        "meta_cv_best_score": gscv.best_score_,
    }
