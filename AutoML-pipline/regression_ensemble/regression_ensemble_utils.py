import sys
import os
project_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__),   
                 "..")                  
)
sys.path.append(project_root)
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from classification_ensemble.ensemble_utils import _as_np, _clean_xy, _slice_xy, _fresh_model

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, RepeatedKFold, GridSearchCV

def soft_voting_with_weights_regression(
    base_models,
    weight_list,
    X_test,
    y_test,
    verbose: bool = False,
    compute_rmsle: bool = True,
    clip_nonneg_for_rmsle: bool = True,
    return_per_model: bool = False,
):
    """
    加权集成（回归）：
    - 将多个已训练的回归模型按 LLM 给定的权重做加权平均。
    - 更稳健：输入校验、权重归一、预测形状统一、RMSLE 的非负性检查与可选裁剪。
    
    参数
    ----
    base_models : List[Regressor]
    weight_list : List[float] / np.ndarray
    X_test      : array-like
    y_test      : array-like (连续值)
    verbose     : 打印评估结果
    compute_rmsle : 是否计算 RMSLE
    clip_nonneg_for_rmsle : 计算 RMSLE 前是否将预测裁剪为非负（y<0 时仍不计算 RMSLE）
    return_per_model : 是否在返回字典中包含所有基模型的预测
    
    返回
    ----
    dict:
        {
            "ensemble_metrics": {"mae": ..., "rmse": ..., "r2": ..., "rmsle": ... or None},
            "ensemble_preds": np.ndarray,   # (n_samples,)
            "normalized_weights": np.ndarray,  # (n_models,)
            "per_model_preds": np.ndarray or None  # (n_models, n_samples) if return_per_model=True
        }
    """
    # ------ 基本校验 ------
    n_models = len(base_models)
    if n_models == 0:
        raise ValueError("base_models 为空。")

    w = np.asarray(weight_list, dtype=np.float64).ravel()
    if w.size != n_models:
        raise ValueError(f"权重数量({w.size})与模型数量({n_models})不一致。")

    # 负权重处理：默认不允许负权重，直接截断为0（也可以选择报错）
    if np.any(w < 0):
        if verbose:
            print("Warning: 存在负权重，已截断为 0。")
        w = np.maximum(w, 0.0)

    s = w.sum()
    if s <= 0:
        if verbose:
            print("Warning: 权重和为 0，回退为均匀权重。")
        w = np.ones(n_models, dtype=np.float64) / n_models
    else:
        w = w / s
    normalized_weights = w

    # ------ 收集预测并统一形状 ------
    y_test = np.asarray(y_test, dtype=np.float64).ravel()
    n_samples = y_test.shape[0]

    preds_list = []
    for i, m in enumerate(base_models):
        pred = np.asarray(m.predict(X_test), dtype=np.float64).ravel()
        if pred.shape[0] != n_samples:
            raise ValueError(f"第 {i} 个模型预测长度({pred.shape[0]})与 y_test({n_samples}) 不一致。")
        if not np.all(np.isfinite(pred)):
            raise ValueError(f"第 {i} 个模型预测存在 NaN/Inf。")
        preds_list.append(pred)

    per_model_preds = np.vstack(preds_list)  # (n_models, n_samples)

    # ------ 加权融合 ------
    ensemble_preds = normalized_weights @ per_model_preds  # (n_samples,)

    # ------ 评估指标 ------
    metrics = {}
    metrics["mae"] = float(mean_absolute_error(y_test, ensemble_preds))
    metrics["rmse"] = float(np.sqrt(mean_squared_error(y_test, ensemble_preds)))
    metrics["r2"] = float(r2_score(y_test, ensemble_preds))

    # RMSLE（仅当非负且用户要求时）
    rmsle_val = None
    if compute_rmsle:
        # y 必须非负；预测如果允许裁剪，则先裁为非负
        if np.all(y_test >= 0):
            preds_for_rmsle = ensemble_preds
            if clip_nonneg_for_rmsle:
                preds_for_rmsle = np.maximum(preds_for_rmsle, 0.0)
            # log1p对0安全，无需加epsilon；对极小负数也会报错，已做裁剪
            rmsle_val = float(
                np.sqrt(mean_squared_error(np.log1p(y_test), np.log1p(preds_for_rmsle)))
            )
        else:
            if verbose:
                print("Info: y_test 含负值，未计算 RMSLE（设为 None）。")
    metrics["rmsle"] = rmsle_val

    if verbose:
        print("\n✅ 回归任务评估结果：")
        print(f"MAE : {metrics['mae']:.6f}")
        print(f"RMSE: {metrics['rmse']:.6f}")
        print(f"R2  : {metrics['r2']:.6f}")
        if compute_rmsle:
            print(f"RMSLE: {metrics['rmsle']:.6f}" if metrics["rmsle"] is not None else "RMSLE: None")

    return {
        "ensemble_metrics": metrics,
        "ensemble_preds": ensemble_preds,
        "normalized_weights": normalized_weights,
        "per_model_preds": per_model_preds if return_per_model else None,
    }


def stacking_ensemble_regression(
    base_models,                 # 模型实例列表 或 工厂函数列表（每次调用返回新实例）
    X_train, y_train,
    X_test, y_test,
    weight_list,                 # 与 base_models 对齐的权重（非负；会归一化）
    n_folds: int = 5,
    meta_cv_repeats: int = 3,
    meta_alpha_grid = (1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 0.3, 1, 3, 10),
    scale_meta: bool = True,
    random_state: int = 42,
    verbose: bool = False,
    drop_x_na_in_train: bool = False,
    drop_x_na_in_test: bool = False,
    compute_rmsle: bool = True,
    clip_nonneg_for_rmsle: bool = True,
):
    """
    两层 Stacking（回归，Ridge 元模型）：
      1) KFold OOF 生成 train_meta；
      2) 在 train_meta 上用 RepeatedKFold + GridSearchCV 搜参（alpha）；
      3) 用全量 base 模型得到 test_meta，评估一次。

    返回：
      {
        "meta_model": best_meta,
        "stacking_metrics": {"mae":..., "rmse":..., "rmsle":..., "r2":...},
        "train_meta": (n_train, n_models),
        "test_meta":  (n_test,  n_models),
        "weights_used": 归一化后的权重,
        "meta_cv_best_params": ...,
        "meta_cv_best_score": ...   # 以 RMSE 形式返回（正数，越低越好）
      }
    """
    # ---------- 清洗与对齐 ----------
    X_train, y_train = _clean_xy(X_train, y_train, drop_x_na=drop_x_na_in_train, name="train")
    X_test,  y_test  = _clean_xy(X_test,  y_test,  drop_x_na=drop_x_na_in_test,  name="test")
    y_train = _as_np(y_train).ravel().astype(float)
    y_test  = _as_np(y_test).ravel().astype(float)

    if len(y_train) == 0 or len(y_test) == 0:
        raise ValueError("y_train 或 y_test 为空。")

    # ---------- 权重 ----------
    n_models = len(base_models)
    if weight_list is None or len(weight_list) != n_models:
        raise ValueError("weight_list 必须提供，且长度与 base_models 一致。")

    w = np.asarray(weight_list, dtype=float).ravel()
    if (w < 0).any() or not np.isfinite(w).all():
        raise ValueError("weight_list 需为非负且有限数。")
    s = w.sum()
    w = (w / s) if s > 0 else np.ones(n_models, dtype=float) / n_models
    if verbose:
        print(f"[Info] Using {n_models} base models; normalized weights: {np.round(w, 4)}")

    # ---------- Step 1: OOF 生成 train_meta ----------
    n_train = len(y_train)
    train_meta = np.zeros((n_train, n_models), dtype=float)

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    for m_idx, base in enumerate(base_models):
        if verbose:
            print(f"\n[Base-{m_idx+1}/{n_models}] Building OOF predictions ...")

        for fold_idx, (tr_idx, va_idx) in enumerate(kf.split(_as_np(X_train))):
            X_tr, y_tr = _slice_xy(X_train, y_train, tr_idx)
            X_va, _    = _slice_xy(X_train, y_train, va_idx)

            seed = (random_state + 1009 * (m_idx + 1) + 31 * (fold_idx + 1)) & 0x7fffffff
            model_f = _fresh_model(base, seed=seed)
            # 训练并预测
            model_f.fit(X_tr, y_tr)
            pred_va = np.asarray(model_f.predict(X_va), dtype=float).ravel()
            if not np.all(np.isfinite(pred_va)):
                if verbose:
                    print(f"[Warn] Base-{m_idx+1} fold-{fold_idx+1} 预测含 NaN/Inf，已替换为 0。")
                pred_va = np.nan_to_num(pred_va, nan=0.0, posinf=0.0, neginf=0.0)

            train_meta[va_idx, m_idx] = pred_va * w[m_idx]

            if verbose:
                print(f"  fold {fold_idx+1}/{n_folds} done.")

    # ---------- Step 2: 全量训练基模型 → test_meta ----------
    n_test = len(y_test)
    test_meta = np.zeros((n_test, n_models), dtype=float)
    for m_idx, base in enumerate(base_models):
        seed = (random_state + 2027 * (m_idx + 1)) & 0x7fffffff
        model_full = _fresh_model(base, seed=seed)
        model_full.fit(X_train, y_train)
        pred_te = np.asarray(model_full.predict(X_test), dtype=float).ravel()
        if not np.all(np.isfinite(pred_te)):
            if verbose:
                print(f"[Warn] Base-{m_idx+1} 全量预测含 NaN/Inf，已替换为 0。")
            pred_te = np.nan_to_num(pred_te, nan=0.0, posinf=0.0, neginf=0.0)
        test_meta[:, m_idx] = pred_te * w[m_idx]

    # ---------- Step 3: Ridge 元模型（CV 搜参） ----------
    steps = []
    if scale_meta:
        steps.append(("scaler", StandardScaler()))
    steps.append(("reg", Ridge(random_state=random_state)))
    pipe = Pipeline(steps)

    param_grid = {"reg__alpha": list(meta_alpha_grid)}
    rkf = RepeatedKFold(n_splits=n_folds, n_repeats=meta_cv_repeats, random_state=random_state)

    # scoring: neg_root_mean_squared_error（越高越好；取反得到 RMSE）
    gscv = GridSearchCV(
        pipe, param_grid=param_grid,
        scoring="neg_root_mean_squared_error",
        cv=rkf, n_jobs=-1, refit=True, verbose=0
    )
    gscv.fit(train_meta, y_train)
    best_meta = gscv.best_estimator_

    if verbose:
        best_rmse = -gscv.best_score_
        print(f"\n[Meta] Best alpha: {gscv.best_params_['reg__alpha']}, CV RMSE: {best_rmse:.6f}")

    # ---------- Step 4: 测试评估 ----------
    y_hat = best_meta.predict(test_meta).ravel()
    # 基础指标
    mae = float(mean_absolute_error(y_test, y_hat))
    rmse = float(np.sqrt(mean_squared_error(y_test, y_hat)))
    r2 = float(r2_score(y_test, y_hat))

    # RMSLE（仅在适用时计算）
    rmsle_val = None
    if compute_rmsle:
        if np.all(y_test >= 0):
            y_hat_r = np.maximum(y_hat, 0.0) if clip_nonneg_for_rmsle else y_hat
            rmsle_val = float(np.sqrt(mean_squared_error(np.log1p(y_test), np.log1p(y_hat_r))))
        elif verbose:
            print("Info: y_test 含负值，未计算 RMSLE（设为 None）。")

    metrics = {"mae": mae, "rmse": rmse, "rmsle": rmsle_val, "r2": r2}

    if verbose:
        print("\n✅ Evaluation on Test (Regression Stacking):")
        print(f"  MAE : {mae:.6f}")
        print(f"  RMSE: {rmse:.6f}")
        print(f"  R2  : {r2:.6f}")
        if compute_rmsle:
            print(f"  RMSLE: {rmsle_val:.6f}" if rmsle_val is not None else "  RMSLE: None")

    return {
        "meta_model": best_meta,
        "stacking_metrics": metrics,
        "train_meta": train_meta,
        "test_meta": test_meta,
        "weights_used": w,
        "meta_cv_best_params": gscv.best_params_,
        "meta_cv_best_score": float(-gscv.best_score_),  # 转成正的 RMSE
    }
