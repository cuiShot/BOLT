import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score


def get_metaModel(meta_model_name):
    """
    根据模型名称列表返回对应的已初始化分类模型列表。"""

    from sklearn.ensemble import (
        RandomForestClassifier,
        BaggingClassifier
    )
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.tree import DecisionTreeClassifier
    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier
    from catboost import CatBoostClassifier

    model_dict = {
        'RandomForestClassifier': RandomForestClassifier(n_estimators=100, random_state=42),
        'XGBClassifier': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
        'LGBMClassifier': LGBMClassifier(random_state=42),
        'CatBoostClassifier': CatBoostClassifier(verbose=0, random_state=42),
        'SVC': SVC(probability=True, random_state=42),
        'DecisionTreeClassifier': DecisionTreeClassifier(random_state=42),
        'LogisticRegression': LogisticRegression(random_state=42,max_iter=300),
        'BaggingClassifier': BaggingClassifier(random_state=42)
    }

    if meta_model_name not in model_dict:
        raise ValueError(f"模型名称 '{meta_model_name}' 不被支持，请检查拼写或补充定义。")
    return model_dict[meta_model_name]

def get_regression_metaModel(modelNameList):
    """
    根据模型名称列表返回对应的已初始化回归模型列表。

    支持的模型名称包括：
    - 'RandomForestRegressor'
    - 'XGBRegressor'
    - 'LGBMRegressor'
    - 'CatBoostRegressor'
    - 'SVR'
    - 'DecisionTreeRegressor'
    - 'LinearRegression'
    - 'Ridge'
    - 'Lasso'
    - 'ElasticNet'
    - 'MLPRegressor'
    - 'KNeighborsRegressor'
    - 'BaggingRegressor'

    参数：
    - modelNameList (List[str]): 模型名称字符串列表

    返回：
    - List[object]: 已初始化的回归模型实例列表
    """

    from sklearn.ensemble import (
        RandomForestRegressor,
        BaggingRegressor
    )
    from sklearn.linear_model import (
        LinearRegression,
        Ridge,
        Lasso,
        ElasticNet
    )
    from sklearn.svm import SVR
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.neural_network import MLPRegressor
    from xgboost import XGBRegressor
    from lightgbm import LGBMRegressor
    from catboost import CatBoostRegressor

    model_dict = {
        'RandomForestRegressor': RandomForestRegressor(n_estimators=100, random_state=42),
        'XGBRegressor': XGBRegressor(objective='reg:squarederror', random_state=42),
        'LGBMRegressor': LGBMRegressor(random_state=42),
        'CatBoostRegressor': CatBoostRegressor(verbose=0, random_state=42),
        'SVR': SVR(),
        'DecisionTreeRegressor': DecisionTreeRegressor(random_state=42),
        'LinearRegression': LinearRegression(),
        'Ridge': Ridge(random_state=42),
        'Lasso': Lasso(random_state=42),
        'ElasticNet': ElasticNet(random_state=42),
        'MLPRegressor': MLPRegressor(random_state=42, max_iter=1000),
        'KNeighborsRegressor': KNeighborsRegressor(),
        'BaggingRegressor': BaggingRegressor(random_state=42)
    }

    models = []
    for name in modelNameList:
        if name not in model_dict:
            raise ValueError(f"模型名称 '{name}' 不被支持，请检查拼写或补充定义。")
        models.append(model_dict[name])

    return models

def weighted_ensemble_with_llm_weights(base_models, llm_weights, X_test, y_test, task_type="classification", verbose=False):
    """
    使用权重对基础模型进行集成，并返回标准化评估结果
    
    参数:
    - base_models: 训练好的基础模型列表
    - llm_weights: LLM生成的权重列表，与基础模型一一对应
    - X_test: 测试特征数据
    - y_test: 测试标签
    - task_type: 任务类型，"classification"或"regression"
    - verbose: 是否打印评估结果
    
    返回:
    - 包含集成预测和评估指标的字典
    """
    # 标准化权重
    normalized_weights = np.array(llm_weights) / np.sum(llm_weights)
    
    # 收集所有模型的预测结果
    all_predictions = []
    all_proba = []  # 存储概率预测用于AUC计算
    for model in base_models:
        if task_type == "classification":
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(X_test)
                # 新增：校验 proba 维度，若为一维，重塑为 (n_samples, 1)
                if len(proba.shape) == 1:
                    proba = proba.reshape(-1, 1)
                # 二分类取正类概率
                if proba.shape[1] > 1:
                    all_proba.append(proba[:, 1])
                    preds = np.argmax(proba, axis=1)
                else:
                    all_proba.append(proba.flatten())
                    preds = (proba > 0.5).astype(int).flatten()
            else:
                preds = model.predict(X_test)
                all_proba.append(preds)  # 无概率输出时用预测值替代
            all_predictions.append(preds)
        else:
            preds = model.predict(X_test)
            all_predictions.append(preds)
    
    # 基于LLM权重进行加权融合
    ensemble_proba = np.average(all_proba, axis=0, weights=normalized_weights) if all_proba else None
    ensemble_preds = np.round(ensemble_proba).astype(int) if task_type == "classification" else \
                     np.average(all_predictions, axis=0, weights=normalized_weights)
    
    # 计算评估指标
    metrics = {}
    if task_type == "classification":
        metrics = {
            "accuracy": accuracy_score(y_test, ensemble_preds),
            "f1": f1_score(y_test, ensemble_preds),
            "precision": precision_score(y_test, ensemble_preds),
            "recall": recall_score(y_test, ensemble_preds),
            "auc": roc_auc_score(y_test, ensemble_proba) if ensemble_proba is not None else None
        }
    
    if verbose:
        print("\n✅ 评估结果:")
        for k, v in metrics.items():
            if v is not None:
                print(f"{k.capitalize()}: {v:.4f}")
    
    return {
        "ensemble_metrics": metrics,
        "ensemble_preds": ensemble_preds,
        "ensemble_proba": ensemble_proba
    }

#
def weighted_voting_with_llm_weights(base_models, llm_weights, X_test, y_test, verbose=False):
    """基于LLM权重的加权投票集成（带标准评估结果格式）"""
    # 获取所有模型的硬标签预测
    all_labels = []
    for model in base_models:
        all_labels.append(model.predict(X_test))
    
    # 转换为数组便于处理 (模型数, 样本数)
    label_array = np.array(all_labels)
    num_samples = label_array.shape[1]
    ensemble_preds = []
    
    # 对每个样本应用加权投票
    for i in range(num_samples):
        label_weights = {}
        for label, weight in zip(label_array[:, i], llm_weights):
            if label not in label_weights:
                label_weights[label] = 0
            label_weights[label] += weight
        ensemble_preds.append(max(label_weights, key=label_weights.get))
    
    ensemble_preds = np.array(ensemble_preds)
    
    # 计算概率（用于AUC）- 取加权平均概率
    all_proba = []
    for model in base_models:
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(X_test)
            # 处理不同维度的概率输出
            if len(proba.shape) == 1:
                # 一维数组直接使用
                all_proba.append(proba)
            elif proba.shape[1] > 1:
                # 多维数组取正类概率
                all_proba.append(proba[:, 1])
            else:
                # 单类概率的二维数组展平
                all_proba.append(proba.flatten())
    
    # 计算加权平均概率
    ensemble_proba = None
    if all_proba:
        normalized_weights = np.array(llm_weights) / np.sum(llm_weights)
        ensemble_proba = np.average(all_proba, axis=0, weights=normalized_weights)
    
    # 计算评估指标
    metrics = {
        "accuracy": accuracy_score(y_test, ensemble_preds),
        "f1": f1_score(y_test, ensemble_preds),
        "precision": precision_score(y_test, ensemble_preds),
        "recall": recall_score(y_test, ensemble_preds),
        "auc": roc_auc_score(y_test, ensemble_proba) if ensemble_proba is not None else None
    }
    
    if verbose:
        print("\n✅ 评估结果:")
        for k, v in metrics.items():
            if v is not None:
                print(f"{k.capitalize()}: {v:.4f}")
    
    return {
        "voting_metrics": metrics,
        "voting_preds": ensemble_preds,
        "voting_proba": ensemble_proba
    }

def tiered_ensemble_with_llm_weights(base_models, llm_weights, X_test, y_test, threshold=0.5, verbose=False):
    """分层集成（带标准评估结果格式）"""
    # 按权重排序模型
    sorted_indices = np.argsort(llm_weights)[::-1]
    sorted_models = [base_models[i] for i in sorted_indices]
    sorted_weights = [llm_weights[i] for i in sorted_indices]
    
    # 高权重模型
    primary_model = sorted_models[0]
    primary_proba = None
    if hasattr(primary_model, 'predict_proba'):
        proba = primary_model.predict_proba(X_test)
        # 处理不同维度的概率输出
        if len(proba.shape) == 1:
            primary_proba = proba
        elif proba.shape[1] > 1:
            primary_proba = proba[:, 1]
        else:
            primary_proba = proba.flatten()
    
    primary_preds = primary_model.predict(X_test)
    
    # 其他模型作为辅助
    secondary_models = sorted_models[1:]
    ensemble_preds = primary_preds.copy()
    ensemble_proba = primary_proba.copy() if primary_proba is not None else None
    
    if secondary_models and primary_proba is not None:
        # 确定不确定样本
        uncertain_mask = (primary_proba >= (0.5 - threshold/2)) & (primary_proba <= (0.5 + threshold/2))
        
        if np.any(uncertain_mask):
            uncertain_X = X_test[uncertain_mask]
            secondary_preds = [model.predict(uncertain_X) for model in secondary_models]
            secondary_proba = []
            
            for model in secondary_models:
                if hasattr(model, 'predict_proba'):
                    proba = model.predict_proba(uncertain_X)
                    # 处理不同维度的概率输出
                    if len(proba.shape) == 1:
                        secondary_proba.append(proba)
                    elif proba.shape[1] > 1:
                        secondary_proba.append(proba[:, 1])
                    else:
                        secondary_proba.append(proba.flatten())
            
            # 辅助模型权重标准化
            secondary_weights = sorted_weights[1:]
            secondary_weights = np.array(secondary_weights) / np.sum(secondary_weights)
            
            # 融合辅助模型结果
            secondary_ensemble_preds = np.average(secondary_preds, axis=0, weights=secondary_weights)
            secondary_ensemble_preds = np.round(secondary_ensemble_preds).astype(int)
            
            # 更新预测结果
            ensemble_preds[uncertain_mask] = secondary_ensemble_preds
            if secondary_proba:
                ensemble_proba[uncertain_mask] = np.average(secondary_proba, axis=0, weights=secondary_weights)
    
    # 计算评估指标
    metrics = {
        "accuracy": accuracy_score(y_test, ensemble_preds),
        "f1": f1_score(y_test, ensemble_preds),
        "precision": precision_score(y_test, ensemble_preds),
        "recall": recall_score(y_test, ensemble_preds),
        "auc": roc_auc_score(y_test, ensemble_proba) if ensemble_proba is not None else None
    }
    
    if verbose:
        print("\n✅ 评估结果:")
        for k, v in metrics.items():
            if v is not None:
                print(f"{k.capitalize()}: {v:.4f}")
    
    return {
        "tiered_metrics": metrics,
        "tiered_preds": ensemble_preds,
        "tiered_proba": ensemble_proba
    }
from sklearn.model_selection import StratifiedKFold


# TODO meta_model 获取和参数搜索
def stacking_ensemble_copy(base_models, meta_model, X_train, y_train, X_test, y_test, n_folds=5, weight_list=None, verbose=True, random_state=42):
    """
    执行 Stacking 集成学习，直接根据给定的权重进行加权。

    参数：
        base_models: list of model instances，必须已实例化，但尚未 fit。
        meta_model: sklearn 风格模型实例。
        X_train, y_train: 训练集。
        X_test, y_test: 测试集。
        weight_list: list of float，基础模型的权重（长度应与 base_models 一致）。
        n_folds: int，交叉验证的折数。
        verbose: bool，是否输出详细日志。
        random_state: int，随机种子。

    返回：
        dict，包含最终评估结果。
    """
    
    if weight_list is None or len(weight_list) != len(base_models):
        raise ValueError("weight_list must be provided and have the same length as base_models.")
    
    n_models = len(base_models)
    train_meta = np.zeros((X_train.shape[0], n_models))
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    # Step 1: 训练一级模型，构建 meta 特征
    for model_idx, model in enumerate(base_models):
        if verbose:
            print(f"\nTraining base model {model_idx + 1}/{n_models}")

        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
            X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_fold_train = y_train.iloc[train_idx]

            model.fit(X_fold_train, y_fold_train)
            val_pred_proba = model.predict_proba(X_fold_val)

            if len(val_pred_proba.shape) == 1 or val_pred_proba.shape[1] == 1:
                train_meta[val_idx, model_idx] = val_pred_proba.ravel() * weight_list[model_idx]  # 加权
            else:
                train_meta[val_idx, model_idx] = val_pred_proba[:, 1] * weight_list[model_idx]  # 加权

    # Step 2: 训练二层元模型
    if verbose:
        print("\nTraining meta model...")
    meta_model.fit(train_meta, y_train)

    # Step 3: 测试集集成预测
    test_meta = np.zeros((X_test.shape[0], n_models))
    for model_idx, model in enumerate(base_models):
        model.fit(X_train, y_train)
        test_pred_proba = model.predict_proba(X_test)

        if len(test_pred_proba.shape) == 1 or test_pred_proba.shape[1] == 1:
            test_meta[:, model_idx] = test_pred_proba.ravel() * weight_list[model_idx]  # 加权
        else:
            test_meta[:, model_idx] = test_pred_proba[:, 1] * weight_list[model_idx]  # 加权

    final_preds = meta_model.predict(test_meta)
    final_proba = meta_model.predict_proba(test_meta)[:, 1]

    # Step 4: 评估
    metrics = {
        "accuracy": accuracy_score(y_test, final_preds),
        "f1": f1_score(y_test, final_preds),
        "precision": precision_score(y_test, final_preds),
        "recall": recall_score(y_test, final_preds),
        "auc": roc_auc_score(y_test, final_proba)
    }

    if verbose:
        print("\n✅ Evaluation Results:")
        for k, v in metrics.items():
            print(f"{k.capitalize()}: {v:.4f}")

    return {
        "meta_model": meta_model,
        "stacking_metrics": metrics
    }
 

import numpy as np
import pandas as pd
import copy
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import LabelEncoder

# ---------- utils ----------
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

def _get_proba(model, X):
    # 统一取“正类概率”
    if hasattr(model, "predict_proba"):
        p = model.predict_proba(X)
        return p[:, 1] if p.ndim == 2 else p.ravel()
    elif hasattr(model, "decision_function"):
        s = np.asarray(model.decision_function(X)).ravel()
        return _safe_sigmoid(s)
    else:
        yhat = np.asarray(model.predict(X)).ravel()
        if set(np.unique(yhat)).issubset({-1, 1}):
            yhat = (yhat + 1) / 2.0
        return yhat.astype(float)

def _safe_logit(P, eps=1e-6):
    P = np.clip(P, eps, 1 - eps)
    return np.log(P / (1 - P))

def _clean_xy(X, y, drop_x_na=False, name="train"):
    # 清 y 的 NaN/inf，并可选清 X 的 NaN 行；保持索引对齐
    y_arr = _as_np(y).ravel()
    y_finite_mask = np.isfinite(y_arr)
    y_notna_mask = ~pd.isna(y_arr)
    mask = y_finite_mask & y_notna_mask
    if drop_x_na:
        X_arr = _as_np(X)
        mask = mask & ~np.isnan(X_arr).any(axis=1)
    if hasattr(X, "loc") and hasattr(y, "loc"):
        X2 = X.loc[mask]
        y2 = y.loc[mask]
    else:
        X2 = _as_np(X)[mask]
        y2 = _as_np(y)[mask]
    dropped = len(y) - len(y2)
    if dropped > 0:
        print(f"[clean] Dropped {dropped} rows from {name}.")
    return X2, y2

def _encode_binary_y(y):
    y_np = _as_np(y).ravel()
    uniq = np.unique(y_np[~pd.isna(y_np)])
    if set(uniq).issubset({-1, 1}):
        y_np = (y_np + 1) / 2.0
    uniq = np.unique(y_np[~pd.isna(y_np)])
    if not set(uniq).issubset({0, 1}):
        le = LabelEncoder()
        y_np = le.fit_transform(y_np)
        if len(le.classes_) != 2:
            raise ValueError(f"y is not binary after encoding. Classes={list(le.classes_)}")
    return y_np.astype(int)

def _fresh_model(base_model, seed=None):
    """
    优先用工厂函数创建新实例；否则 deepcopy。
    - 若 base_model 是可调用（工厂/lambda/class），调用得到全新实例；
    - 否则 copy.deepcopy(base_model)。
    然后尽力设置常见随机种子属性（如存在）。
    """
    m = base_model() if callable(base_model) else copy.deepcopy(base_model)
    # 尝试设置常见 seed 属性（若黑箱里有就改，没有就跳过）
    if seed is not None:
        for attr in ("random_state", "seed"):
            if hasattr(m, attr):
                try:
                    setattr(m, attr, seed)
                except Exception:
                    pass
    return m

def stacking_ensemble(
    base_models,                 # 支持两种输入：1) 模型实例列表；2) 工厂函数列表（每次调用返回新实例）
    X_train, y_train,
    X_test, y_test,
    weight_list,
    n_folds=5,
    meta_cv_repeats=3,
    meta_C_grid=(0.01, 0.03, 0.1, 0.3, 1, 3, 10),
    use_logit=True,
    scale_meta=True,
    class_weight="balanced",
    random_state=42,
    verbose=False,
    drop_x_na_in_train=False,
    drop_x_na_in_test=False
):
    """
    两层 Stacking（固定 LogisticRegression 元模型）：
      1) OOF 生成 train_meta；
      2) 在 train_meta 上做 RepeatedStratifiedKFold + GridSearchCV 搜参（仅调 C）；
      3) 用全量 base 模型得到 test_meta，评估一次。
    兼容黑箱基模型：使用 deepcopy 克隆；如提供工厂函数，则优先用工厂函数创建干净实例。
    """
    if weight_list is None or len(weight_list) != len(base_models):
        raise ValueError("weight_list 必须提供，且长度与 base_models 一致。")

    # 清洗对齐
    X_train, y_train = _clean_xy(X_train, y_train, drop_x_na=drop_x_na_in_train, name="train")
    X_test,  y_test  = _clean_xy(X_test,  y_test,  drop_x_na=drop_x_na_in_test,  name="test")
    y_train = _encode_binary_y(y_train)
    y_test  = _encode_binary_y(y_test)
    if len(np.unique(y_train)) < 2:
        raise ValueError("y_train 仅包含单一类别，无法训练。")

    # 权重归一化
    n_models = len(base_models)
    weights = np.asarray(weight_list, dtype=float)
    if not np.isfinite(weights).all() or (weights < 0).any():
        raise ValueError("weight_list 需为非负且有限数。")
    s = weights.sum()
    weights = (weights / s) if s > 0 else np.ones_like(weights) / len(weights)

    if verbose:
        print(f"[Info] Using {n_models} base models; normalized weights: {np.round(weights, 4)}")

    # Step 1: OOF 生成 train_meta
    train_meta = np.zeros((len(y_train), n_models), dtype=float)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    for m_idx, base in enumerate(base_models):
        if verbose:
            print(f"\n[Base-{m_idx+1}/{n_models}] Building OOF predictions ...")
        for fold_idx, (tr_idx, va_idx) in enumerate(skf.split(_as_np(X_train), _as_np(y_train))):
            X_tr, y_tr = _slice_xy(X_train, y_train, tr_idx)
            X_va, _    = _slice_xy(X_train, y_train, va_idx)

            # 关键：每折拿“全新实例”
            seed = (random_state + 1009 * (m_idx + 1) + 31 * (fold_idx + 1)) & 0x7fffffff
            model_f = _fresh_model(base, seed=seed)

            # ⚠️ 假设 fit() 会覆盖旧状态；如你的模型是增量学习，请改为传工厂函数
            model_f.fit(X_tr, y_tr)
            proba_va = _get_proba(model_f, X_va)
            train_meta[va_idx, m_idx] = proba_va * weights[m_idx]

            if verbose:
                print(f"  fold {fold_idx+1}/{n_folds} done.")

    # 兜底处理
    if np.isnan(train_meta).any() or ~np.isfinite(train_meta).all():
        if verbose:
            print("[Warn] train_meta 出现 NaN/inf，已用 0.5/0 替换（NaN->0.5，inf->0）。")
        train_meta = np.nan_to_num(train_meta, nan=0.5, posinf=0.0, neginf=0.0)

    # Step 2: 全量训练基模型 → test_meta
    test_meta = np.zeros((len(y_test), n_models), dtype=float)
    for m_idx, base in enumerate(base_models):
        seed = (random_state + 2027 * (m_idx + 1)) & 0x7fffffff
        model_full = _fresh_model(base, seed=seed)
        model_full.fit(X_train, y_train)
        proba_te = _get_proba(model_full, X_test)
        test_meta[:, m_idx] = proba_te * weights[m_idx]

    if np.isnan(test_meta).any() or ~np.isfinite(test_meta).all():
        if verbose:
            print("[Warn] test_meta 出现 NaN/inf，已用 0.5/0 替换（NaN->0.5，inf->0）。")
        test_meta = np.nan_to_num(test_meta, nan=0.5, posinf=0.0, neginf=0.0)

    # Step 3: 在 train_meta 上搜参的 LogReg 元模型
    steps = []
    if use_logit:
        steps.append(("logit", FunctionTransformer(_safe_logit, validate=False)))
    if scale_meta:
        steps.append(("scaler", StandardScaler()))
    steps.append(("clf", LogisticRegression(
        solver="lbfgs", max_iter=10000, class_weight=class_weight, random_state=random_state
    )))
    pipe = Pipeline(steps)
    param_grid = {"clf__C": list(meta_C_grid)}

    meta_cv = RepeatedStratifiedKFold(n_splits=n_folds, n_repeats=meta_cv_repeats, random_state=random_state)
    gscv = GridSearchCV(pipe, param_grid=param_grid, scoring="roc_auc", cv=meta_cv, n_jobs=-1, refit=True, verbose=0)
    gscv.fit(train_meta, y_train)

    best_meta = gscv.best_estimator_
    if verbose:
        print(f"\n[Meta] Best C: {gscv.best_params_['clf__C']}, CV AUC: {gscv.best_score_:.4f}")

    # Step 4: 测试评估
    final_proba = best_meta.predict_proba(test_meta)[:, 1]
    final_pred = (final_proba >= 0.5).astype(int)
    metrics = {
        "accuracy": accuracy_score(y_test, final_pred),
        "f1": f1_score(y_test, final_pred),
        "precision": precision_score(y_test, final_pred),
        "recall": recall_score(y_test, final_pred),
        "auc": roc_auc_score(y_test, final_proba),
    }

    if verbose:
        print("\n✅ Evaluation on Test:")
        for k, v in metrics.items():
            print(f"  {k.capitalize()}: {v:.4f}")

    return {
        "meta_model": best_meta,
        "stacking_metrics": metrics,
        "train_meta": train_meta,
        "test_meta": test_meta,
        "weights_used": weights,
        "meta_cv_best_params": gscv.best_params_,
        "meta_cv_best_score": gscv.best_score_,
    }



def stacking_ensemble_balance(
    base_models,                 # 支持两种输入：1) 模型实例列表；2) 工厂函数列表（每次调用返回新实例）
    X_train, y_train,
    X_test, y_test,
    weight_list,
    n_folds=5,
    meta_cv_repeats=3,
    meta_C_grid=(0.01, 0.03, 0.1, 0.3, 1, 3, 10),
    use_logit=True,
    scale_meta=True,
    class_weight="balanced",
    random_state=42,
    verbose=False,
    drop_x_na_in_train=False,
    drop_x_na_in_test=False
):
    """
    两层 Stacking（固定 LogisticRegression 元模型）：
      1) OOF 生成 train_meta；
      2) 在 train_meta 上做 RepeatedStratifiedKFold + GridSearchCV 搜参（仅调 C）；
      3) 用全量 base 模型得到 test_meta，评估一次。
    兼容黑箱基模型：使用 deepcopy 克隆；如提供工厂函数，则优先用工厂函数创建干净实例。
    """
    if weight_list is None or len(weight_list) != len(base_models):
        raise ValueError("weight_list 必须提供，且长度与 base_models 一致。")

    # 清洗对齐
    X_train, y_train = _clean_xy(X_train, y_train, drop_x_na=drop_x_na_in_train, name="train")
    X_test,  y_test  = _clean_xy(X_test,  y_test,  drop_x_na=drop_x_na_in_test,  name="test")
    y_train = _encode_binary_y(y_train)
    y_test  = _encode_binary_y(y_test)
    if len(np.unique(y_train)) < 2:
        raise ValueError("y_train 仅包含单一类别，无法训练。")

    # 权重归一化
    n_models = len(base_models)
    weights = np.asarray(weight_list, dtype=float)
    if not np.isfinite(weights).all() or (weights < 0).any():
        raise ValueError("weight_list 需为非负且有限数。")
    s = weights.sum()
    weights = (weights / s) if s > 0 else np.ones_like(weights) / len(weights)

    if verbose:
        print(f"[Info] Using {n_models} base models; normalized weights: {np.round(weights, 4)}")

    # Step 1: OOF 生成 train_meta
    train_meta = np.zeros((len(y_train), n_models), dtype=float)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    for m_idx, base in enumerate(base_models):
        if verbose:
            print(f"\n[Base-{m_idx+1}/{n_models}] Building OOF predictions ...")
        for fold_idx, (tr_idx, va_idx) in enumerate(skf.split(_as_np(X_train), _as_np(y_train))):
            X_tr, y_tr = _slice_xy(X_train, y_train, tr_idx)
            X_va, _    = _slice_xy(X_train, y_train, va_idx)

            # 关键：每折拿“全新实例”
            seed = (random_state + 1009 * (m_idx + 1) + 31 * (fold_idx + 1)) & 0x7fffffff
            model_f = _fresh_model(base, seed=seed)

            # ⚠️ 假设 fit() 会覆盖旧状态；如你的模型是增量学习，请改为传工厂函数
            model_f.fit(X_tr, y_tr)
            proba_va = _get_proba(model_f, X_va)
            train_meta[va_idx, m_idx] = proba_va * weights[m_idx]

            if verbose:
                print(f"  fold {fold_idx+1}/{n_folds} done.")

    # 兜底处理
    if np.isnan(train_meta).any() or ~np.isfinite(train_meta).all():
        if verbose:
            print("[Warn] train_meta 出现 NaN/inf，已用 0.5/0 替换（NaN->0.5，inf->0）。")
        train_meta = np.nan_to_num(train_meta, nan=0.5, posinf=0.0, neginf=0.0)

    # Step 2: 全量训练基模型 → test_meta
    test_meta = np.zeros((len(y_test), n_models), dtype=float)
    for m_idx, base in enumerate(base_models):
        seed = (random_state + 2027 * (m_idx + 1)) & 0x7fffffff
        model_full = _fresh_model(base, seed=seed)
        model_full.fit(X_train, y_train)
        proba_te = _get_proba(model_full, X_test)
        test_meta[:, m_idx] = proba_te * weights[m_idx]

    if np.isnan(test_meta).any() or ~np.isfinite(test_meta).all():
        if verbose:
            print("[Warn] test_meta 出现 NaN/inf，已用 0.5/0 替换（NaN->0.5，inf->0）。")
        test_meta = np.nan_to_num(test_meta, nan=0.5, posinf=0.0, neginf=0.0)

    # Step 3: 在 train_meta 上搜参的 LogReg 元模型
    steps = []
    if use_logit:
        steps.append(("logit", FunctionTransformer(_safe_logit, validate=False)))
    if scale_meta:
        steps.append(("scaler", StandardScaler()))
    steps.append(("clf", LogisticRegression(
        solver="lbfgs", max_iter=10000, class_weight=class_weight, random_state=random_state
    )))
    pipe = Pipeline(steps)
    param_grid = {"clf__C": list(meta_C_grid)}

    meta_cv = RepeatedStratifiedKFold(n_splits=n_folds, n_repeats=meta_cv_repeats, random_state=random_state)
    gscv = GridSearchCV(pipe, param_grid=param_grid, scoring="roc_auc", cv=meta_cv, n_jobs=-1, refit=True, verbose=0)
    gscv.fit(train_meta, y_train)

    best_meta = gscv.best_estimator_
    if verbose:
        print(f"\n[Meta] Best C: {gscv.best_params_['clf__C']}, CV AUC: {gscv.best_score_:.4f}")

    # Step 4: 测试评估
    final_proba = best_meta.predict_proba(test_meta)[:, 1]
    final_pred = (final_proba >= 0.5).astype(int)
    metrics = {
        "accuracy": accuracy_score(y_test, final_pred),
        "f1": f1_score(y_test, final_pred),
        "precision": precision_score(y_test, final_pred),
        "recall": recall_score(y_test, final_pred),
        "auc": roc_auc_score(y_test, final_proba),
    }

    if verbose:
        print("\n✅ Evaluation on Test:")
        for k, v in metrics.items():
            print(f"  {k.capitalize()}: {v:.4f}")

    return {
        "meta_model": best_meta,
        "stacking_metrics": metrics,
        "train_meta": train_meta,
        "test_meta": test_meta,
        "weights_used": weights,
        "meta_cv_best_params": gscv.best_params_,
        "meta_cv_best_score": gscv.best_score_,
    }



