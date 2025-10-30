import numpy as np
from typing import List


def _predict_labels_safe(model, X):
    """获取模型的离散预测标签。优先使用 predict；若无则用概率/决策分数 argmax 回退。"""
    if hasattr(model, 'predict'):
        y = model.predict(X)
        return np.asarray(y).ravel()
    # 回退方案
    if hasattr(model, 'predict_proba'):
        proba = np.asarray(model.predict_proba(X))
        if proba.ndim == 1:
            proba = proba.reshape(-1, 1)
        return np.argmax(proba, axis=1)
    if hasattr(model, 'decision_function'):
        df = np.asarray(model.decision_function(X))
        if df.ndim == 1:  # 二分类 margin
            p1 = 1.0 / (1.0 + np.exp(-df))
            proba = np.stack([1 - p1, p1], axis=1)
        else:  # 多分类 softmax
            z = df - np.max(df, axis=1, keepdims=True)
            ez = np.exp(z)
            proba = ez / np.clip(ez.sum(axis=1, keepdims=True), 1e-12, None)
        return np.argmax(proba, axis=1)
    # 再退化：无法获得概率/分数时，抛错
    raise AttributeError("Model must implement predict / predict_proba / decision_function.")


# 多分类预测差异矩阵（Disagreement Rate）
def get_model_pre_differences_multiclass(best_fitted_model_instance_list, val_x, val_y=None):
    """
    计算模型之间在多分类任务上的预测差异程度（Disagreement Rate 矩阵）。

    参数：
        best_fitted_model_instance_list: 已拟合模型实例列表（需至少具备 .predict / .predict_proba / .decision_function 之一）
        val_x: 验证集特征
        val_y: 验证集标签（本函数不使用，保留以与二分类版本签名兼容）

    返回：
        N x N 对称矩阵，matrix[i][j] 表示模型 i 与模型 j 在验证集上预测不一致的比例（越大说明互补性越强）。
    """
    n = len(best_fitted_model_instance_list)
    if n == 0:
        return np.zeros((0, 0))

    # 统一得到每个模型的离散预测标签（top-1）
    preds = []
    for m in best_fitted_model_instance_list:
        yi = _predict_labels_safe(m, val_x)
        preds.append(yi)

    matrix = np.zeros((n, n), dtype=float)
    for i in range(n):
        matrix[i, i] = 0.0
        for j in range(i + 1, n):
            # 不一致比例（Hamming 距离的归一化形式）
            disagree_rate = float(np.mean(preds[i] != preds[j]))
            matrix[i, j] = disagree_rate
            matrix[j, i] = disagree_rate

    return matrix


def _error_matrix_multiclass(best_fitted_model_instance_list: List, val_x, val_y) -> np.ndarray:
    """
    生成错误指示矩阵 E（多分类安全版），形状 (n_models, n_samples)。
    E[j, i] = 1 表示第 j 个模型在第 i 个样本上预测错误；否则为 0。
    与二分类思想相同，只看“是否正确”。为了健壮性，缺少 predict 时会回退到概率/分数。
    """
    y_true = np.asarray(val_y).ravel()
    n_models = len(best_fitted_model_instance_list)
    n_samples = y_true.shape[0]
    E = np.zeros((n_models, n_samples), dtype=np.uint8)

    for j, model in enumerate(best_fitted_model_instance_list):
        y_pred = _predict_labels_safe(model, val_x)
        y_pred = np.asarray(y_pred).ravel()
        if y_pred.shape[0] != n_samples:
            raise ValueError(f"Model {j} predict length {y_pred.shape[0]} != {n_samples}")
        E[j] = (y_pred != y_true).astype(np.uint8)
    return E


# 多分类版：Jaccard-Fault 矩阵
def get_model_jaccard_fault_multiclass(best_fitted_model_instance_list: List, val_x, val_y) -> np.ndarray:
    """
    定义每个模型的“错误集合” E_i = {k | 模型 i 在样本 k 上出错}。
    返回 N×N 对称矩阵 J，其中 J[i, j] = |E_i ∩ E_j| / |E_i ∪ E_j|。
      - 若两者从不出错且 |E_i ∪ E_j|=0，则定义 J[i, j] = 0（避免 NaN）。
      - 取值范围 [0, 1]；数值越小，说明两模型的错误重叠越少（互补性越强）。
      - 对角线 J[i, i] = 1（如果该模型有至少一个错误）；若完全无错，定义为 0。
    注：本函数面向多分类，思想与二分类版保持一致。
    """
    E = _error_matrix_multiclass(best_fitted_model_instance_list, val_x, val_y).astype(np.float64)
    # 交集计数
    inter_counts = E @ E.T  # (N, N)
    # 每个模型的错误数
    err_counts = E.sum(axis=1)  # (N,)
    # 并集计数 |A ∪ B| = |A| + |B| − |A ∩ B|
    union_counts = err_counts[:, None] + err_counts[None, :] - inter_counts

    with np.errstate(divide='ignore', invalid='ignore'):
        J = inter_counts / union_counts
        # 处理 union 为 0 的情况（两者都无错）：定义 J=0
        J = np.where(union_counts == 0.0, 0.0, J)

    # 对角线处理：若该模型有错，则 J[i,i]=1；否则 0
    has_err = (err_counts > 0)
    np.fill_diagonal(J, 0.0)
    for i, flag in enumerate(has_err):
        J[i, i] = 1.0 if flag else 0.0

    # 数值稳定：限制到 [0,1]
    np.clip(J, 0.0, 1.0, out=J)
    return J


# 多分类版本 Rescue-Confidence 矩阵
def get_rescue_confidence_matrix_multiclass(
    best_fitted_model_instance_list: List,
    val_x,
    val_y,
    beta: float = 5.0,
    eps: float = 1e-12,
    label_smoothing: float = 1e-3,
) -> np.ndarray:
    """
    仅用于多分类任务的 Rescue–Confidence 矩阵 R（非对称）。
    R[i, j] 表示“模型 i 在模型 j 犯错时把样本救回来的能力”。

    要求（多分类）：
    - 各模型应提供 predict 与（优先）predict_proba。
    - 若未提供 predict_proba 或无法对齐类别，将回退到基于预测类别的平滑伪概率。
    """
    models = best_fitted_model_instance_list
    N = len(models)
    val_y = np.asarray(val_y)

    def _get_model_classes(m):
        if hasattr(m, "model") and hasattr(m.model, "classes_"):
            return np.asarray(m.model.classes_)
        elif hasattr(m, "classes_"):
            return np.asarray(m.classes_)
        return None

    # ---------- 统一全局类别空间 ----------
    classes = set(np.unique(val_y))
    for m in models:
        cls = _get_model_classes(m)
        if cls is not None:
            classes.update(list(cls))
    global_classes = np.array(sorted(list(classes)))
    C = len(global_classes)
    cls_index = {c: i for i, c in enumerate(global_classes)}

    # 索引化真标签
    y_true_idx = np.vectorize(lambda c: cls_index[c])(val_y)
    M = len(val_y)

    preds_idx = []
    probas = []

    for m in models:
        # 预测类别并索引化
        y_pred = m.predict(val_x)
        y_pred_idx = np.vectorize(lambda c: cls_index[c])(y_pred)
        preds_idx.append(y_pred_idx)

        # 目标：构造对齐到全局类别的 P: (M, C)
        P = np.zeros((M, C), dtype=np.float64)

        if hasattr(m, "predict_proba"):
            p_local = np.asarray(m.predict_proba(val_x))
            local_classes = _get_model_classes(m)

            if p_local.ndim == 2 and p_local.shape[0] == M:
                # 具备二维概率
                if (local_classes is not None) and (len(local_classes) == p_local.shape[1]):
                    # 按模型自身类顺序映射到全局
                    for j_cls, cls in enumerate(local_classes):
                        P[:, cls_index[cls]] = p_local[:, j_cls]
                else:
                    # 无法可靠对齐 → 仅保障预测类概率正确，其余均匀摊
                    maxp = np.clip(p_local.max(axis=1), 0.0, 1.0)
                    P[np.arange(M), y_pred_idx] = maxp
                    remain = 1.0 - maxp
                    if C > 1:
                        P += (remain / (C - 1))[:, None]
                        P[np.arange(M), y_pred_idx] -= remain / (C - 1)
            else:
                # 概率形状异常（如一维）→ 退化兜底
                maxp = np.clip(p_local.ravel(), 0.0, 1.0)
                if maxp.shape[0] != M:
                    maxp = np.full(M, 1.0, dtype=np.float64)
                P[np.arange(M), y_pred_idx] = maxp
                remain = 1.0 - maxp
                if C > 1:
                    P += (remain / (C - 1))[:, None]
                    P[np.arange(M), y_pred_idx] -= remain / (C - 1)
        else:
            # 无 predict_proba：用 label smoothing 伪概率
            eps_ls = float(label_smoothing)
            if C > 1:
                P[:] = eps_ls
                P[np.arange(M), y_pred_idx] = 1.0 - (C - 1) * eps_ls
            else:
                P[:, 0] = 1.0

        # 数值剪裁 + 归一
        P = np.clip(P, 1e-12, 1.0)
        P /= P.sum(axis=1, keepdims=True)
        probas.append(P)

    preds_idx = np.asarray(preds_idx)              # (N, M)
    probas = np.asarray(probas)                    # (N, M, C)

    # 正确/错误掩码
    E = (preds_idx != y_true_idx[None, :]).astype(np.float64)  # j 错
    Cmask = 1.0 - E                                            # i 对

    # p_i(y|x)
    rows = np.arange(M)
    s_true = probas[:, rows, y_true_idx]           # (N, M)

    # margin_j(x) = p_j(ŷ_j|x) - p_j(y|x)
    p_pred = probas[np.arange(N)[:, None], rows[None, :], preds_idx]  # (N, M)
    margin = p_pred - s_true

    # 错样权重（仅在 j 错时起作用）
    z = np.clip(beta * margin, -50, 50)
    W = (1.0 / (1.0 + np.exp(-z))) * E             # (N, M)
    Z = W.sum(axis=1) + eps                        # (N,)

    # 计算 R（列按 j 归一）
    CS = Cmask * s_true                            # (N, M)
    R = np.zeros((N, N), dtype=np.float64)
    for j in range(N):
        num = CS @ W[j]                            # (N,)
        R[:, j] = num / Z[j]

    np.fill_diagonal(R, 0.0)
    np.clip(R, 0.0, 1.0, out=R)
    return R

# 多分类模型生成提示词
def get_model_prompt_multiclass(target_column_name=None, samples=None):
    prompt = f"""
        Here is what I provide:

        All column names and a few sample rows of data (in tabular form):
        {samples}

        The name of the target column for multi-class classification:
        "{target_column_name}"

        Based on this information, you must:
        1. Generate a new Python classifier class named `myclassifier` each round.
        2. The model must differ from previous versions in **model type or structure**.
        3. The goal is to **maximize AUC (Area Under the ROC Curve)** on the given test data.

        The class must support the following methods:
            model = myclassifier()                        # Initialize the classifier
            model.fit(train_aug_x, train_aug_y)           # Train the model (both are pandas DataFrames)
            pred = model.predict(test_aug_x)              # Predict class labels (1D array or list)
            proba = model.predict_proba(test_aug_x)       # Predict probabilities for all classes (2D: n_samples x n_classes)

        Important instructions (must follow **strictly**):
        - Only output the complete Python class named `myclassifier`. No explanation, no markdown, no extra output.
        - The class must be ready to use in Python (all necessary imports included, no undefined variables).
        - `train_aug_x`, `train_aug_y`, and `test_aug_x` are all pandas DataFrames.
        - `predict` must return a 1D array or list (same length as `test_aug_x`).
        - `predict_proba` must return the **full class probability matrix** with shape (n_samples, n_classes). Do **not** slice a single column.
        - Each new version must differ from the previous ones (by model type or structure).
        - Use diverse models across rounds (e.g., LogisticRegression, RandomForest, GradientBoosting, SVC, stacking/ensemble, etc.).
        - You may use libraries such as scikit-learn (>=1.4), XGBoost, CatBoost, etc.
        - **You must NOT use `LightGBM` under any circumstances. It is forbidden due to compatibility issues.**
        - Always prefer models that support probabilistic outputs suitable for AUC optimization.
        Output must be **only** the full Python class named `myclassifier`. No extra explanation or code.
        """
    return prompt

def get_model_prompt_multiclass_new(ds_description):
    prompt = f"""
        Here is what I provide:

        The dataset description for multi-class classification:
        {ds_description}

        Based on this information, you must:
        1. Generate a new Python classifier class named `myclassifier` each round.
        2. The model must differ from previous versions in **model type or structure**.
        3. The goal is to **maximize AUC (Area Under the ROC Curve)** on the given test data.

        The class must support the following methods:
            model = myclassifier()                        # Initialize the classifier
            model.fit(train_aug_x, train_aug_y)           # Train the model (both are pandas DataFrames)
            pred = model.predict(test_aug_x)              # Predict class labels (1D array or list)
            proba = model.predict_proba(test_aug_x)       # Predict probabilities for all classes (2D: n_samples x n_classes)

        Important instructions (must follow **strictly**):
        - Only output the complete Python class named `myclassifier`. No explanation, no markdown, no extra output.
        - The class must be ready to use in Python (all necessary imports included, no undefined variables).
        - `train_aug_x`, `train_aug_y`, and `test_aug_x` are all pandas DataFrames.
        - `predict` must return a 1D array or list (same length as `test_aug_x`).
        - `predict_proba` must return the **full class probability matrix** with shape (n_samples, n_classes). Do **not** slice a single column.
        - Each new version must differ from the previous ones (by model type or structure).
        - Use diverse models across rounds (e.g., LogisticRegression, RandomForest, GradientBoosting, SVC, stacking/ensemble, etc.).
        - You may use libraries such as scikit-learn (>=1.4), XGBoost, CatBoost, etc.
        - **You must NOT use `LightGBM` under any circumstances. It is forbidden due to compatibility issues.**
        - Always prefer models that support probabilistic outputs suitable for AUC optimization.
        Output must be **only** the full Python class named `myclassifier`. No extra explanation or code.
        """
    return prompt

# 原来实验的多分类模型生成提示词
def get_model_prompt_multiclass_o(target_column_name=None, samples=None):
    prompt = f"""
        Here is what I provide:

        All column names and a few sample rows of data (in tabular form):
        {samples}

        The name of the target column for multi-class classification:
        "{target_column_name}"

        Based on this information, you must:
        1. Generate a new Python classifier class named `myclassifier` each round.
        2. The model must differ from previous versions in **model type or structure**.
        3. The goal is to maximize classification performance on the given test data.

        The class must support the following methods:
            model = myclassifier()                        # Initialize the classifier
            model.fit(train_aug_x, train_aug_y)           # Train the model (both are pandas DataFrames)
            pred = model.predict(test_aug_x)              # Predict class labels (1D array or list)
            proba = model.predict_proba(test_aug_x)       # Predict probabilities for each class

        Important instructions (must follow **strictly**):
        - Only output the complete Python class named `myclassifier`. No explanation, no markdown, no extra output.
        - The class must be ready to use in Python (all necessary imports included, no undefined variables).
        - `train_aug_x`, `train_aug_y`, and `test_aug_x` are all pandas DataFrames.
        - `predict` must return a 1D array or list of predicted labels (same length as `test_aug_x`).
        - `predict_proba` must return a 2D NumPy array of shape (n_samples, n_classes), representing probabilities for **each class**:
              return self.model.predict_proba(test_aug_x)
        - Each new version must differ from the previous ones (by model type or structure).
        - Use diverse models across rounds (e.g., LogisticRegression, RandomForest, GradientBoosting, SVC, MLP, stacking/ensemble, etc.).
        - You may use libraries such as scikit-learn (>=1.4), XGBoost, LightGBM, CatBoost, etc.
        - The model must be designed for multi-class classification tasks (i.e., handle `n_classes > 2`).
        - **You must NOT use `BaggingClassifier` under any circumstances. It is forbidden due to compatibility issues.**

        Output must be **only** the full Python class named `myclassifier`. No extra explanation or code.
        """
    return prompt