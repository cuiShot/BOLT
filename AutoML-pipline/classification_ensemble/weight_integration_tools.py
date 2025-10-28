# =====================  Classification-only integration utils  =====================
# 仅包含“分类任务”的三策略权重整合（D / J / R）：
# - 方法A：软最大(−τ·loss) → 策略层权重 α → 几何池化 → 均匀收缩
# - 方法B：Hedge 在线专家 → α → 几何池化 → 均匀收缩
# - 方法C：单阶段正则化优化（logloss + 正则项）→ PGD
#
# 依赖：numpy
# -----------------------------------------------------------------------------

from typing import List, Dict, Tuple, Optional
import numpy as np

# ----------------------- 基础工具 -----------------------

def _project_to_simplex(v: np.ndarray) -> np.ndarray:
    """
    投影到概率单纯形 {w >= 0, sum w = 1}（Duchi et al., 2008）
    """
    v = np.asarray(v, dtype=np.float64)
    n = v.size
    if n == 0:
        return v
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho_idx = np.nonzero(u * (np.arange(1, n + 1)) > (cssv - 1))[0]
    if len(rho_idx) == 0:
        return np.ones_like(v) / n
    rho = rho_idx[-1]
    theta = (cssv[rho] - 1.0) / (rho + 1)
    w = np.maximum(v - theta, 0.0)
    s = w.sum()
    if s <= 0:
        return np.ones_like(w) / n
    return w / s

def _uniform_shrinkage(w: np.ndarray, lam: float) -> np.ndarray:
    """
    均匀收缩：w <- (1-λ)·w + λ·1/n  （可减小过拟合）
    """
    n = w.size
    u = np.ones(n, dtype=np.float64) / n
    w2 = (1.0 - lam) * w + lam * u
    return w2 / w2.sum()

def _geometric_pool(weights_list: List[np.ndarray], alpha: np.ndarray, clip: float = 1e-12) -> np.ndarray:
    """
    几何池化（对数意见汇合）：w ∝ Π_k w_k^{α_k}，最后归一化到单纯形
    """
    g = np.ones_like(weights_list[0], dtype=np.float64)
    for wk, ak in zip(weights_list, alpha):
        g *= np.power(np.clip(wk, clip, 1.0), ak)
    g /= g.sum()
    return g

# ----------------------- 预测与损失（分类） -----------------------

def _collect_outputs_classification(
    models: List, X, y_true, label_smoothing: float = 1e-3
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    汇集所有模型的对齐概率：
      返回：
        probas: (N, M, C)   各模型在全局类空间上的概率
        y_true_idx: (M,)    真类索引（在全局类数组中的位置）
        classes: (C,)       全局类标签（升序）
    兼容 predict_proba 形状：(M, C) / (M, 1) / (M,)
    若模型无 predict_proba，则对预测类使用 label smoothing 生成伪概率。
    """
    y_true = np.asarray(y_true)
    # 全局类集合
    classes = set(np.unique(y_true))
    for m in models:
        if hasattr(m, "classes_"):
            try:
                classes.update(list(m.classes_))
            except Exception:
                pass
    classes = np.array(sorted(list(classes)))
    C = len(classes)
    cls_index = {c: i for i, c in enumerate(classes)}

    y_true_idx = np.vectorize(lambda c: cls_index[c])(y_true)
    M = len(y_true)

    probas = []
    for m in models:
        y_pred = m.predict(X)
        y_pred_idx = np.vectorize(lambda c: cls_index[c])(y_pred)
        P = np.zeros((M, C), dtype=np.float64)

        if hasattr(m, "predict_proba"):
            p_local = np.asarray(m.predict_proba(X))
            if p_local.ndim == 1:
                # 一维 → 视作二分类正类概率
                if hasattr(m, "classes_") and len(getattr(m, "classes_")) == 2:
                    neg, pos = m.classes_[0], m.classes_[1]
                    pos_idx, neg_idx = cls_index[pos], cls_index[neg]
                    p_pos = np.clip(p_local, 0.0, 1.0)
                    P[:, pos_idx] = p_pos
                    P[:, neg_idx] = 1.0 - p_pos
                else:
                    # 兜底：将该概率赋给预测类，其余均匀分摊
                    maxp = np.clip(p_local, 0.0, 1.0)
                    P[np.arange(M), y_pred_idx] = maxp
                    remain = 1.0 - maxp
                    if C > 1:
                        P += (remain / (C - 1))[:, None]
                        P[np.arange(M), y_pred_idx] -= remain / (C - 1)
            else:
                # 二维 (M, C_m)
                if hasattr(m, "classes_"):
                    for j, cls in enumerate(m.classes_):
                        P[:, cls_index[cls]] = p_local[:, j]
                else:
                    # 无 classes_ → 把最大列对齐到预测类
                    maxp = p_local.max(axis=1)
                    P[np.arange(M), y_pred_idx] = maxp
                    remain = 1.0 - maxp
                    if C > 1:
                        P += (remain / (C - 1))[:, None]
                        P[np.arange(M), y_pred_idx] -= remain / (C - 1)
        else:
            # 无概率：用 label smoothing 伪概率
            eps_ls = label_smoothing if C > 1 else 0.0
            P[:] = eps_ls
            if C > 1:
                P[np.arange(M), y_pred_idx] = 1.0 - (C - 1) * eps_ls
            else:
                P[:, 0] = 1.0

        P = np.clip(P, 1e-12, 1.0)
        P /= P.sum(axis=1, keepdims=True)
        probas.append(P)

    probas = np.asarray(probas)  # (N, M, C)
    return probas, y_true_idx, classes

def _loss_classification_logloss(weights: np.ndarray, s_true: np.ndarray, clip=1e-15) -> float:
    """
    多模型混合的 logloss：
      s_true: (N, M)  第 i 个模型在真类上的概率 p_i(y|x)
      L = - 1/M Σ_x log( Σ_i w_i * s_i(x) )
    """
    S = np.clip((weights[:, None] * s_true).sum(axis=0), clip, 1.0)
    return -float(np.mean(np.log(S)))

# ----------------------- 策略层整合：方法A -----------------------

def integrate_softmax_geometric_cls(
    w_D: np.ndarray, w_J: np.ndarray, w_R: np.ndarray,
    models: List, val_x, val_y,
    tau: float = 10.0, shrink: float = 0.1, label_smoothing: float = 1e-3
) -> Tuple[np.ndarray, Dict]:
    """
    方法A（分类）：软最大(−τ·loss) → α → 几何池化 → 均匀收缩
    返回 (w_final, info)
    """
    N = len(models)
    w_D = _project_to_simplex(w_D); w_J = _project_to_simplex(w_J); w_R = _project_to_simplex(w_R)
    assert w_D.size == w_J.size == w_R.size == N

    probas, y_true_idx, _ = _collect_outputs_classification(models, val_x, val_y, label_smoothing)
    M = len(val_y)
    rows = np.arange(M)
    s_true = probas[np.arange(N)[:, None], rows[None, :], y_true_idx]  # (N, M)

    L_D = _loss_classification_logloss(w_D, s_true)
    L_J = _loss_classification_logloss(w_J, s_true)
    L_R = _loss_classification_logloss(w_R, s_true)

    losses = np.array([L_D, L_J, L_R], dtype=np.float64)
    # α ∝ exp(-τ·loss)（数值稳定：减去最小值）
    exps = np.exp(-tau * (losses - losses.min()))
    alpha = exps / exps.sum()

    w_geo = _geometric_pool([w_D, w_J, w_R], alpha)
    w_final = _uniform_shrinkage(w_geo, shrink)

    info = {"losses": {"D": L_D, "J": L_J, "R": L_R}, "alpha": alpha, "w_geo": w_geo}
    return w_final, info

# ----------------------- 策略层整合：方法B -----------------------

def integrate_hedge_geometric_cls(
    w_D: np.ndarray, w_J: np.ndarray, w_R: np.ndarray,
    models: List, val_x, val_y,
    K: int = 5, eta: Optional[float] = None, shrink: float = 0.1,
    seed: int = 42, label_smoothing: float = 1e-3
) -> Tuple[np.ndarray, Dict]:
    """
    方法B（分类）：Hedge 在线专家更新策略层 α，再几何池化 + 均匀收缩
    """
    N = len(models)
    w_D = _project_to_simplex(w_D); w_J = _project_to_simplex(w_J); w_R = _project_to_simplex(w_R)
    assert w_D.size == w_J.size == w_R.size == N

    probas, y_true_idx, _ = _collect_outputs_classification(models, val_x, val_y, label_smoothing)
    M = len(val_y)
    rows = np.arange(M)
    s_true = probas[np.arange(N)[:, None], rows[None, :], y_true_idx]  # (N, M)

    # 构造 K 折在线序列
    indices = np.arange(M)
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)
    folds = np.array_split(indices, K)

    alpha = np.ones(3, dtype=np.float64) / 3.0
    if eta is None:
        # 常用步长上界：sqrt(2*log(#experts)/K)
        eta = np.sqrt(2.0 * np.log(3.0) / max(K, 1))

    for f in folds:
        L_D = _loss_classification_logloss(w_D, s_true[:, f])
        L_J = _loss_classification_logloss(w_J, s_true[:, f])
        L_R = _loss_classification_logloss(w_R, s_true[:, f])
        losses = np.array([L_D, L_J, L_R])
        # Hedge 更新（平移稳定）
        alpha *= np.exp(-eta * (losses - losses.min()))
        alpha /= alpha.sum()

    w_geo = _geometric_pool([w_D, w_J, w_R], alpha)
    w_final = _uniform_shrinkage(w_geo, shrink)

    info = {"alpha": alpha, "K": K, "eta": eta, "w_geo": w_geo}
    return w_final, info

# ----------------------- 策略层整合：方法B 两种权重指导，用于消融实验-----------------------
def integrate_hedge_geometric_cls_two(
    w_A: np.ndarray, w_B: np.ndarray,
    models: List, val_x, val_y,
    K: int = 5, eta: Optional[float] = None, shrink: float = 0.1,
    seed: int = 42, label_smoothing: float = 1e-3
) -> Tuple[np.ndarray, Dict]:
    """
    方法B（分类，双专家）：
      - 用 Hedge 在线更新策略层 α ∈ Δ^1（两维）
      - 再对两个基础权重向量做几何池化（对数意见汇合）
      - 最后做一次均匀收缩以提升稳健性

    参数
    ----
    w_A, w_B : np.ndarray
        两个候选/来源的基础权重（长度均为模型数 N），会被投影到概率单纯形。
    models : List
        已训练好的分类模型列表（需有 predict；如有 predict_proba 则更佳）。
    val_x, val_y
        验证集特征与标签（用于在线序列与 Hedge 更新）。
    K : int
        在线序列的折数（将验证集打乱后分 K 折顺序“到来”）。
    eta : Optional[float]
        Hedge 步长；若为 None，则使用 sqrt(2*log(2)/K) 的常用上界。
    shrink : float
        均匀收缩强度 λ ∈ [0,1]。
    seed : int
        随机种子（打乱验证集顺序）。
    label_smoothing : float
        当模型没有概率输出时，用于构造伪概率的平滑系数。

    返回
    ----
    w_final : np.ndarray
        最终融合后的模型权重（在概率单纯形上）。
    info : Dict
        诊断信息：{'alpha', 'K', 'eta', 'w_geo'}
    """
    N = len(models)
    w_A = _project_to_simplex(w_A)
    w_B = _project_to_simplex(w_B)
    assert w_A.size == w_B.size == N

    # 收集对齐后的各模型概率输出，以及真类索引
    probas, y_true_idx, _ = _collect_outputs_classification(
        models, val_x, val_y, label_smoothing
    )
    M = len(val_y)
    rows = np.arange(M)
    # s_true[i, m] = 第 i 个模型在第 m 个样本真类上的概率
    s_true = probas[np.arange(N)[:, None], rows[None, :], y_true_idx]  # (N, M)

    # 构造 K 折在线“到达”序列
    indices = np.arange(M)
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)
    folds = np.array_split(indices, K)

    # 策略层：两位“专家”（来自 w_A / w_B），alpha 初始均匀
    alpha = np.ones(2, dtype=np.float64) / 2.0
    if eta is None:
        # 两专家的常用步长上界
        eta = np.sqrt(2.0 * np.log(2.0) / max(K, 1))

    for f in folds:
        L_A = _loss_classification_logloss(w_A, s_true[:, f])
        L_B = _loss_classification_logloss(w_B, s_true[:, f])
        losses = np.array([L_A, L_B])

        # Hedge 更新（减去最小值做平移，提升数值稳定性）
        alpha *= np.exp(-eta * (losses - losses.min()))
        alpha /= alpha.sum()

    # 几何池化 + 均匀收缩
    w_geo = _geometric_pool([w_A, w_B], alpha)
    w_final = _uniform_shrinkage(w_geo, shrink)

    info = {"alpha": alpha, "K": K, "eta": eta, "w_geo": w_geo}
    return w_final, info


# ----------------------- 单阶段正则化优化：方法C -----------------------

def optimize_weights_regularized_cls(
    models: List, val_x, val_y,
    D: Optional[np.ndarray] = None,
    J: Optional[np.ndarray] = None,
    R_matrix: Optional[np.ndarray] = None,
    init_w: Optional[np.ndarray] = None,
    steps: int = 400, lr: float = 0.5,
    lambda_J: float = 0.0, lambda_D: float = 0.0, mu_R: float = 0.0, eta_l2: float = 0.0,
    shrink: float = 0.1, label_smoothing: float = 1e-3
) -> Tuple[np.ndarray, Dict]:
    """
    方法C（分类，PGD）：最小化
        L_logloss(w) + λ_J·w^T J w − λ_D·w^T D w − μ·(out^T w) + η·||w||_2^2
      其中 out = R_matrix.sum(axis=1)
    返回 (w_final, info)
    """
    N = len(models)
    if init_w is None:
        init_w = np.ones(N, dtype=np.float64) / N
    w = _project_to_simplex(init_w)

    probas, y_true_idx, _ = _collect_outputs_classification(models, val_x, val_y, label_smoothing)
    M = len(val_y)
    rows = np.arange(M)
    s_true = probas[np.arange(N)[:, None], rows[None, :], y_true_idx]  # (N, M)

    for _ in range(steps):
        S = np.clip((w[:, None] * s_true).sum(axis=0), 1e-15, 1.0)  # (M,)
        # d/dw_i [-1/M Σ log S] = -(1/M) Σ (s_i / S)
        grad = -(s_true / S[None, :]).mean(axis=1)  # (N,)

        if J is not None:
            grad += 2.0 * lambda_J * (J @ w)
        if D is not None:
            grad += -2.0 * lambda_D * (D @ w)
        if R_matrix is not None and mu_R != 0.0:
            out = R_matrix.sum(axis=1)  # (N,)
            grad += -mu_R * out
        if eta_l2 != 0.0:
            grad += 2.0 * eta_l2 * w

        w = _project_to_simplex(w - lr * grad)

    w_final = _uniform_shrinkage(w, shrink)
    info = {"steps": steps, "lr": lr, "lambda_J": lambda_J, "lambda_D": lambda_D, "mu_R": mu_R, "eta_l2": eta_l2}
    return w_final, info

# ----------------------- 一键接口（分类） -----------------------

def integrate_three_strategies_cls(
    weight_vector_list: List[np.ndarray],
    models: List, val_x, val_y,
    method: str = "softmax-geo",
    **kwargs
) -> Tuple[np.ndarray, Dict]:
    """
    分类任务入口：
      method ∈ {"softmax-geo", "hedge-geo", "regularized-opt"}
    额外参数（按方法）：
      - softmax-geo: tau, shrink, label_smoothing
      - hedge-geo  : K, eta, shrink, seed, label_smoothing
      - regularized-opt: D, J, R_matrix, init_w, steps, lr,
                         lambda_J, lambda_D, mu_R, eta_l2, shrink, label_smoothing
    """
    method = method.lower()
    w_D = weight_vector_list[0]
    w_J = weight_vector_list[1]
    w_R = weight_vector_list[2]
    if method == "softmax-geo":
        return integrate_softmax_geometric_cls(
            w_D, w_J, w_R, models, val_x, val_y,
            tau=kwargs.get("tau", 10.0),
            shrink=kwargs.get("shrink", 0.1),
            label_smoothing=kwargs.get("label_smoothing", 1e-3),
        )
    elif method == "hedge-geo":
        return integrate_hedge_geometric_cls(
            w_D, w_J, w_R, models, val_x, val_y,
            K=kwargs.get("K", 5),
            eta=kwargs.get("eta", None),
            shrink=kwargs.get("shrink", 0.1),
            seed=kwargs.get("seed", 42),
            label_smoothing=kwargs.get("label_smoothing", 1e-3),
        )
    elif method == "regularized-opt":
        return optimize_weights_regularized_cls(
            models, val_x, val_y,
            D=kwargs.get("D", None),
            J=kwargs.get("J", None),
            R_matrix=kwargs.get("R_matrix", None),
            init_w=kwargs.get("init_w", None),
            steps=kwargs.get("steps", 400),
            lr=kwargs.get("lr", 0.5),
            lambda_J=kwargs.get("lambda_J", 0.0),
            lambda_D=kwargs.get("lambda_D", 0.0),
            mu_R=kwargs.get("mu_R", 0.0),
            eta_l2=kwargs.get("eta_l2", 0.0),
            shrink=kwargs.get("shrink", 0.1),
            label_smoothing=kwargs.get("label_smoothing", 1e-3),
        )
    else:
        raise ValueError("Unknown method. Choose from {'softmax-geo','hedge-geo','regularized-opt'}.")


# ----------------------- 使用示例 -----------------------
# w_final, info = integrate_three_strategies_cls(
#     w_D, w_J, w_R, best_fitted_model_instance_list, val_x, val_y,
#     method="softmax-geo", tau=10.0, shrink=0.1
# )
# 或：
# w_final, info = integrate_three_strategies_cls(
#     w_D, w_J, w_R, best_fitted_model_instance_list, val_x, val_y,
#     method="regularized-opt",
#     D=D, J=J, R_matrix=R, lambda_J=0.1, lambda_D=0.1, mu_R=0.2, eta_l2=0.01,
#     steps=500, lr=0.3, shrink=0.1
# )
