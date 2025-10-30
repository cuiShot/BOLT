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

# def _geometric_pool(weight_list, alpha, eps=1e-12):
#     import numpy as np
#     W = np.stack(weight_list, axis=0).astype(np.float64)  # (S, N)

#     # ε-平滑，避免出现 0^alpha
#     N = W.shape[1]
#     W = (1.0 - eps * N) * W + eps  # 每个坐标都有 >0 的最小质量
#     logW = np.log(W)

#     g_log = (alpha[:, None] * logW).sum(axis=0)
#     g = np.exp(g_log)
#     s = g.sum()

#     # 兜底：若仍异常，用加权算术平均代替
#     if not np.isfinite(s) or s <= 0:
#         g = np.average(W, axis=0, weights=alpha)
#         s = g.sum()

#     return g / s


# ----------------------- 预测与损失（回归） -----------------------

def _collect_outputs_regression(models: List, X) -> np.ndarray:
    """返回 preds: (N, M)；若出现 NaN/Inf 自动替换为 0。"""
    preds = []
    for m in models:
        y_hat = np.asarray(m.predict(X), dtype=np.float64).ravel()
        if not np.all(np.isfinite(y_hat)):
            y_hat = np.nan_to_num(y_hat, nan=0.0, posinf=0.0, neginf=0.0)
        preds.append(y_hat)
    return np.asarray(preds)


def _loss_regression_mse(weights: np.ndarray, preds: np.ndarray, y_true: np.ndarray) -> float:
    """preds: (N,M), y_true:(M,) → MSE of weighted blend."""
    y_hat = (weights[:, None] * preds).sum(axis=0)
    return float(np.mean((y_true - y_hat) ** 2))

# ----------------------- 策略层整合：方法A -----------------------

def integrate_softmax_geometric_reg(
    w_D: np.ndarray, w_J: np.ndarray, w_R: np.ndarray,
    models: List, val_x, val_y,
    tau: float = 10.0, shrink: float = 0.1
) -> Tuple[np.ndarray, Dict]:
    """
    方法A（回归）：软最大(−τ·MSE) → α → 几何池化 → 均匀收缩
    返回 (w_final, info)
    """
    N = len(models)
    w_D = _project_to_simplex(w_D); w_J = _project_to_simplex(w_J); w_R = _project_to_simplex(w_R)
    assert w_D.size == w_J.size == w_R.size == N

    preds = _collect_outputs_regression(models, val_x)  # (N,M)
    y_true = np.asarray(val_y, dtype=np.float64).ravel()

    L_D = _loss_regression_mse(w_D, preds, y_true)
    L_J = _loss_regression_mse(w_J, preds, y_true)
    L_R = _loss_regression_mse(w_R, preds, y_true)

    losses = np.array([L_D, L_J, L_R], dtype=np.float64)
    exps = np.exp(-tau * (losses - losses.min()))  # 稳定化
    alpha = exps / exps.sum()

    w_geo = _geometric_pool([w_D, w_J, w_R], alpha)
    w_final = _uniform_shrinkage(w_geo, shrink)

    info = {"losses": {"D": L_D, "J": L_J, "R": L_R}, "alpha": alpha, "w_geo": w_geo}
    return w_final, info

# ----------------------- 策略层整合：方法B -----------------------

def integrate_hedge_geometric_reg(
    w_D: np.ndarray, w_J: np.ndarray, w_R: np.ndarray,
    models: List, val_x, val_y,
    K: int = 5, eta: Optional[float] = None, shrink: float = 0.1,
    seed: int = 42
) -> Tuple[np.ndarray, Dict]:
    """
    方法B(回归):Hedge 在线专家更新策略层 α，再几何池化 + 均匀收缩
    """
    N = len(models)
    w_D = _project_to_simplex(w_D); w_J = _project_to_simplex(w_J); w_R = _project_to_simplex(w_R)
    assert w_D.size == w_J.size == w_R.size == N

    preds = _collect_outputs_regression(models, val_x)  # (N,M)
    y_true = np.asarray(val_y, dtype=np.float64).ravel()
    M = y_true.shape[0]

    

    indices = np.arange(M)
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)
    folds = np.array_split(indices, K)

    alpha = np.ones(3, dtype=np.float64) / 3.0
    if eta is None:
        eta = np.sqrt(2.0 * np.log(3.0) / max(K, 1))

    for f in folds:
        if f.size == 0:
            continue  # 安全起见，跳过
        L_D = _loss_regression_mse(w_D, preds[:, f], y_true[f])
        L_J = _loss_regression_mse(w_J, preds[:, f], y_true[f])
        L_R = _loss_regression_mse(w_R, preds[:, f], y_true[f])

        # 自检
        # print("preds NaN?", np.isnan(preds).any())
        # print("val_y NaN?", np.isnan(y_true).any())
        # print("w_D/J/R NaN?", np.isnan([w_D, w_J, w_R]).any())
        # print("losses:", L_D, L_J, L_R)

        losses = np.array([L_D, L_J, L_R])
        if not np.isfinite(losses).all():
            continue  # 若本折异常，直接跳过
        alpha *= np.exp(-eta * (losses - losses.min()))
        alpha = np.maximum(alpha, 1e-12)      # 防止下溢
        alpha /= alpha.sum()

    w_geo = _geometric_pool([w_D, w_J, w_R], alpha)
    w_final = _uniform_shrinkage(w_geo, shrink)

    info = {"alpha": alpha, "K": K, "eta": eta, "w_geo": w_geo}
    return w_final, info

def integrate_hedge_geometric_reg_two(
    w_A: np.ndarray, w_B: np.ndarray,
    models: List, val_x, val_y,
    K: int = 5, eta: Optional[float] = None, shrink: float = 0.1,
    seed: int = 42
) -> Tuple[np.ndarray, Dict]:
    """
    方法B(回归, 双策略版):
    使用 Hedge 在策略层对两种策略 (A,B) 的权重进行在线更新，得到 α ∈ Δ^2，
    然后用几何池化融合基模型权重，再做均匀收缩。

    参数
    ----
    w_A, w_B : 两种策略给出的基模型权重（长度 N，与 models 数量一致）
    models   : 基模型实例列表
    val_x, val_y : 验证集（用于在线更新 α）
    K        : 将验证集随机划分为 K 折做在线更新
    eta      : Hedge 学习率；None 时自适应设为 sqrt(2 ln 2 / K)
    shrink   : 均匀收缩系数 λ
    seed     : 随机种子

    返回
    ----
    w_final : 融合后用于集成的基模型权重（投影到概率单纯形）
    info    : 调试信息，包含 alpha/K/eta/w_geo
    """
    N = len(models)
    w_A = _project_to_simplex(w_A)
    w_B = _project_to_simplex(w_B)
    assert w_A.size == w_B.size == N, "权重长度需与 models 数量一致"

    preds = _collect_outputs_regression(models, val_x)  # (N, M)
    y_true = np.asarray(val_y, dtype=np.float64).ravel()
    M = y_true.shape[0]

    # 将验证集打散为 K 折
    indices = np.arange(M)
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)
    folds = np.array_split(indices, K)

    # 两策略的初始权重 α，若未指定学习率则按 K 自适应
    alpha = np.ones(2, dtype=np.float64) / 2.0
    if eta is None:
        eta = np.sqrt(2.0 * np.log(2.0) / max(K, 1))

    # 在线更新（Hedge）
    for f in folds:
        if f.size == 0:
            continue
        L_A = _loss_regression_mse(w_A, preds[:, f], y_true[f])
        L_B = _loss_regression_mse(w_B, preds[:, f], y_true[f])
        losses = np.array([L_A, L_B], dtype=np.float64)

        if not np.isfinite(losses).all():
            continue

        # 做一个平移（减去最小值）来增强数值稳定性
        losses -= losses.min()
        alpha *= np.exp(-eta * losses)
        alpha = np.maximum(alpha, 1e-12)
        alpha /= alpha.sum()

    # 用几何池化融合两种策略给出的基模型权重，然后做均匀收缩
    w_geo = _geometric_pool([w_A, w_B], alpha)
    w_final = _uniform_shrinkage(w_geo, shrink)

    info = {"alpha": alpha, "K": K, "eta": eta, "w_geo": w_geo}
    return w_final, info


# ----------------------- 单阶段正则化优化：方法C -----------------------

def optimize_weights_regularized_reg(
    models: List, val_x, val_y,
    D: Optional[np.ndarray] = None,
    J: Optional[np.ndarray] = None,
    R_matrix: Optional[np.ndarray] = None,
    init_w: Optional[np.ndarray] = None,
    steps: int = 400, lr: float = 0.5,
    lambda_J: float = 0.0, lambda_D: float = 0.0, mu_R: float = 0.0, eta_l2: float = 0.0,
    shrink: float = 0.1
) -> Tuple[np.ndarray, Dict]:
    """
    方法C（回归，PGD）：最小化
        L_MSE(w) + λ_J·w^T J w − λ_D·w^T D w − μ·(out^T w) + η·||w||_2^2
      其中 out = R_matrix.sum(axis=1)
    返回 (w_final, info)
    """
    N = len(models)
    if init_w is None:
        init_w = np.ones(N, dtype=np.float64) / N
    w = _project_to_simplex(init_w)

    preds = _collect_outputs_regression(models, val_x)  # (N,M)
    y_true = np.asarray(val_y, dtype=np.float64).ravel()
    M = y_true.shape[0]

    for _ in range(steps):
        y_hat = (w[:, None] * preds).sum(axis=0)  # (M,)
        # d/dw_i [1/M Σ (y - Σ w·ŷ)^2] = (-2/M) Σ (y - y_hat)·ŷ_i
        grad = (-2.0 / M) * (preds * (y_true - y_hat)[None, :]).sum(axis=1)

        if J is not None:
            grad += 2.0 * lambda_J * (J @ w)
        if D is not None:
            grad += -2.0 * lambda_D * (D @ w)
        if R_matrix is not None and mu_R != 0.0:
            out = R_matrix.sum(axis=1)
            grad += -mu_R * out
        if eta_l2 != 0.0:
            grad += 2.0 * eta_l2 * w

        w = _project_to_simplex(w - lr * grad)

    w_final = _uniform_shrinkage(w, shrink)
    info = {"steps": steps, "lr": lr, "lambda_J": lambda_J, "lambda_D": lambda_D, "mu_R": mu_R, "eta_l2": eta_l2}
    return w_final, info

# ----------------------- 一键接口（回归） -----------------------

def integrate_three_strategies_reg(
    weight_vector_list: List[np.ndarray],
    models: List, val_x, val_y,
    method: str = "softmax-geo",
    **kwargs
) -> Tuple[np.ndarray, Dict]:
    """
    回归任务入口：
      method ∈ {"softmax-geo", "hedge-geo", "regularized-opt"}
    额外参数（按方法）：
      - softmax-geo: tau, shrink
      - hedge-geo  : K, eta, shrink, seed
      - regularized-opt: D, J, R_matrix, init_w, steps, lr,
                         lambda_J, lambda_D, mu_R, eta_l2, shrink
    """
    # 兼容 ('softmax-geo',) 传法
    if isinstance(method, (tuple, list)):
        if len(method) != 1:
            raise TypeError("method 传入 tuple/list 时应为单元素")
        method = method[0]
    if not isinstance(method, str):
        raise TypeError("method 应为字符串，例如 'softmax-geo' | 'hedge-geo' | 'regularized-opt'")

    method = method.lower()
    w_D = weight_vector_list[0]
    w_J = weight_vector_list[1]
    w_R = weight_vector_list[2]
    # print("W_D:", w_D)
    # print("W_J:", w_J)
    # print("W_R:", w_R)

    if method == "softmax-geo":
        return integrate_softmax_geometric_reg(
            w_D, w_J, w_R, models, val_x, val_y,
            tau=kwargs.get("tau", 10.0),
            shrink=kwargs.get("shrink", 0.1),
        )
    elif method == "hedge-geo":
        return integrate_hedge_geometric_reg(
            w_D, w_J, w_R, models, val_x, val_y,
            K=kwargs.get("K", 5),
            eta=kwargs.get("eta", None),
            shrink=kwargs.get("shrink", 0.1),
            seed=kwargs.get("seed", 42),
        )
    elif method == "regularized-opt":
        return optimize_weights_regularized_reg(
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
        )
    else:
        raise ValueError("Unknown method. Choose from {'softmax-geo','hedge-geo','regularized-opt'}.")


# ----------------------- 使用示例 -----------------------
# weight_vector_list = [w_D, w_J, w_R]  # 三种策略各自的权重（长度均为 N，非负；函数内部会投影到单纯形）
# w_final, info = integrate_three_strategies_reg(
#     weight_vector_list,
#     best_fitted_model_instance_list,  # models
#     val_x, val_y,
#     method="softmax-geo", tau=10.0, shrink=0.1
# )
# 或：
# w_final, info = integrate_three_strategies_reg(
#     weight_vector_list,
#     best_fitted_model_instance_list,
#     val_x, val_y,
#     method="regularized-opt",
#     D=D_matrix, J=J_matrix, R_matrix=R_matrix,
#     lambda_J=0.1, lambda_D=0.1, mu_R=0.2, eta_l2=0.01,
#     steps=500, lr=0.3, shrink=0.1
# )
