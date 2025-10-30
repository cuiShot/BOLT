import pandas as pd
from typing import List, Optional, Sequence
import math
import numpy as np

def get_regression_param_prompt(
        best_code, best_rmse, dataset_description,
        X_test, feature_columns, dataset_name=None, max_rows=10
    ):
    """生成适用于LLM回归模型参数优化的提示词（无参数约束版本）"""
    if isinstance(X_test, pd.DataFrame):
        df_show = X_test.copy()
    else:
        df_show = pd.DataFrame(X_test, columns=feature_columns)

    table = df_show.head(max_rows).to_string(index=False)
    data_shape = df_show.shape if hasattr(df_show, 'shape') else (len(df_show), len(feature_columns))
    if dataset_name is None:
        dataset_name = "unknown"

    # 计算特征统计信息，增强数据理解
    numeric_features = df_show.select_dtypes(include=['number']).columns.tolist()
    feature_stats = []
    for col in numeric_features[:5]:  # 限制展示前5个数值特征的统计信息
        stats = f"{col}: mean={df_show[col].mean():.2f}, std={df_show[col].std():.2f}, range=[{df_show[col].min():.2f}, {df_show[col].max():.2f}]"
        feature_stats.append(stats)
    feature_stats_str = "\n".join(feature_stats) if feature_stats else "No numeric features available"

    prompt = (
        f"Your task is to optimize hyperparameters of the provided regression model to minimize 5-fold cross-validation RMSE.\n\n"
        f"Current best model (RMSE: {best_rmse:.4f}):\n"
        f"```python\n{best_code}\n```\n\n"
        f"Task Context:\n"
        f"- Dataset: {dataset_name}\n"
        f"- Description: {dataset_description}\n"
        f"- Feature columns: {', '.join(feature_columns)}\n"
        f"- Dataset shape: {data_shape} (samples × features)\n"
        f"- Key numeric feature statistics (first 5):\n{feature_stats_str}\n"
        f"- First {max_rows} rows of data:\n{table}\n\n"
        "Optimization Guidelines:\n"
        "1. Focus exclusively on hyperparameter tuning - DO NOT change the algorithm type or model architecture\n"
        "2. Explore diverse hyperparameter combinations to find the optimal balance between:\n"
        "   - Predictive performance (lower RMSE)\n"
        "   - Generalization ability (avoid overfitting)\n"
        "   - Computational efficiency\n"
        "3. Prioritize multi-dimensional adjustments across different parameter categories:\n"
        "   - Model complexity parameters\n"
        "   - Regularization strength parameters\n"
        "   - Optimization/learning parameters\n"
        "   - Sampling/ensemble parameters (where applicable)\n"
        "4. Consider the dataset characteristics when tuning (size, feature types, noise level)\n"
        "5. Ensure full compatibility with scikit-learn 1.6.1 using only valid, non-deprecated parameters\n\n"
        "Output Requirements:\n"
        "ONLY return the optimized Python class code with:\n"
        "- __init__ method with explicit, adjustable hyperparameters\n"
        "- fit method that trains the model on input data\n"
        "- predict method that returns regression predictions\n"
        "No explanations, comments, markdown, or additional text - just clean, executable code."
    )

    return prompt

# ========================= 回归版：预测差异矩阵 =========================
def get_model_pre_differences(
    best_fitted_model_instance_list: List,
    val_x,
    val_y,
    metric: str = "pearson",   # "pearson" 或 "mae_norm"
    eps: float = 1e-12,
    scale_ref: str = "y"       # 当 metric="mae_norm" 时，归一化尺度参考："y" 或 "avg_pred"
) -> np.ndarray:
    """
    回归任务的“预测差异矩阵”（Regression Prediction Difference Matrix）。
    返回 N×N 对称矩阵，数值越大表示两模型预测越不同（互补潜力越高）。

    两种度量（通过 metric 指定）：
      1) "pearson"（默认）：Diff[i,j] = (1 - corr(ŷ_i, ŷ_j)) / 2 ∈ [0,1]
         - 完全相同 → 0；强负相关 → 1；无关 → 约 0.5。
      2) "mae_norm"：Diff[i,j] = mean(|ŷ_i - ŷ_j|) / S ，再裁剪到 [0,1]
         - S 为尺度：scale_ref="y" 用 y 的 MAD（更稳健）；"avg_pred" 用(ŷ_i+ŷ_j)/2 的 MAD。

    参数:
      metric: "pearson" 或 "mae_norm"
      eps: 数值稳定项
      scale_ref: 仅当 metric="mae_norm" 生效

    注意：
      - 对角线恒为 0。
      - 若某个模型的预测是常数（方差≈0），pearson 分支将按以下规则处理：
          · 两常数且近似相等 → corr=1 → Diff=0
          · 其它情况 → corr=0 → Diff≈0.5
    """
    models = best_fitted_model_instance_list
    n = len(models)
    # 收集预测
    preds = [np.asarray(m.predict(val_x), dtype=np.float64).ravel() for m in models]
    preds = np.vstack(preds)  # (N, M)
    M = preds.shape[1]

    matrix = np.zeros((n, n), dtype=np.float64)

    if metric == "pearson":
        # 逐对计算 Pearson 相关并映射到 [0,1]
        for i in range(n):
            yi = preds[i]
            yi_mean = yi.mean()
            yi_std = yi.std(ddof=0)
            for j in range(i + 1, n):
                yj = preds[j]
                yj_mean = yj.mean()
                yj_std = yj.std(ddof=0)
                # 常数向量的处理
                if yi_std <= eps or yj_std <= eps:
                    if np.allclose(yi, yj, atol=1e-9, rtol=0.0):
                        corr = 1.0   # 完全一致的常数
                    else:
                        corr = 0.0   # 一个或两个几乎常数，但不一致 → 视作无关
                else:
                    num = ((yi - yi_mean) * (yj - yj_mean)).sum()
                    den = M * yi_std * yj_std
                    corr = float(np.clip(num / (den + eps), -1.0, 1.0))

                diff = (1.0 - corr) / 2.0
                matrix[i, j] = matrix[j, i] = np.clip(diff, 0.0, 1.0)

    elif metric == "mae_norm":
        # 使用归一化 MAE 作为差异度; 归一化尺度 S 使用 MAD（稳健）
        y_true = np.asarray(val_y, dtype=np.float64).ravel()
        # 稳健尺度函数
        def _mad(v):
            med = np.median(v)
            return np.median(np.abs(v - med)) + eps

        for i in range(n):
            yi = preds[i]
            for j in range(i + 1, n):
                yj = preds[j]
                mae = float(np.mean(np.abs(yi - yj)))
                if scale_ref == "avg_pred":
                    base = 0.5 * (yi + yj)
                    S = _mad(base)
                else:
                    S = _mad(y_true)
                val = mae / S
                matrix[i, j] = matrix[j, i] = float(np.clip(val, 0.0, 1.0))
    else:
        raise ValueError("metric must be 'pearson' or 'mae_norm'.")

    # 对角线为 0
    np.fill_diagonal(matrix, 0.0)
    return matrix


# ========================= 回归版：Jaccard-Fault 矩阵 =========================
def get_model_jaccard_fault(
    best_fitted_model_instance_list: List,
    val_x,
    val_y,
    high_err_quantile: float = 0.8
) -> np.ndarray:
    """
    回归任务的 Jaccard-Fault 矩阵（高误差重叠）。
    定义每个模型的“高误差集合”：
        E_i = { k | |y_k - ŷ_i(x_k)| > T_i }
      其中 T_i 是该模型绝对残差 |r_i| 的分位阈值（默认 0.8 分位）。

    返回：
      N×N 对称矩阵 J，J[i,j] = |E_i ∩ E_j| / |E_i ∪ E_j|  ∈ [0,1]
        - 若 |E_i ∪ E_j| = 0（两者都没有高误差样本），定义 J[i,j] = 0
        - 对角线：若该模型存在至少一个高误差样本，则 J[i,i] = 1；否则 0

    说明：
      - 使用“各自阈值 T_i”的好处是对齐每个模型的误差尺度，避免强者/弱者的偏置。
      - 若你更希望用“全局阈值”，可把 T_i 替换成全模型合并后 |r| 的统一分位数。
    """
    models = best_fitted_model_instance_list
    n = len(models)
    y_true = np.asarray(val_y, dtype=np.float64).ravel()
    # 预测与绝对残差
    preds = [np.asarray(m.predict(val_x), dtype=np.float64).ravel() for m in models]
    preds = np.vstack(preds)  # (N, M)
    abs_resid = np.abs(y_true[None, :] - preds)  # (N, M)

    # 各模型的高误差阈值 T_i
    T = np.quantile(abs_resid, high_err_quantile, axis=1)  # (N,)

    # 高误差指示矩阵 E
    E = (abs_resid > T[:, None]).astype(np.float64)  # (N, M)

    # 交集与并集计数
    inter_counts = E @ E.T                          # (N, N)
    err_counts = E.sum(axis=1)                      # (N,)
    union_counts = err_counts[:, None] + err_counts[None, :] - inter_counts  # (N, N)

    with np.errstate(divide='ignore', invalid='ignore'):
        J = inter_counts / union_counts
        J = np.where(union_counts == 0.0, 0.0, J)

    # 对角线：是否存在至少一个高误差样本
    has_high_err = (err_counts > 0)
    np.fill_diagonal(J, 0.0)
    for i, flag in enumerate(has_high_err):
        J[i, i] = 1.0 if flag else 0.0

    np.clip(J, 0.0, 1.0, out=J)
    return J

# 功能：获取“预测差异矩阵”的提示词（回归版）
def get_model_performance_differences_prompt(model_difference_matrix):
    """
    获取回归任务下的“预测差异矩阵”提示词。
    :param model_difference_matrix: N x N 对称矩阵（numpy array 或嵌套 list）
        - 该矩阵可由 (1 - Pearson 相关)/2 或 归一化 MAE 等方式计算得到，数值范围通常在 [0,1]。
        - 数值越大表示两模型在验证集上的预测差异越大（互补潜力越强）。
    :return: 字符串，描述性提示词
    """
    n = len(model_difference_matrix)

    # 格式化矩阵为字符串（保留 4 位小数）
    matrix_str = "[\n"
    for row in model_difference_matrix:
        formatted_row = ", ".join(f"{val:.4f}" for val in row)
        matrix_str += f" [{formatted_row}],\n"
    matrix_str = matrix_str.rstrip(",\n") + "\n]"

    prompt = (
        "The pairwise prediction divergence between regression models on an independent validation set "
        "is given by the matrix below. Each element [i][j] denotes a normalized prediction difference "
        "between model i and model j (e.g., (1 - Pearson correlation)/2 or a normalized MAE-based score). "
        "A larger value indicates greater divergence in predictions and thus stronger potential complementarity.\n"
        "The model order in the matrix follows the list above (Model 1, Model 2, ...), and the matrix is 1-indexed "
        "to match the model numbering.\n"
        f"The prediction-difference matrix is as follows (unit: normalized score in [0,1], rounded to four decimals):\n{matrix_str}\n"
    )
    return prompt


# 功能：获取“Jaccard-Fault 矩阵（高误差重叠）”的提示词（回归版）
def get_model_intersection_union_prompt(model_jaccard_fault_matrix):
    """
    获取回归任务下的 Jaccard-Fault（高误差重叠）矩阵提示词。
    :param model_jaccard_fault_matrix: N x N 对称矩阵（numpy array 或嵌套 list）
        - 对每个模型 i，将其在验证集上的“高误差样本集合” E_i（例如 |残差| 超过该模型的分位阈值）进行比较，
          定义 J[i][j] = |E_i ∩ E_j| / |E_i ∪ E_j|。
        - 数值越小表示两模型的“高误差”重叠越少，互补性越强；对角线可表示该模型是否存在高误差样本。
    :return: 字符串，描述性提示词
    """
    n = len(model_jaccard_fault_matrix)

    # 格式化矩阵为字符串（保留 4 位小数）
    matrix_str = "[\n"
    for row in model_jaccard_fault_matrix:
        formatted_row = ", ".join(f"{val:.4f}" for val in row)
        matrix_str += f" [{formatted_row}],\n"
    matrix_str = matrix_str.rstrip(",\n") + "\n]"

    prompt = (
        "The high-error overlap (Jaccard-Fault for regression) on an independent validation set is shown below. "
        "Each element [i][j] denotes the Jaccard similarity between the high-error sets of model i and model j "
        "(e.g., samples where absolute residual exceeds a model-specific threshold). "
        "Smaller values indicate less overlap of high-error cases and thus stronger complementarity.\n"
        "The model order in the matrix follows the list above (Model 1, Model 2, ...), and the matrix is 1-indexed "
        "to match the model numbering.\n"
        f"The Jaccard-Fault matrix is as follows (unit: proportion in [0,1], rounded to four decimals):\n{matrix_str}\n"
    )
    return prompt

# ========================= 回归版：Rescue-Confidence 矩阵 =========================
def get_rescue_confidence_matrix(
    best_fitted_model_instance_list: List,
    val_x,
    val_y,
    high_err_quantile: float = 0.8,
    eps: float = 1e-12,
    weight_power: float = 1.0,
    use_global_threshold: bool = False,
    robust_scale: bool = False,
) -> np.ndarray:
    """
    回归任务的“救援-置信矩阵” R（非对称），R[i, j] 表示：
      当模型 j 在其“高误差样本”上表现很差时，模型 i 能把这些样本的误差降低到何种程度（0~1）。

    计算思路（天然 ∈ [0,1]）：
      1) 先确定模型 j 的“高误差样本集合” hard_j：
         - 默认用“各自阈值” T_j = quantile(|r_j|, high_err_quantile)；
         - 若 use_global_threshold=True，则用所有模型残差合并后的全局阈值 T。
      2) 在 hard_j 上，对每个样本 k 计算“分数化救援率”
           frac_improve_{i,j,k} = max(0, 1 - |r_{i,k}| / (|r_{j,k}| + eps))
         该值 ∈ [0,1]，越接近 1 表示 i 将 j 的大误差几乎“救没了”。
      3) 按权重 w_k 对以上分数做加权平均得到 R[i,j]：
           w_k = |r_{j,k}| ** weight_power   （默认 p=1，越强调 j 的大误差样本）
      4) R 的对角线置 0（i 不救自己），R 为非对称矩阵。

    参数
    ----
    best_fitted_model_instance_list : List
        已训练好的回归模型列表，每个需实现 .predict(val_x)
    val_x, val_y : 验证集特征与标签（y 为连续值）
    high_err_quantile : float
        判定“高误差样本”的分位数阈值（常用 0.8~0.9）
    eps : float
        数值稳定项，避免 0 除
    weight_power : float
        样本加权的指数 p，w_k = |r_{j,k}| ** p；p 越大越强调“极大误差样本”
    use_global_threshold : bool
        是否使用“全局阈值”（所有模型残差合并后统一阈值）；默认 False = 各自阈值
    robust_scale : bool
        是否对每个模型的残差使用稳健尺度（MAD）进行标准化后再判定高误差

    返回
    ----
    R : np.ndarray (N, N)
        非对称矩阵，R[i, j] ∈ [0, 1]，对角线为 0。数值越大表示 i 对 j 的“救援能力”越强。
    """
    models = best_fitted_model_instance_list
    N = len(models)

    # 收集预测并统一形状
    y_true = np.asarray(val_y, dtype=np.float64).ravel()
    preds = []
    for m in models:
        y_hat = np.asarray(m.predict(val_x), dtype=np.float64).ravel()
        if not np.all(np.isfinite(y_hat)):
            y_hat = np.nan_to_num(y_hat, nan=0.0, posinf=0.0, neginf=0.0)
        preds.append(y_hat)
    preds = np.asarray(preds)  # (N, M)

    # 绝对残差
    abs_resid = np.abs(y_true[None, :] - preds)  # (N, M)

    # 可选：稳健尺度标准化（按模型维度），提高不同模型间的可比性
    if robust_scale:
        # MAD: median(|x - median(x)|)
        med = np.median(abs_resid, axis=1, keepdims=True)
        mad = np.median(np.abs(abs_resid - med), axis=1, keepdims=True) + eps
        abs_resid = abs_resid / mad

    # 阈值：各自 or 全局
    if use_global_threshold:
        T_global = float(np.quantile(abs_resid, high_err_quantile))
        thresholds = np.full((N,), T_global, dtype=np.float64)
    else:
        thresholds = np.quantile(abs_resid, high_err_quantile, axis=1)  # (N,)

    R = np.zeros((N, N), dtype=np.float64)

    for j in range(N):
        rj = abs_resid[j]  # (M,)
        hard_mask = (rj > thresholds[j])

        if not np.any(hard_mask):
            # j 没有高误差样本 → 整列置 0（代表“无需救援”）
            continue

        # 加权：w_k = |r_{j,k}| ** p
        w = (rj[hard_mask] ** float(weight_power))
        denom = w.sum() + eps

        # 分数化救援率：1 - |r_i| / (|r_j| + eps)，下界为 0
        with np.errstate(divide="ignore", invalid="ignore"):
            frac_improve = np.maximum(
                0.0,
                1.0 - (abs_resid[:, hard_mask] / (rj[hard_mask][None, :] + eps)),
            )
            frac_improve[~np.isfinite(frac_improve)] = 0.0  # 极端数值兜底

        # 按样本权重做加权平均（逐 i）
        num = (frac_improve * w[None, :]).sum(axis=1)  # (N,)
        R[:, j] = num / denom

    # 非对称：i→j；对角线置 0；范围保护
    np.fill_diagonal(R, 0.0)
    np.clip(R, 0.0, 1.0, out=R)
    return R


# 功能：根据“救援-置信矩阵（回归版）”生成提示词
def get_rescue_confidence_prompt(rescue_confidence_matrix):
    """
    获取回归任务下的“Rescue-Confidence 矩阵”提示词（非对称矩阵）。
    :param rescue_confidence_matrix: N x N 非对称矩阵（numpy array 或嵌套 list）
        - 元素 [i][j] 衡量：当模型 j 在某些样本上出现“高误差”时，模型 i 在这些样本上能将误差
          降低多少（可按 j 的误差幅度/权重进行归一化）。数值越大表示 i 对 j 的“救援”能力越强。
        - 该矩阵具有方向性（i→j），通常对角线为 0。
    :return: 字符串，描述性提示词
    """
    n = len(rescue_confidence_matrix)

    # 格式化矩阵为字符串（保留 4 位小数）
    matrix_str = "[\n"
    for row in rescue_confidence_matrix:
        formatted_row = ", ".join(f"{val:.4f}" for val in row)
        matrix_str += f" [{formatted_row}],\n"
    matrix_str = matrix_str.rstrip(",\n") + "\n]"

    prompt = (
        "The Rescue-Confidence matrix for regression is shown below. "
        "Each element [i][j] quantifies how effectively model i reduces the error on the validation samples "
        "where model j exhibits high errors (often weighted by j’s error magnitude). "
        "Larger values mean model i is more capable of correcting model j’s high-error cases. "
        "This matrix is directional (i rescues j), and typically the diagonal is zero.\n"
        "The model order in the matrix follows the list above (Model 1, Model 2, ...), and the matrix is 1-indexed "
        "to match the model numbering.\n"
        f"The Rescue-Confidence matrix is as follows (unit: normalized score in [0,1], rounded to four decimals):\n{matrix_str}\n"
    )
    return prompt


def generate_llm_weight_prompt(
    base_prompt=None,
    dataset_description_prompt=None,
    model_code_prompt=None,
    model_performance_differences_prompt=None,  # 回归版：预测差异矩阵（如 (1-Pearson)/2 或归一化MAE）
    model_double_error_prompt=None,              # 可留空；若有“误差相关矩阵/双高误差”等说明，可放这里
    model_Jaccard_Fault_prompt=None,             # 回归版：高误差重叠（Jaccard-Fault）
    model_rescue_confidence_prompt=None,         # 回归版：Rescue–Confidence（非对称）
    single_shot_prompt_D=None,                   # 针对“预测差异矩阵”的 one-shot 示例（回归）
    single_shot_prompt_J=None,                   # 针对“Jaccard-Fault（回归）”的 one-shot 示例
    single_shot_prompt_R=None,                   # 针对“Rescue–Confidence（回归）”的 one-shot 示例
):
    """
    Build the final LLM prompt for assigning ensemble weights (Regression version).
    All instructions and outputs must be in English.

    Notes for regression:
      - Primary objective: minimize RMSE on the validation set.
      - Secondary: MAE, then R^2; include rank-correlation (Spearman/Pearson) if available.
      - Consider diversity signals: prediction-difference matrix (e.g., (1-Pearson)/2),
        high-error overlap (Jaccard-Fault for regression),
        and Rescue–Confidence (directional ability to fix another model’s high-error cases).
      - Weights must be non-negative, 2 decimals, and sum EXACTLY to 1.00 after rounding.
    """

    parts = []

    # 1) 可选：全局/基础提示（回归任务说明、评价指标、约束等）
    if base_prompt is not None:
        parts.append(base_prompt.strip())

    # 2) 固定的输出格式与硬性约束
    parts.append(
        """
You must output the following three parts:
(1) Output the weight for each model. A model's weight may be 0. Weights must be non-negative and keep TWO decimals.
(2) Briefly explain the reason for each model's weight, referencing metrics and diversity signals when relevant.
(3) On the LAST line, output ONLY a weight list in the format:
    Final model weight list: [0.13, 0.22, 0.00, 0.33, 0.32]
IMPORTANT:
- Ensure the weights SUM TO EXACTLY 1.00 AFTER ROUNDING to two decimals.
- The last line will be parsed automatically. STRICTLY follow the exact format above.
""".strip()
    )

    # 3) 可选：数据集/特征/任务描述
    if dataset_description_prompt is not None:
        parts.append(dataset_description_prompt.strip())

    # 4) 可选：模型/代码摘要
    if model_code_prompt is not None:
        parts.append(model_code_prompt.strip())

    # 5) 可选：预测差异矩阵（回归）
    if model_performance_differences_prompt is not None:
        parts.append(model_performance_differences_prompt.strip())
        if single_shot_prompt_D is not None:
            parts.append(
                f"""single-shot example for using the prediction-difference matrix:
{{
{single_shot_prompt_D.strip()}
}}"""
            )

    # 6) 可选：误差相关/双高误差等（若你有单独的说明，可放这里）
    if model_double_error_prompt is not None:
        parts.append(model_double_error_prompt.strip())

    # 7) 可选：Jaccard-Fault（回归的高误差重叠）
    if model_Jaccard_Fault_prompt is not None:
        parts.append(model_Jaccard_Fault_prompt.strip())
        if single_shot_prompt_J is not None:
            parts.append(
                f"""single-shot example for using the high-error-overlap (Jaccard-Fault):
{{
{single_shot_prompt_J.strip()}
}}"""
            )

    # 8) 可选：Rescue–Confidence（回归，非对称）
    if model_rescue_confidence_prompt is not None:
        parts.append(model_rescue_confidence_prompt.strip())
        if single_shot_prompt_R is not None:
            parts.append(
                f"""single-shot example for using the Rescue-Confidence matrix:
{{
{single_shot_prompt_R.strip()}
}}"""
            )

    # 9) 最后再次强调输出格式
    parts.append(
        "FINAL REMINDER: The very last line must exactly match this pattern:\n"
        "Final model weight list: [w1, w2, w3, ...]\n"
        "Ensure two decimals per weight and the sum equals 1.00."
    )

    return "\n\n".join(parts)


def get_model_code_prompt(
    model_code_list: Optional[Sequence[str]] = None,
    val_MAE_list: Optional[Sequence[float]] = None,
    val_RMSE_list: Optional[Sequence[float]] = None,
    val_RMSLE_list: Optional[Sequence[float]] = None,
    use_code: bool = False,
) -> str:
    """
    获取“模型清单 + 验证集表现”的提示词（回归版）。

    参数
    ----
    model_code_list : 可选，模型代码/配置的字符串列表；若 use_code=True 则会包含在提示词中
    val_MAE_list    : 可选，验证集 MAE 列表（与模型顺序对齐）
    val_RMSE_list   : 可选，验证集 RMSE 列表（与模型顺序对齐；建议作为主要指标）
    val_RMSLE_list  : 可选，验证集 RMSLE 列表（与模型顺序对齐；仅当 y>0 时有意义）
    use_code        : 是否将模型代码一并输出到提示词

    返回
    ----
    model_code_prompt : str
        形如：
        Model list:
        model1:
        - Validation set performance: RMSE=..., MAE=..., RMSLE=...
        model2:
        ...

    说明
    ----
    - 自动对齐各列表长度；缺失或 NaN 的指标会被跳过。
    - 数值统一格式化为 4 位小数。
    """

    def _safe_len(x):
        return len(x) if x is not None else 0

    # 确定模型个数 n：取所有已提供列表的最大长度
    n = max(
        _safe_len(model_code_list),
        _safe_len(val_MAE_list),
        _safe_len(val_RMSE_list),
        _safe_len(val_RMSLE_list),
        0,
    )

    if n == 0:
        return "Model list:\n"  # 没有可列出的信息

    def _fmt_metric(name: str, seq: Optional[Sequence[float]], i: int) -> Optional[str]:
        if seq is None or i >= len(seq):
            return None
        val = seq[i]
        if val is None:
            return None
        try:
            if isinstance(val, (float, int)) and (not math.isnan(float(val))):
                return f"{name}={float(val):.4f}"
        except Exception:
            return None
        return None

    model_code_prompt = "Model list:\n"
    for i in range(n):
        model_code_prompt += f"model{i + 1}:\n"

        # 可选：输出模型代码/配置
        if use_code and model_code_list is not None and i < len(model_code_list):
            code = model_code_list[i]
            if code is not None:
                model_code_prompt += f"code: {code}\n"

        # 收集该模型的有效指标（按 RMSE, MAE, RMSLE 的顺序）
        metrics = []
        m_rmse = _fmt_metric("RMSE", val_RMSE_list, i)
        m_mae  = _fmt_metric("MAE",  val_MAE_list,  i)
        m_rmsle= _fmt_metric("RMSLE",val_RMSLE_list,i)

        for m in (m_rmse, m_mae, m_rmsle):
            if m is not None:
                metrics.append(m)

        if metrics:
            model_code_prompt += f"- Validation set performance: {', '.join(metrics)}\n"

    return model_code_prompt


def build_prompt_samples(df):
    samples = ""
    df_ = df.head(5)
    for i in list(df_):
        # show the list of values
        s = df_[i].tolist()
        # 如果该列的数据类型是 float64（浮点数），就把前10个样本值四舍五入保留两位小数
        if str(df[i].dtype) == "float64":
            s = [round(sample, 2) for sample in s]
        # 构建一行字符串，表示当前列的列名、数据类型、缺失值频率，以及样本值列表，并把它追加到 samples 这个总字符串中
        samples += (
            f"{df_[i].name} ({df[i].dtype}): Samples {s}\n"
        )
    return samples


"""
回归任务模型生成的提示词
"""
def get_regression_model_prompt(
        ds_description=None
):
    prompt = f'''
    The dataset description for regression task is as follows:
    {ds_description}

    Based on this information, you must:
    1. Generate a new Python regressor class named `myregressor` each round.
    2. The model should differ from previous versions (in model type or structure).
    3. The goal is to minimize RMSE on the given test data.

    After each iteration, I will return the current RMSE value, the lowest RMSE value in all previous iterations, and the regressor model code when the lowest RMSE value was achieved in all previous iterations.

    The class must support the following methods:
    ```python
    model = myregressor()                          # Initialize the regressor
    model.fit(train_aug_x, train_aug_y)            # Train the model (both are pandas DataFrames)
    pred = model.predict(test_aug_x)               # Predict regression values (returns 1D array or list)
    ```
    
    Important instructions (must follow strictly):
    - Only output the complete Python class named myregressor (no explanation, no comments, no extra code).
    - The class must be ready to use in Python (no missing imports or undefined variables).
    - You should lean more towards exploring powerful regressors, such as LightGBM, CatBoost, XGBoost, or ensemble/regression classification_stacking methods.
    - train_aug_x, train_aug_y and test_aug_x are all pandas DataFrames.
    - The predict method must return a 1D array or list of predicted values (using the format: return self.model.predict(test_aug_x), do not convert it to a python list using .tolist()).
    - Each new version must differ from the previous one (by model type or structure).
    - **You must NOT use `LightGBM` under any circumstances. It is forbidden due to compatibility issues.**
    - After I provide the current RMSE and the best (lowest) RMSE in previous iterations, improve the model in the next round accordingly.
    '''

    return prompt