import os
import sys
project_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__),   # 当前文件所在目录
                 "..", "..")                  # 向上跳三级
)
sys.path.append(project_root)

import re
import numpy as np
import pandas as pd
import pickle
import torch
from openai import OpenAI
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.base import BaseEstimator, ClassifierMixin
from typing import List
import copy



import os
import pickle
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split


def load_origin_data(dataset_name, seed=0):
    # 需要走 .pkl 的旧数据集关键词（子串匹配）
    old_keys = ('credit','cd1','cc1','ld1','cc2','cd2','cf1','balance-scale')
    name_l = dataset_name.lower()
    is_old = any(k in name_l for k in old_keys)

    if is_old:
        loc = f"{project_root}/tests/data/{dataset_name}.pkl"
        with open(loc, 'rb') as f:
            ds = pickle.load(f)

        # credit 系列需要先 split，其它旧数据集直接 ds[1]/ds[2]
        if 'credit' in name_l:
            ds, df_train, df_test = get_data_split(ds, seed=seed)
        else:
            df_train, df_test = ds[1], ds[2]

        target_column_name = ds[4][-1]
        dataset_description = ds[-1]
        return df_train, df_test, target_column_name, dataset_description

    # 否则读取新的 CSV 数据集
    base_loc = f"{project_root}/new_dataSet/binaryclass/"
    return load_new_dataset(dataset_name, base_loc=base_loc, seed=seed, test_size=0.2)


from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder

def _normalize_target_to_int(s: pd.Series) -> pd.Series:
    """把目标列统一转成整型标签：二分类'yes/no'等→0/1；其它情况→LabelEncoder的0..K-1。"""
    # 丢掉首尾空白，统一小写字符串视图（不改变原值）
    s_str = s.astype("string").str.strip().str.lower()

    # 常见二分类文案映射
    map01 = {
        "yes": 1, "y": 1, "true": 1, "t": 1, "positive": 1, "pos": 1, "1": 1,
        "no": 0,  "n": 0, "false": 0,"f": 0, "negative": 0, "neg": 0, "0": 0
    }

    # 纯数值/布尔直接转
    if pd.api.types.is_bool_dtype(s):
        return s.astype(int)
    if pd.api.types.is_numeric_dtype(s):
        # 若是0/1浮点，直接转；否则四舍五入（按需可改）
        uniq = set(pd.unique(s.dropna()))
        if uniq <= {0, 1, 0.0, 1.0}:
            return s.astype(int)
        return s.round().astype(int)

    # 纯二分类文案时，按映射转
    non_na = s_str.dropna()
    if not non_na.empty and non_na.isin(map01).all():
        return s_str.map(map01).astype(int)

    # 其他情况（含多分类/混合文案）：LabelEncoder 到 0..K-1
    le = LabelEncoder()
    enc = le.fit_transform(s_str.fillna("__missing__"))
    return pd.Series(enc, index=s.index, dtype=int)

def get_data_split(ds, seed):
    def get_df(X, y):
        df = pd.DataFrame(
            data=np.concatenate([X, np.expand_dims(y, -1)], -1), columns=ds[4]
        )
        cat_features = ds[3]
        for c in cat_features:
            if len(np.unique(df.iloc[:, c])) > 50:
                cat_features.remove(c)
                continue
            df[df.columns[c]] = df[df.columns[c]].astype("int32")
        return df.infer_objects()

    ds = copy.deepcopy(ds)

    X = ds[1].numpy() if type(ds[1]) == torch.Tensor else ds[1]
    y = ds[2].numpy() if type(ds[2]) == torch.Tensor else ds[2]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed
    )

    df_train = get_df(X_train, y_train)
    df_test = get_df(X_test, y_test)
    df_train.iloc[:, -1] = df_train.iloc[:, -1].astype("category")
    df_test.iloc[:, -1] = df_test.iloc[:, -1].astype("category")

    return ds, df_train, df_test


def load_new_dataset(dataset_name: str,
                     base_loc: str = f"{project_root}/AutoML_pipline/new_dataSet/binaryclass/",
                     seed: int = 42,
                     test_size: float = 0.2):
    """
    读取 {base_loc}/{dataset_name}.csv，做最小预处理并切分训练/测试。
    返回: df_train, df_test, target_column_name, dataset_description
    - df_* 含目标列（最后一列），不拆 X/y
    - 预处理仅作用于特征列，目标列转为 int 并保留在最后
    """
    # 1) 读数据
    loc = os.path.join(base_loc, dataset_name + ".csv")
    df = pd.read_csv(loc).convert_dtypes()

    # 2) 目标列名 = 最后一列；去掉没有标签的行
    target_column_name = df.columns[-1]
    df = df[~df[target_column_name].isna()].copy()

    # 3) 先把目标列规范为整型（防止后续 to_pd 里 astype(int) 报错）
    df[target_column_name] = _normalize_target_to_int(df[target_column_name])

    # 4) 先切分整表（不泄漏）；这里用普通随机切分，若需分层可加自定义分层
    df_train_raw, df_test_raw = train_test_split(df, test_size=test_size, random_state=seed)

    # 5) 仅对特征列做预处理（数值填0；类别序数编码，未知→-1）
    feature_cols = [c for c in df.columns if c != target_column_name]
    X_train = df_train_raw[feature_cols]
    X_test  = df_test_raw[feature_cols]

    num_cols = X_train.select_dtypes(include=["number", "boolean"]).columns.tolist()
    cat_cols = [c for c in feature_cols if c not in num_cols]

    num_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value=0)),
    ])
    cat_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="__MISSING__")),
        ("encode", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
    ])

    preproc = ColumnTransformer(
        transformers=[
            ("num", num_pipe, num_cols),
            ("cat", cat_pipe, cat_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    X_train_enc = preproc.fit_transform(X_train)
    X_test_enc  = preproc.transform(X_test)

    feature_order = num_cols + cat_cols
    df_train_feat = pd.DataFrame(X_train_enc, columns=feature_order, index=X_train.index)
    df_test_feat  = pd.DataFrame(X_test_enc,  columns=feature_order, index=X_test.index)

    # 6) 目标列追加回最后（已是 int）
    df_train = pd.concat(
        [df_train_feat.reset_index(drop=True),
         df_train_raw[target_column_name].reset_index(drop=True)],
        axis=1
    )
    df_test = pd.concat(
        [df_test_feat.reset_index(drop=True),
         df_test_raw[target_column_name].reset_index(drop=True)],
        axis=1
    )

    # 7) 数据集描述
    dataset_description = read_txt_file(os.path.join(base_loc, f"{dataset_name}-description.txt"))

    return df_train, df_test, target_column_name, dataset_description

# 读取其他数据集
def load_origin_data_1(loc):
    # 读取数据集
    with open(loc, 'rb') as f:
        ds = pickle.load(f)

    df_train = ds[1]
    df_test = ds[2]
    target_column_name = ds[4][-1]
    dataset_description = ds[-1]

    return df_train, df_test, target_column_name, dataset_description
# 读取credit数据集
def load_credit_origin_data(loc):
    # 读取数据集
    with open(loc, 'rb') as f:
        ds = pickle.load(f)
    ds, df_train, df_test = get_data_split(ds, seed=0)
    target_column_name = ds[4][-1]
    dataset_description = ds[-1]

    return df_train, df_test, target_column_name, dataset_description
# 功能：生成基础模型 随机森林
def base_model(seed):
    rforest = RandomForestClassifier(n_estimators=100, random_state=seed, class_weight='balanced')
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)  # 可重复的随机数据划分
    param_grid = {
        "min_samples_leaf": [0.001, 0.01, 0.05],  # 调整范围
        "max_depth": [5, 10, None]  # 新增深度控制
    }
    gsmodel = GridSearchCV(rforest, param_grid, cv=cv, scoring='f1')

    return gsmodel
# 功能：执行代码并捕获错误
def code_exec(code):
    try:
        # 尝试编译检查（compile 成 AST 再执行）
        compiled_code = compile(code, "<string>", "exec")
        exec(compiled_code, globals())
        return None
    except Exception as e:
        print("Code could not be executed:", e)
        return str(e)
# 功能：清理 LLM 生成的代码
def clean_llm_code(code: str) -> str:
    import re
    # 去除 ``` 开头的代码块标记和末尾附加内容
    code = re.sub(r"^```python\s*", "", code.strip(), flags=re.IGNORECASE)
    code = re.sub(r"```$", "", code.strip())

    # 清除 <end> 和非代码文字（可能来自 LLM）
    code = re.sub(r"<end>", "", code)

    # 移除 LLM 输出中的解释段或文本开头错误提示
    lines = code.strip().splitlines()
    cleaned_lines = []
    for line in lines:
        if line.strip().startswith("class myclassifier") or line.strip().startswith("import") or line.strip().startswith(
                "from"):
            cleaned_lines.append(line)
        elif cleaned_lines:  # 如果已开始记录代码块，继续添加后续代码
            cleaned_lines.append(line)
    return "\n".join(cleaned_lines)
# # 将数据集分成训练集和验证集
# def train_test_split(df, test_size=0.20, random_state=None):
#     """    
#     将数据集分成训练集和验证集
#     :param df: 数据集
#     :param test_size: 验证集占比
#     :param random_state: 随机种子
#     :return: 训练集和验证集
#     """
#     if random_state is not None:
#         np.random.seed(random_state)
    
#     # 打乱数据集
#     shuffled_indices = np.random.permutation(len(df))
#     test_set_size = int(len(df) * test_size)
    
#     test_indices = shuffled_indices[:test_set_size]
#     train_indices = shuffled_indices[test_set_size:]
    
#     df_train = df.iloc[train_indices].reset_index(drop=True)
#     df_val = df.iloc[test_indices].reset_index(drop=True)
    
#     return df_train, df_val
# 读取txt文件的全部内容
def read_txt_file(file_path):
    """
    读取txt文件的全部内容，保留格式，返回字符串。
    
    参数：
        file_path (str): txt 文件的完整路径。
    
    返回：
        str: 文件内容（保留换行、空格等格式），读取失败时返回 None。
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        return content
    except FileNotFoundError:
        print(f"[错误] 文件未找到: {file_path}")
    except UnicodeDecodeError:
        print(f"[错误] 文件编码无法识别（尝试使用 UTF-8 编码失败）: {file_path}")
    except PermissionError:
        print(f"[错误] 没有权限读取该文件: {file_path}")
    except Exception as e:
        print(f"[错误] 读取文件时发生未知错误: {e}")
    
    return None
# 生成 single-shot
def generate_single_shot(dataset_description_prompt,
                         model_code_prompt,
                         model_performance_differences_prompt,
                         best_weights,
                         best_auc,
                         output_txt_path="single_shot_prompt.txt"):
    """
    构造 single-shot 提示词并写入到 txt 文件。

    :param dataset_description_prompt: 数据集描述部分（字符串）
    :param model_code_prompt: 模型代码与性能部分（字符串）
    :param model_performance_differences_prompt: 模型差异矩阵部分（字符串）
    :param best_weights: 最优权重列表或元组
    :param best_auc: 最优 AUC 值
    :param output_txt_path: 输出 txt 文件路径
    :return: 写入的完整 single-shot prompt 字符串
    """
    # 格式化权重向量为输出字符串
    weight_lines = []
    for i, w in enumerate(best_weights):
        weight_lines.append(f"模型{i+1}：权重 = {w:.4f}")
    weight_text = "\n".join(weight_lines)

    # 构造 single-shot 内容
    shot_prompt = f"""以下是一个二分类任务的示例，展示如何根据多个模型的表现与互补性信息分配合理的集成权重。

    任务描述：
    你将看到若干分类模型及其在验证集上的表现，同时给出模型之间的预测差异矩阵。你的目标是为这些模型分配合理的集成权重（非负，且总和为1），以在验证集上获得更好的泛化性能。

    {dataset_description_prompt.strip()}
    {model_code_prompt.strip()}
    {model_performance_differences_prompt.strip()}

    【最佳权重向量建议】
    {weight_text}

    最优验证集 AUC：{best_auc:.4f}

    加权理由示范：
    - 模型1具有最高的 AUC，因此给予最高权重；
    - 模型2性能略低，但稳定性较好；
    - 模型3虽然性能偏弱，但在预测上与模型1和2互补性强，因此保留一定权重；
    - 整体策略兼顾了个体性能与模型间差异，提升集成模型的鲁棒性。
    """

    # 写入文件
    with open(output_txt_path, 'w', encoding='utf-8') as f:
        f.write(shot_prompt)

    print(f"✅ single-shot 提示词已写入到：{output_txt_path}")
    return shot_prompt

# def get_dataset_description(ds_name, dataset_description):
#     """
#     获取数据集描述的提示词
#     :param ds_name: 数据集名称
#     :param dataset_description: 数据集描述
#     :return: 数据集描述的提示词
#     """
#     loc = "/home/usr01/cuicui/autoML-ensemble/data/" + ds_name + ".json"
#     # 读取数据集特征描述json文件全部内容
#     if os.path.exists(loc):
#         with open(loc, 'r', encoding='utf-8') as f:
#             feature_description = f.read()
#     else:
#         print(f"数据集描述文件 {loc} 不存在，使用默认描述。")

#     return f"""
#     数据集名称：{ds_name}
#     数据集描述：{dataset_description}
#     数据集特征描述：{feature_description}
#     """

# 功能：获取模型代码与模型表现提示词
def get_model_code_prompt(model_code_list=None,
                          val_acc_list=None,
                          val_auc_list=None,
                          val_f1_list=None,
                          val_pre_list=None,
                          val_rec_list=None,
                          use_code=False):
    """
    获取模型代码与模型表现提示词
    :param model_code_list: 模型代码列表
    :param val_acc_list: 验证集准确率列表
    """
    model_code_prompt = "Model list:\n"
    for i, code in enumerate(model_code_list):
        model_code_prompt += f"Model {i + 1}:\n"
        if use_code:
            model_code_prompt += f"code: {code}\n"
        # 收集该模型的所有有效指标
        metrics = []
        if val_acc_list is not None:
            metrics.append(f"Accuracy={val_acc_list[i]:.2f},")
        if val_auc_list is not None:
            metrics.append(f"AUC={val_auc_list[i]:.2f},")
        if val_pre_list is not None:
            metrics.append(f"Precision={val_pre_list[i]:.2f},")
        if val_rec_list is not None:
            metrics.append(f"Recall={val_rec_list[i]:.2f},")
        if val_f1_list is not None:
            metrics.append(f"F1={val_f1_list[i]:.2f}")
        
        # 仅当有指标时才添加表现信息
        if metrics:
            model_code_prompt += f"- Validation set performance: {' '.join(metrics)}\n"
    
    return model_code_prompt

# 功能：获取模型表现差异矩阵,模型在独立验证集上的预测差异矩阵
def get_model_pre_differences(best_fitted_model_instance_list,val_x,val_y):
    """
    计算模型之间的预测差异程度(Disagreement Rate 矩阵)
    :param model_list: 模型实例列表（每个模型应具有 .predict 方法）
    :param val_x: 验证集特征
    :return: N x N 对称矩阵，元素[i][j] 表示模型i与模型j预测不一致的样本比例（Disagreement Rate）
    值越高 → 模型互补性越强；
    """
    n = len(best_fitted_model_instance_list)
    preds = [model.predict(val_x) for model in best_fitted_model_instance_list]

    matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(i + 1, n):
            diff = preds[i] != preds[j]
            disagree_rate = np.mean(diff)
            matrix[i][j] = disagree_rate
            matrix[j][i] = disagree_rate  # 对称

    return matrix

# 得到模型的生成错误指示矩阵（向量）E
def _error_matrix(best_fitted_model_instance_list: List, val_x, val_y) -> np.ndarray:
    """
    生成错误指示矩阵 E，形状为 (n_models, n_samples)。
    E[j, i] = 1 表示第 j 个模型在第 i 个样本上预测错误；否则为 0。
    适用于二分类或多分类（只看“是否正确”）。
    """
    y_true = np.asarray(val_y).ravel()
    n_models = len(best_fitted_model_instance_list)
    n_samples = y_true.shape[0]
    E = np.zeros((n_models, n_samples), dtype=np.uint8)

    for j, model in enumerate(best_fitted_model_instance_list):
        y_pred = np.asarray(model.predict(val_x)).ravel()
        if y_pred.shape[0] != n_samples:
            raise ValueError(f"Model {j} predict length {y_pred.shape[0]} != {n_samples}")
        E[j] = (y_pred != y_true).astype(np.uint8)
    return E

# 功能：获取模型双错误预测差异矩阵
def get_model_double_error_differences(best_fitted_model_instance_list: List, val_x, val_y) -> np.ndarray:
    """
    经典“double-fault”矩阵（不要做 1−JointErr 互补）：
    返回 N×N 对称矩阵 DF，其中 DF[i, j] = 同时出错的样本比例 = |{k: e_i(k)=1 且 e_j(k)=1}| / |D_valid|。
    - 取值范围 [0, 1]；越小表示两模型“共同犯错”越少（多样性越高）。
    - 对角线 DF[i, i] = 模型 i 的单体错误率。
    """
    E = _error_matrix(best_fitted_model_instance_list, val_x, val_y).astype(np.float64)
    n_samples = E.shape[1]
    # 交集计数：E @ E^T（把 True/1 当作 1 计数）
    inter_counts = E @ E.T  # 形状 (N, N)
    DF = inter_counts / float(n_samples)
    # 数值稳定：限制到 [0,1]
    np.clip(DF, 0.0, 1.0, out=DF)
    return DF

# 功能：获取模型 Jaccard-Fault 矩阵 交集/并集
def get_model_jaccard_fault(best_fitted_model_instance_list: List, val_x, val_y) -> np.ndarray:
    """
    增强版方案 3：Jaccard-Fault 矩阵。
    定义每个模型的“错误集合” E_i = {k | 模型 i 在样本 k 上出错}。
    返回 N×N 对称矩阵 J，其中 J[i, j] = |E_i ∩ E_j| / |E_i ∪ E_j|。
      - 若两者从不出错且 |E_i ∪ E_j|=0，则定义 J[i, j] = 0（避免 NaN）。
      - 取值范围 [0, 1]；数值越小，说明两模型的错误重叠越少（互补性越强）。
      - 对角线 J[i, i] = 1（如果该模型有至少一个错误）；若完全无错，定义为 0。
    """
    E = _error_matrix(best_fitted_model_instance_list, val_x, val_y).astype(np.float64)
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
        if flag:
            J[i, i] = 1.0
        else:
            J[i, i] = 0.0

    # 数值稳定：限制到 [0,1]
    np.clip(J, 0.0, 1.0, out=J)
    return J

# 功能：获取模型表现差异提示词
def get_model_performance_differences_prompt(model_difference_matrix):
    """
    获取模型表现差异提示词
    :param disagree_matrix: N x N 对称矩阵（numpy array 或嵌套 list）
    :return: 字符串，描述性提示词
    """
    n = len(model_difference_matrix)

    # 格式化矩阵为字符串
    matrix_str = "[\n"
    for row in model_difference_matrix:
        formatted_row = ", ".join(f"{val:.4f}" for val in row)
        matrix_str += f" [{formatted_row}],\n"
    matrix_str = matrix_str.rstrip(",\n") + "\n]"

    prompt = (
    "The pairwise prediction differences on an independent validation set are shown in the matrix below. "
    "Each element [i][j] denotes the prediction disagreement rate between model i and model j on the validation set; "
    "a larger value indicates greater divergence in predictions and thus stronger complementarity.\n"
    "The model order in the matrix follows the list above (Model 1, Model 2, ...), and the matrix is 1-indexed to match the model numbering.\n"
    f"The disagreement matrix is as follows (unit: proportion, rounded to two decimal places):\n{matrix_str}\n"
    )

    return prompt

# 双错误预测差异矩阵提示词
def get_model_double_error_prompt(model_double_error_matrix):
    """
    获取模型双错误预测差异矩阵提示词
    :param model_double_error_matrix: N x N 对称矩阵（numpy array 或嵌套 list）
    :return: 字符串，描述性提示词
    """
    n = len(model_double_error_matrix)

    # 格式化矩阵为字符串
    matrix_str = "[\n"
    for row in model_double_error_matrix:
        formatted_row = ", ".join(f"{val:.4f}" for val in row)
        matrix_str += f" [{formatted_row}],\n"
    matrix_str = matrix_str.rstrip(",\n") + "\n]"

    prompt = (
    "The double-fault prediction differences on an independent validation set are shown in the matrix below. "
    "Each element [i][j] denotes the proportion of samples on which model i and model j are simultaneously wrong on the validation set; "
    "a smaller value indicates fewer joint errors between the two models and thus greater diversity.\n"
    "The model order in the matrix follows the list above (Model 1, Model 2, ...), and the matrix is 1-indexed to match the model numbering.\n"
    f"The double-fault difference matrix is as follows (unit: proportion, rounded to two decimal places):\n{matrix_str}\n"
    )

    return prompt

# Jaccard-Fault 矩阵提示词
def get_model_intersection_union_prompt(model_jaccard_fault_matrix):
    """
    获取模型 Jaccard-Fault 矩阵提示词
    :param model_jaccard_fault_matrix: N x N 对称矩阵（numpy array 或嵌套 list）
    :return: 字符串，描述性提示词
    """
    n = len(model_jaccard_fault_matrix)

    # 格式化矩阵为字符串
    matrix_str = "[\n"
    for row in model_jaccard_fault_matrix:
        formatted_row = ", ".join(f"{val:.4f}" for val in row)
        matrix_str += f" [{formatted_row}],\n"
    matrix_str = matrix_str.rstrip(",\n") + "\n]"

    prompt = (
    "The Jaccard-Fault differences on an independent validation set are shown in the matrix below. "
    "Each element [i][j] denotes the Jaccard similarity between the error sets of model i and model j on the validation set; "
    "a smaller value indicates less overlap in their errors and thus stronger complementarity.\n"
    "The model order in the matrix follows the list above (Model 1, Model 2, ...), and the matrix is 1-indexed to match the model numbering.\n"
    f"The Jaccard-Fault matrix is as follows (unit: proportion, rounded to two decimal places):\n{matrix_str}\n"
    )

    return prompt

# 救援-置信矩阵
def get_rescue_confidence_matrix(
    best_fitted_model_instance_list: List,
    val_x,
    val_y,
    task: str,
    beta: float = 5.0,
    high_err_quantile: float = 0.8,
    eps: float = 1e-12,
    label_smoothing: float = 1e-3,
) -> np.ndarray:
    """
    Rescue–Confidence 矩阵 R（非对称）：R[i, j] 表示“i 在 j 犯错时把它救回的能力”。
    你需显式传入 task ∈ {"classification", "regression"}。
    """
    models = best_fitted_model_instance_list
    N = len(models)
    val_y = np.asarray(val_y)

    if task == "classification":
        # ---------- 统一类别空间 ----------
        classes = set(np.unique(val_y))
        for m in models:
            if hasattr(m, "classes_"):
                try:
                    classes.update(list(m.classes_))
                except Exception:
                    pass
        global_classes = np.array(sorted(list(classes)))
        C = len(global_classes)
        cls_index = {c: i for i, c in enumerate(global_classes)}

        # 索引化真标签
        y_true_idx = np.vectorize(lambda c: cls_index[c])(val_y)
        M = len(val_y)

        preds_idx = []
        probas = []

        for m in models:
            y_pred = m.predict(val_x)
            y_pred_idx = np.vectorize(lambda c: cls_index[c])(y_pred)
            preds_idx.append(y_pred_idx)

            # 目标：构造对齐到全局类别的 P: (M, C)
            P = np.zeros((M, C), dtype=np.float64)

            if hasattr(m, "predict_proba"):
                p_local = m.predict_proba(val_x)
                p_local = np.asarray(p_local)

                # 统一成 (M, ?)
                if p_local.ndim == 1:
                    # 一维 → 视作正类概率
                    if hasattr(m, "classes_") and len(getattr(m, "classes_")) == 2:
                        neg_cls, pos_cls = m.classes_[0], m.classes_[1]
                        pos_idx = cls_index[pos_cls]
                        neg_idx = cls_index[neg_cls]
                        p_pos = np.clip(p_local, 0.0, 1.0)
                        P[:, pos_idx] = p_pos
                        P[:, neg_idx] = 1.0 - p_pos
                    else:
                        # 无 classes_ 或多类但只给了一列概率（不规范情况）→ 退化兜底
                        maxp = np.clip(p_local, 0.0, 1.0)
                        P[:] = 0.0
                        P[np.arange(M), y_pred_idx] = maxp
                        remain = 1.0 - maxp
                        if C > 1:
                            P += (remain / (C - 1))[:, None]
                            P[np.arange(M), y_pred_idx] -= remain / (C - 1)
                else:
                    # 二维 (M, C_m)
                    if hasattr(m, "classes_"):
                        # 按模型自身类顺序映射到全局
                        for j_cls, cls in enumerate(m.classes_):
                            P[:, cls_index[cls]] = p_local[:, j_cls]
                    else:
                        # 无 classes_ → 将最大列映射到预测类，其余均匀摊
                        maxp = p_local.max(axis=1)
                        P[np.arange(M), y_pred_idx] = maxp
                        remain = 1.0 - maxp
                        if C > 1:
                            P += (remain / (C - 1))[:, None]
                            P[np.arange(M), y_pred_idx] -= remain / (C - 1)
            else:
                # 无 predict_proba：用 label smoothing 伪概率
                eps_ls = float(label_smoothing)
                P[:] = eps_ls if C > 1 else 0.0
                if C > 1:
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

        # 错样权重
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

    elif task == "regression":
        # ---------- 回归路径 ----------
        preds = np.asarray([m.predict(val_x).astype(np.float64) for m in models])  # (N, M)
        y_true = val_y.astype(np.float64)
        resid = y_true[None, :] - preds                    # (N, M)
        abs_resid = np.abs(resid)

        R = np.zeros((N, N), dtype=np.float64)
        for j in range(N):
            # 高误差样本集合（j 的难错）
            T = np.quantile(abs_resid[j], high_err_quantile)
            hard = (abs_resid[j] > T).astype(np.float64)   # (M,)
            w = abs_resid[j] * hard                        # (M,)
            Zj = w.sum() + eps

            # 改善量：(|r_j|-|r_i|)_+ · |r_j| · 1[hard]
            diff = np.maximum(0.0, abs_resid[j][None, :] - abs_resid)     # (N, M)
            num = (diff * abs_resid[j][None, :] * hard[None, :]).sum(axis=1)  # (N,)
            R[:, j] = num / Zj

        np.fill_diagonal(R, 0.0)
        np.clip(R, 0.0, 1.0, out=R)
        return R

    else:
        raise ValueError("task 必须为 'classification' 或 'regression'")

# 根据救援-置信矩阵生成 LLM 提示词
def get_rescue_confidence_prompt(rescue_confidence_matrix):
    """
    获取救援-置信矩阵的提示词
    :param rescue_confidence_matrix: N x N 非对称矩阵（numpy array 或嵌套 list）
    :return: 字符串，描述性提示词
    """
    n = len(rescue_confidence_matrix)

    # 格式化矩阵为字符串
    matrix_str = "[\n"
    for row in rescue_confidence_matrix:
        formatted_row = ", ".join(f"{val:.4f}" for val in row)
        matrix_str += f" [{formatted_row}],\n"
    matrix_str = matrix_str.rstrip(",\n") + "\n]"

    prompt = (
    "The Rescue-Confidence matrix is shown below. "
    "Each element [i][j] indicates the ability of model i to rescue model j when it makes a mistake; "
    "a larger value means model i is more effective at correcting model j's errors.\n"
    "The model order in the matrix follows the list above (Model 1, Model 2, ...), and the matrix is 1-indexed to match the model numbering.\n"
    f"The Rescue-Confidence matrix is as follows (unit: proportion, rounded to two decimal places):\n{matrix_str}\n"
    )

    return prompt

def generate_llm_weight_prompt(
    base_prompt=None,
    dataset_description_prompt=None,
    model_code_prompt=None,
    model_performance_differences_prompt=None,
    model_double_error_prompt=None,
    model_Jaccard_Fault_prompt=None,
    model_rescue_confidence_prompt=None,
    single_shot_prompt_D=None,
    single_shot_prompt_J=None,
    single_shot_prompt_R=None,
):
    """
    Build the final LLM prompt for assigning ensemble weights. All instructions are in English.
    """
    # Initialize prompt parts with the fixed instructions
    prompt_parts = [
        """
You must output the following three parts:
(1) Output the weight for each model. A model's weight may be 0, and the sum of all weights must be 1.00.
(2) Briefly explain the reason for each model's weight.
(3) On the last line, output a weight list alone, in the format: Final model weight list: [0.13, 0.22, 0.00, 0.33, 0.32]
The last line will be parsed automatically by the program, so strictly follow this format.
"""
    ]

    # Append optional sections if provided
    if model_code_prompt is not None:
        prompt_parts.append(model_code_prompt.strip())

    if model_performance_differences_prompt is not None:
        prompt_parts.append(model_performance_differences_prompt.strip())
        prompt_parts.append(
            f"""
single-shot example:
{{
{single_shot_prompt_D.strip()}
}}
"""
        )

    if model_double_error_prompt is not None:
        prompt_parts.append(model_double_error_prompt.strip())

    if model_Jaccard_Fault_prompt is not None:
        prompt_parts.append(model_Jaccard_Fault_prompt.strip())
        prompt_parts.append(
            f"""
single-shot example:
{{
{single_shot_prompt_J.strip()}
}}
"""
        )

    if model_rescue_confidence_prompt is not None:
        prompt_parts.append(model_rescue_confidence_prompt.strip())
        prompt_parts.append(
            f"""
single-shot example:
{{
{single_shot_prompt_R.strip()}
}}
"""
        )

    # Final formatting constraint reminder
    prompt_parts.append(
        """
IMPORTANT: The last line of your output must exactly match the format of the last line in the single-shot example!
IMPORTANT:If the model weight is 0, you must also output it, and you must ensure that the dimension of the obtained weight vector is consistent with the number of models!
"""
    )
    return "\n\n".join(prompt_parts)

def extract_weight_list(llm_response: str):
    """
    从 LLM 返回的文本中提取权重列表，支持格式如：
    最终的权重列表：[0.00, 0.40, 0.00, 0.30, 0.30]
    或 模型权重列表：[0.20, 0.25, 0.00, 0.30, 0.25]
    以及 **Final model weight list:** [0.40, ...] 等 Markdown 加粗形式
    """
    print("\n🔍 模型权重推理结果原始输出:\n")
    print(llm_response)

    pattern = r"""
    (?ixs)
    (?:\*\*)?\s*
    (?:最终的?权重列表|模型权重列表|权重列表|Final\s*model\s*weight\s*list)\s*
    (?:\*\*)?\s*
    [:：=]?
    \s*(?:\*\*)?\s*
    \[(.*?)\]
    """
    match = re.search(pattern, llm_response, re.IGNORECASE | re.DOTALL | re.VERBOSE)

    if match:
        raw_weights = match.group(1)
        try:
            weights = [float(w.strip()) for w in raw_weights.split(',')]
            print("\n✅ 提取成功，权重向量为：", weights)
            return weights
        except Exception as e:
            print("❌ 权重提取失败，格式可能有误：", str(e))
            return []
    else:
        print("❌ 未在输出中找到权重列表标记")
        return []

# 生成下游模型代码
def generate_model(model, messages):
    # 创建一个 OpenAI 的 API 客户端实例
    client = OpenAI(
        base_url='xxx',
        api_key='xxx',
    )
    # 这一段是调用 OpenAI 的 Chat Completion 接口，让模型根据 messages 对话上下文生成回复
    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        stop=["```end"],
        temperature=0.7,
        max_tokens=700
    )
    # 从模型的返回结果中提取第一条生成的消息内容
    code = completion.choices[0].message.content
    code = code.replace("```python", "").replace("```", "").replace("<end>", "")
    return code

# 调用 openAI 的 ChatGPT 接口
def call_llm_chat_completion(model, messages):
    client = OpenAI(
        base_url='xxx',
        api_key='xxx',
    )

    try:
        completion = client.chat.completions.create(
            model=model,
            messages=messages,
            stop=["```end"],  # 可选：如需控制终止
            temperature=0.4,
            max_tokens=700  # 官方参数名是 max_tokens，不是 max_completion_tokens
        )
        return completion

    except Exception as e:
        print(f"❌ 调用 OpenAI 接口失败: {e}")
        return None
    

def get_model_prompt(ds_description=None):
    prompt = f"""
        Here is what I provide:

        Dataset description:
        {ds_description}

        Based on this information, you must:
        1. Generate a new Python classifier class named `myclassifier` each round.
        2. The model must differ from previous versions in **model type or structure**.
        3. The goal is to **maximize AUC (Area Under the ROC Curve)** on the given test data.

        The class must support the following methods:
            model = myclassifier()                        # Initialize the classifier
            model.fit(train_aug_x, train_aug_y)           # Train the model (both are pandas DataFrames)
            pred = model.predict(test_aug_x)              # Predict class labels (1D array or list)
            proba = model.predict_proba(test_aug_x)       # Predict probabilities of the positive class

        Important instructions (must follow **strictly**):
        - Only output the complete Python class named `myclassifier`. No explanation, no markdown, no extra output.
        - The class must be ready to use in Python (all necessary imports included, no undefined variables).
        - `train_aug_x`, `train_aug_y`, and `test_aug_x` are all pandas DataFrames.
        - `predict` must return a 1D array or list (same length as `test_aug_x`).
        - `predict_proba` must return the **positive class** probability:
              return self.model.predict_proba(test_x)[:, 1]
        - Each new version must differ from the previous ones (by model type or structure).
        - Use diverse models across rounds (e.g., LogisticRegression, RandomForest, GradientBoosting, etc.).
        - You may use libraries such as scikit-learn (>=1.4), XGBoost, CatBoost, etc.
        - **You must NOT use `LightGBM` under any circumstances. It is forbidden due to compatibility issues.**
        - Always prefer models that support probabilistic outputs suitable for AUC optimization.
        Output must be **only** the full Python class named `myclassifier`. No extra explanation or code.
        """
    return prompt


from typing import Tuple, Any
import numpy as np

def train_test_split_new(
    *arrays: Any,
    test_size: float | int = 0.25,
    train_size: float | int | None = None,
    random_state: int | None = None,
    shuffle: bool = True,
) -> Tuple[Any, ...]:
    """
    A minimal reimplementation of sklearn.model_selection.train_test_split.

    Parameters
    ----------
    arrays : one or more array-like, all with the same length (len = n_samples)
    test_size : float in (0,1) or int (number of test samples). Default 0.25
    train_size : float in (0,1) or int, optional. If None, it's n - n_test
    random_state : int seed for reproducibility
    shuffle : whether to shuffle before splitting

    Returns
    -------
    A tuple of length 2*len(arrays): (train_arr1, test_arr1, train_arr2, test_arr2, ...)
    """

    if len(arrays) == 0:
        raise ValueError("At least one array must be provided.")

    n = len(arrays[0])
    for a in arrays[1:]:
        if len(a) != n:
            raise ValueError("All input arrays must have the same length.")

    # Determine sizes
    if isinstance(test_size, float):
        if not (0.0 < test_size < 1.0):
            raise ValueError("When float, test_size must be in (0, 1).")
        n_test = int(np.ceil(n * test_size))
    elif isinstance(test_size, int):
        n_test = test_size
    else:
        raise TypeError("test_size must be float or int.")

    if train_size is None:
        n_train = n - n_test
    else:
        if isinstance(train_size, float):
            if not (0.0 < train_size < 1.0):
                raise ValueError("When float, train_size must be in (0, 1).")
            n_train = int(np.floor(n * train_size))
        elif isinstance(train_size, int):
            n_train = train_size
        else:
            raise TypeError("train_size must be float, int, or None.")

        if n_train + n_test > n:
            raise ValueError("train_size + test_size must be <= n_samples.")

    if n_train <= 0 or n_test <= 0:
        raise ValueError("Both train and test sizes must be >= 1.")

    # Build indices
    indices = np.arange(n)
    if shuffle:
        rng = np.random.default_rng(random_state)
        rng.shuffle(indices)

    # Here we take the first n_train as train, remaining as test
    train_idx = indices[:n_train]
    test_idx = indices[n_train:n_train + n_test]

    # Helper to index arrays while preserving type where possible
    def _index(a, idx):
        # numpy arrays, lists, tuples are supported; for others try __getitem__
        if isinstance(a, np.ndarray):
            return a[idx]
        try:
            return np.asarray(a)[idx]
        except Exception:
            # Fall back to Python-level indexing
            return [a[i] for i in idx]

    out = []
    for a in arrays:
        out.append(_index(a, train_idx))
        out.append(_index(a, test_idx))
    return tuple(out)


def to_pd(df_train, target_name):
    y = df_train[target_name].astype(int)
    x = df_train.drop(target_name, axis=1)
    return x, y


def get_classification_param_prompt_NOconstraint(
        best_code, best_auc, dataset_description,
        X_test, feature_columns, dataset_name=None, max_rows=10
    ):
        """
        生成适用于 LLM 分类模型参数优化的提示词
        """
        import pandas as pd
        if isinstance(X_test, pd.DataFrame):
            df_show = X_test.copy()
        else:
            df_show = pd.DataFrame(X_test, columns=feature_columns)

        table = df_show.head(max_rows).to_string(index=False)
        data_shape = df_show.shape if hasattr(df_show, 'shape') else (len(df_show), len(feature_columns))
        if dataset_name is None:
            dataset_name = "unknown"

        prompt = (
            f"Here is the best classification model code so far, with its current AUC score on the test set:\n\n"
            f"Current best AUC: {best_auc:.4f}\n\n"
            f"Model code:\n"
            f"```python\n{best_code}\n```\n"
            f"The downstream classification task is based on the following dataset.\n"
            f"Dataset name: {dataset_name}\n"
            f"Dataset description:\n{dataset_description}\n\n"
            # f"Feature names:\n{', '.join(feature_columns)}\n\n"
            f"Test set shape: {data_shape}\n"
            f"Here are the first {max_rows} rows of the test set:\n{table}\n\n"
            "Please ONLY optimize the hyperparameters\n"
            "in the given classification model code to further improve the AUC value.\n"
            "DO NOT change the algorithm type or model structure.\n"
            "Output only a new optimized Python code block.\n"
            "No explanation, only code!\n"
            "IMPORTANT CONTEXT:\n"
            "You are writing classification model code in Python using scikit-learn version 1.6.1.\n"
            "STRICT REQUIREMENT:\n"
            "ONLY use parameters that are supported by scikit-learn version 1.6.1.\n"
            "DO NOT use any parameters that are deprecated or only available in versions prior to 1.2.\n"
            "Refer ONLY to the scikit-learn 1.6.1 documentation for valid parameters and their default values."
        )

        return prompt