from openai import OpenAI

# 构建数据样例
"""
可能的输出：
age (float64): Samples [23.12, 25.79, 31.0, 22.2, 29.99, 34.88, 28.1, 30.0, 21.5, 27.2]
gender (object): Samples ['M', 'F', 'F', 'M', 'M', 'F', 'F', 'M', 'M', 'F']
"""


# 生成下游模型代码
def generate_model(model, messages):
    # 创建一个 OpenAI 的 API 客户端实例
    client = OpenAI(
        # api_key=xxx,
        # base_url=xxx  # 如是代理或本地服务，自行修改
        base_url='https://api.openai-proxy.org/v1',
        api_key='sk-M1Xfm04JA1ZhOGda2xvuiCEx33mhYqA4zcDKx8ygtPLSz2Ts',
    )
    # 这一段是调用 OpenAI 的 Chat Completion 接口，让模型根据 messages 对话上下文生成回复
    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        stop=["```end"],
        temperature=0.7,
        max_completion_tokens=700
    )
    # 从模型的返回结果中提取第一条生成的消息内容
    code = completion.choices[0].message.content
    code = code.replace("```python", "").replace("```", "").replace("<end>", "")
    return code

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


# 获取模型构建提示词
"""
函数的作用是：
生成一个用于提示大语言模型（如 ChatGPT）的多轮“自动分类器代码生成”指令模板。
它结合了数据样本（samples）和目标列名（target_column_name），构造出一个要求模型自动输出“Python 分类器类”的 prompt 文本。
返回的是 LLM 生成的分类器的代码 的提示词
二分类任务使用的提示词
"""
def get_model_prompt_skold(
        target_column_name=None, samples=None
):
    prompt = f'''
    Here is what I provide:
    
    All column names and a few sample rows of data (in tabular form):
    {samples}
    
    The name of the target column for classification:
    "{target_column_name}"
    
    Based on this information, you must:
    1.Generate a new Python classifier class named myclassifier each round.
    2.The model should differ from previous versions (in model type or structure).
    3.The goal is to maximize the metrics on the given test data.
    
    The class must support the following methods:
    model = myclassifier()                      # Initialize the classifier
    model.fit(train_aug_x, train_aug_y)                # Train the model (both are pandas DataFrames)
    pred = model.predict(test_aug_x)                # Predict class labels (returns 1D array or list)
    proba = model.predict_proba(test_aug_x)            # Predict class probabilities (for computing AUC)
    
    Important instructions (must follow strictly):
    - Only output the complete Python class named myclassifier (no explanation, no comments, no extra code).
    - The class must be ready to use in Python (no missing imports or undefined variables).
    - train_aug_x, train_aug_y, and test_aug_x are all pandas DataFrames.
    - The predict method must return a 1D array or list of labels (same length as test_x).
    - The predict_proba method must return the probability of the positive class using the format: return self.model.predict_proba(test_x)[:, 1]
    - Each new version must differ from the previous one (by model type).
    - Always use a different model type or structure from all previous versions. Model diversity is critical (e.g., LogisticRegression, RandomForest, Gradient Boosting, SVC with probability, neural nets, pipelines, stacked models, etc.).
    - You may use scikit-learn, XGBoost, LightGBM, CatBoost, or any compatible library.
    - The classifier class must be fully self-contained and ready to run in Python.
    - Strictly output only the complete Python class named myclassifier. No extra explanation or code.
    '''
    return prompt


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

#         - Use diverse models across rounds (e.g., LogisticRegression, RandomForest, GradientBoosting, SVC, MLP, stacking/ensemble, etc.).
# - **You must NOT use `BaggingClassifier` under any circumstances. It is forbidden due to compatibility issues.**

def get_model_prompt_voting(target_column_name=None, samples=None):
    prompt = f"""
        Here is what I provide:

        All column names and a few sample rows of data (in tabular form):
        {samples}

        The name of the target column for classification:
        "{target_column_name}"

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
        - Use diverse models across rounds (e.g., LogisticRegression, RandomForest, GradientBoosting, SVC, MLP, stacking/ensemble, etc.).
        - You may use libraries such as scikit-learn (>=1.4), XGBoost, LightGBM, CatBoost, etc.
        - **You must NOT use `BaggingClassifier` under any circumstances. It is forbidden due to compatibility issues.**
        - Always prefer models that support probabilistic outputs suitable for AUC optimization.

        Output must be **only** the full Python class named `myclassifier`. No extra explanation or code.
        """
    return prompt




def get_model_prompt_multi(target_column_name=None, samples=None):
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




"""
二分类任务模型提示词老版本，不要求模型多样性
"""
def get_model_prompt_old(
        target_column_name=None, samples=None
):
    prompt = f'''
    Here is what I provide:
    
    All column names and a few sample rows of data (in tabular form):
    {samples}
    
    The name of the target column for classification:
    "{target_column_name}"
    
    Based on this information, you must:
    1.Generate a new Python classifier class named myclassifier each round.
    2.The model should differ from previous versions (in model type or structure).
    3.The goal is to maximize the metrics on the given test data.
    
    The class must support the following methods:
    model = myclassifier()                      # Initialize the classifier
    model.fit(train_aug_x, train_aug_y)                # Train the model (both are pandas DataFrames)
    pred = model.predict(test_aug_x)                # Predict class labels (returns 1D array or list)
    proba = model.predict_proba(test_aug_x)            # Predict class probabilities (for computing AUC)
    
    Important instructions (must follow strictly):
    - Only output the complete Python class named myclassifier (no explanation, no comments, no extra code).
    - The class must be ready to use in Python (no missing imports or undefined variables).
    - train_aug_x, train_aug_y, and test_aug_x are all pandas DataFrames.
    - The predict method must return a 1D array or list of labels (same length as test_x).
    - The predict_proba method must return the probability of the positive class using the format: return self.model.predict_proba(test_x)[:, 1]
    - Each new version must differ from the previous one (by model type).
    '''
    return prompt


"""
多分类任务的模型生成提示词，可能需要更改
"""
# TODO 可能需要更改，使生成的模型支持多分类————old 新的是 get_model_prompt_multi
def get_multiclass_model_prompt(
        target_column_name=None, samples=None
):
    prompt = f'''
    Here is what I provide:
    
    All column names and a few sample rows of data (in tabular form):
    {samples}
    
    The name of the target column for classification:
    "{target_column_name}"
    
    Your task is to generate a Python classification model for a **multiclass** task.
    
    Instructions:
    1. You must define a Python class named `myclassifier` for each model version.
    2. Each version must differ from previous ones in model type or structure (e.g., RandomForest, XGBoost, LightGBM, neural networks, stacking, pipelines, etc.).
    3. The goal is to maximize predictive performance (e.g., accuracy, F1-macro, AUC-macro) on the test data.
    4. You must assume that the target variable has more than 2 classes (i.e., this is not a binary classification task).
    
    The class must implement the following interface:
    - `model = myclassifier()`                      # Initialize the classifier
    - `model.fit(train_aug_x, train_aug_y)`         # Train the model (inputs are pandas DataFrames)
    - `pred = model.predict(test_aug_x)`            # Predict class labels (returns 1D list or array)
    - `proba = model.predict_proba(test_aug_x)`     # Return class probabilities (shape: [n_samples, n_classes])
    
    Strict constraints (follow these carefully):
    - Do NOT assume binary classification.
    - The `predict_proba` method must return the full probability matrix for all classes:  
      `return self.model.predict_proba(test_aug_x)`  
      (shape: [n_samples, n_classes])
    - The class must be fully functional, self-contained, and import all required libraries.
    - Only return the complete class definition for `myclassifier`. No extra explanation, no additional code.
    - `train_aug_x`, `train_aug_y`, and `test_aug_x` are all pandas DataFrames.
    - You must use a **different model type or structure each round** to ensure diversity.
    - You may use scikit-learn, LightGBM, XGBoost, CatBoost, or any other compatible Python ML library.
    
    Your final output must be only the complete class definition of `myclassifier`.
    '''
    return prompt



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


def get_regression_model_prompt_new(
    ds_description=None,
    seed=0,
    banlist=None,           # e.g. ["XGBRegressor", "CatBoostRegressor", "StackingRegressor", "RandomForestRegressor"]
    last_family=None        # e.g. "tree_ensemble", "kernel", "linear", "nn", "probabilistic", "instance", "gam", "boosting_sklearn"
):
    banlist = banlist or []
    # 槽位顺序可按需调整；不要出现具体类名，给出“家族标签”即可
    families = [
        "kernel",              # SVR/KernelRidge 等
        "linear",              # Ridge/Lasso/ElasticNet/SGD 等
        "instance",            # KNN/半径近邻等
        "probabilistic",       # GaussianProcess/BayesianRidge/NGBoost(如允许)
        "nn",                  # MLP/简单Tabular NN
        "gam",                 # 广义加性(如 pyGAM / EBM，可退化到 spline-like/GAM 替代)
        "boosting_sklearn",    # sklearn 的 HistGradientBoosting/GradientBoosting（明确禁止 LightGBM）
        "tree_ensemble"        # ExtraTrees/Bagging 树等（非Boosting）
    ]
    slot = seed % len(families)
    family_rule = families[slot]

    # 生成禁用清单文本（避免模型名/库名重复）
    ban_directives = "\n".join([f"- Must NOT contain substring: `{t}`" for t in banlist])

    prompt = f"""
You are generating Python code for a **single** regression class.

Dataset description:
{ds_description}

Your goals:
- Minimize RMSE on the given test set.
- **Strong novelty** each round.

Hard constraints (read carefully):
1) Output **only** a complete Python class named `myregressor`. No explanations, no comments, no extra code outside the class.
2) The class must be ready to run: perform any needed imports **inside the class methods** so there are no missing imports at module level.
3) `fit(train_aug_x, train_aug_y)` and `predict(test_aug_x)` must work with pandas DataFrames; `predict` must return a 1D array (do **not** .tolist()).
4) **LightGBM is forbidden.**
5) **Do not reuse algorithms or libraries from previous rounds.** You must avoid any estimator/library whose name matches the banlist below (substring match):
{ban_directives if banlist else "- (no banlist this round)"}

Diversity rules (must satisfy at least two):
- Change the **algorithm family** compared with `last_family={last_family}` and anything implied by banlist.
- Change the **loss/regularization** or key hyperparameters in a meaningful way.
- Change the **preprocessing pipeline** (e.g., scaling/quantile transform/one-hot vs. target encoding/feature crosses/polynomial features).
- Change the **ensemble topology** (bagging/blending/stacking/voting) or make it **single-learner** if last round was ensemble.
- Add **feature engineering** (e.g., polynomial/spline/binning/quantile) appropriate for tabular regression.

Family slot selection (deterministic via seed):
- This round you must implement a model from the **{family_rule}** family chosen by `seed={seed}`.
- If that conflicts with the banlist or would repeat the previous family, move to the next family in the list (cyclic) until a valid, unused family is selected.

Robustness & compatibility:
- Prefer widely available Python libraries (especially scikit-learn). If you reference an optional third-party package that might be missing, implement a silent **fallback** to a scikit-learn alternative of the **same family** inside the class (via try/except).

Remember:
- Output only the class `myregressor` (no comments).
"""
    return prompt



"""
聚类任务的模型生成提示词
"""
def get_clustering_model_prompt(target_column_name=None, samples=None, n_clusters=3):
    prompt = f'''
        You are to act as an expert in machine learning, especially unsupervised learning and clustering.
        
        Dataset:
        - Samples (tabular form):
        {samples}
        - Number of clusters to discover: {n_clusters}
        
        Task:
        - Generate a Python class named `mycluster` for clustering.
        - Each new version must use a different type of clustering algorithm or pipeline.
        - The objective is to maximize Adjusted Rand Index (ARI).
        
        Requirements for the class:
        - Class name: `mycluster`
        - Constructor must accept `n_clusters` as a parameter (even if not explicitly used).
        - Must implement `fit_predict(X)` method where X is a pandas DataFrame, returning predicted labels.
        - Class must include all required imports.
        - Be ready-to-execute Python code only.
        - Avoid repeating previously used algorithms (e.g. KMeans, Spectral, DBSCAN...); innovate with hybrid, ensemble, or dimensionality-reduction-based models.
        - The generated model should be structurally different from all previous ones.
        - Avoid text, explanations, or comments. Output only the class definition code.
        - **You must NOT use `VotingClassifier` under any circumstances. It is forbidden due to compatibility issues.**
        '''
    return prompt



"""
LLM 自动生成簇数
"""
def get_clustering_model_prompt_v2(samples=None, dataset_summary=None):
    prompt = f'''
You are to act as an expert in machine learning, especially unsupervised learning and clustering.

You will be provided with a dataset preview and statistical summary. Based on this information, estimate a reasonable number of clusters and generate Python code for clustering accordingly.

Dataset Preview (first 10 rows):
{samples}

Dataset Summary:
- Number of samples: {dataset_summary.get("n_samples")}
- Number of features: {dataset_summary.get("n_features")}
- Feature types: {dataset_summary.get("feature_types")}
- Feature ranges: {dataset_summary.get("feature_ranges")}
- Mean and standard deviation (numerical features only): {dataset_summary.get("mean_std")}

Task:
- Analyze the dataset summary to determine the optimal number of clusters automatically (e.g., via internal metrics or distribution analysis).
- Generate a Python class named `mycluster` for clustering.
- Each new version must use a different type of clustering algorithm or pipeline.
- The objective is to maximize Adjusted Rand Index (ARI).

Requirements for the class:
- Class name: `mycluster`
- Do not accept `n_clusters` as a constructor parameter.
- The optimal number of clusters should be hard-coded within the class based on your estimation.
- Must implement `fit_predict(X)` method where X is a pandas DataFrame, returning predicted labels.
- Class must include all required imports.
- Be ready-to-execute Python code only.
- Avoid repeating previously used algorithms (e.g., KMeans, Spectral, DBSCAN...); innovate with hybrid, ensemble, or dimensionality-reduction-based models.
- The generated model should be structurally different from all previous ones.
- Avoid text, explanations, or comments. Output only the class definition code.
- **You must NOT use `VotingClassifier` under any circumstances. It is forbidden due to compatibility issues.**
'''
    return prompt
