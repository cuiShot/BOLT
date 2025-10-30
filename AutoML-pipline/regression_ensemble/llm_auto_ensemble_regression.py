import os
import sys
project_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__),   # 当前文件所在目录
                 "..", "..")                  # 向上跳两级
)
sys.path.append(project_root)
import re
import copy

from regression_ensemble_utils import (
    soft_voting_with_weights_regression,
    stacking_ensemble_regression,
)
from regression_tools import (
    get_regression_param_prompt,
    get_model_jaccard_fault,
    get_model_pre_differences,
    get_model_intersection_union_prompt,
    get_model_performance_differences_prompt,
    get_rescue_confidence_prompt,
    generate_llm_weight_prompt,
    get_model_code_prompt,
    get_rescue_confidence_matrix,
    get_regression_model_prompt
)
from classification_ensemble.tools import(
    train_test_split,
    clean_llm_code,
    read_txt_file,
    call_llm_chat_completion,
    extract_weight_list,
    load_new_dataset,
    generate_model,
    # train_test_split_new
)
from sklearn.metrics import mean_squared_log_error,mean_absolute_error, mean_squared_error
from weight_integration_tools_re import integrate_three_strategies_reg

import numpy as np
import random
import pickle
import argparse
import warnings
from sklearn.model_selection import KFold
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=ResourceWarning)

# 回归模型参数优化函数
def model_optimization_with_param_constraints_regression(
        new_class_name, base_model_code, Iterations, train_x, train_y, 
        dataset_description, ds_name, args, i
    ):
    model_messages = []  # 初始化模型消息列表
    best_cv_mae = float('inf')
    best_cv_rmse = float('inf')
    best_cv_rmlse = float('inf')
    best_code = base_model_code
    best_model_instance = None
    best_model_fitted = None
    best_train_mae = float('inf')
    best_train_rmse = float('inf')
    best_train_rmlse = float('inf')

    try:
        # 初始化5折交叉验证（回归使用KFold）
        cv = KFold(n_splits=5, shuffle=True, random_state=args.default_seed)
        cv_scores_mae = []
        cv_scores_rmse = []
        cv_scores_rmlse = []

        # 基础模型交叉验证评估
        model_class = globals()[new_class_name]
        base_model = model_class()

        for train_idx, val_idx in cv.split(train_x):
            cv_train_x, cv_val_x = train_x.iloc[train_idx], train_x.iloc[val_idx]
            cv_train_y, cv_val_y = train_y.iloc[train_idx], train_y.iloc[val_idx]

            model_copy = copy.deepcopy(base_model)
            model_copy.fit(cv_train_x, cv_train_y)

            # 回归模型只有predict方法，没有predict_proba
            preds = model_copy.predict(cv_val_x)
            
            # 计算回归指标
            mae = mean_absolute_error(cv_val_y, preds)
            rmse = np.sqrt(mean_squared_error(cv_val_y, preds))
            
            # 处理RMLSE计算，确保预测值非负
            if np.any(preds < 0) or np.any(cv_val_y < 0):
                # 如果有负值，无法计算RMLSE，使用NaN表示
                rmlse = np.nan
            else:
                rmlse = np.sqrt(mean_squared_log_error(cv_val_y, preds))
            
            cv_scores_mae.append(mae)
            cv_scores_rmse.append(rmse)
            cv_scores_rmlse.append(rmlse)

        # 计算基础模型交叉验证指标（过滤NaN值）
        base_cv_mae = np.mean([x for x in cv_scores_mae if not np.isnan(x)])
        base_cv_rmse = np.mean([x for x in cv_scores_rmse if not np.isnan(x)])
        base_cv_rmlse = np.mean([x for x in cv_scores_rmlse if not np.isnan(x)])
        
        # print(f"基础模型5折交叉验证平均MAE: {base_cv_mae:.4f}")
        print(f"基础模型5折交叉验证平均RMSE: {base_cv_rmse:.4f}")
        # print(f"基础模型5折交叉验证平均RMLSE: {base_cv_rmlse:.4f}")

        # 初始化最佳值
        best_cv_mae = base_cv_mae
        best_cv_rmse = base_cv_rmse
        best_cv_rmlse = base_cv_rmlse
        best_model_instance = copy.deepcopy(base_model)

        # 全量训练最佳模型并计算训练集指标
        best_model_fitted = copy.deepcopy(base_model)
        best_model_fitted.fit(train_x, train_y)
        
        train_preds = best_model_fitted.predict(train_x)
        best_train_mae = mean_absolute_error(train_y, train_preds)
        best_train_rmse = np.sqrt(mean_squared_error(train_y, train_preds))
        
        # 处理训练集RMLSE
        if np.any(train_preds < 0) or np.any(train_y < 0):
            best_train_rmlse = np.nan
        else:
            best_train_rmlse = np.sqrt(mean_squared_log_error(train_y, train_preds))

    except Exception as e:
        print(f"模型代码执行失败: {str(e)}")
        model_messages.append(
            {"role": "assistant", "content": base_model_code}
        )
        model_messages.append(
            {
                "role": "user",
                "content": f"Code execution failed, error: {type(e)} {e}.\n Code: ```python{base_model_code}```\n Please generate the fixed code block. Note: The model class must NOT contain any cross-validation, parameter search, or grid search logic (such as GridSearchCV, RandomizedSearchCV). Only initialize the model with specific hyperparameters in __init__ and call fit directly in the fit method."
            }
        )
        return {
            'optimized_model_code': base_model_code,
            'optimized_model_MAE': float('inf'),
            'optimized_model_RMSE': float('inf'),
            'optimized_model_RMLSE': float('inf'),
            'optimized_model_instance': None,
            'optimized_model_fitted': None,
            'error_message': str(e),
            'model_messages': model_messages,
            'train_MAE': float('inf'),
            'train_RMSE': float('inf'),
            'train_RMLSE': float('inf')
        }

    # 参数优化准备：修改提示词，适应回归任务
    print(f"----------------参数优化开始，优化代码 {new_class_name}------------------")
    param_prompt = get_regression_param_prompt(
        best_code=best_code,
        best_rmse=best_cv_rmse,
        dataset_description=dataset_description,
        X_test=train_x,
        feature_columns=train_x.columns.tolist(),
        dataset_name=ds_name,
        max_rows=10
    )
    # 补充提示词：明确参数范围约束和多方向探索要求（英文）
    param_prompt += """
    \n\nImportant Constraints (must be strictly followed):
    1. The generated model class must NOT contain any form of cross-validation, parameter search, or optimization logic (such as GridSearchCV, RandomizedSearchCV, cross-validation loops, etc.).
    2. Prioritize multi-dimensional hyperparameter exploration over simply increasing model complexity:
    - Adjust regularization parameters
    - Tune model structure parameters
    - Modify sampling strategies where applicable
    - Optimize learning-related parameters in conjunction with model complexity
    3. The fit method must only contain model fitting logic (e.g., self.model.fit(X, y)) without any parameter adjustment.
    4. Must implement predict method that returns regression predictions.
    5. The code must be an executable Python class with no explanatory text or comments.
    """

    param_messages = [
        {
            "role": "system",
            "content": "You are a regression optimization assistant.\n"
                    "Your task is to help me improve the 5-fold cross-validation RMSE of the given regressor\n"
                    "by tuning hyperparameters only, while maintaining reasonable training efficiency.\n"
                    "Key principles: Explore effective hyperparameter combinations that balance performance and efficiency.\n"
                    "Instead, focus on multi-dimensional hyperparameter adjustments. Your answer must contain only executable Python code with a single class.\n"
                    "Absolutely do NOT include any cross-validation, parameter search (like GridSearchCV) or optimization logic in the class.\n"
                    f"The class name of the model you generate must not be changed; the class name must be: myregressor_{i+1}"
                    "The class must have __init__, fit, and predict methods with clear hyperparameters in __init__."
        },
        {
            "role": "user",
            "content": param_prompt
        },
    ]
    

    # 开始参数优化迭代
    for p_iter in range(Iterations):
        print(f"++++++ 第 {p_iter + 1} 次优化 +++++++")
        try:
            optimized_code = generate_model(args.llm, param_messages)
            optimized_code = clean_llm_code(optimized_code)

            param_new_class_name = f"myregressor_{i+1}_param_{p_iter + 1}"
            optimized_code = optimized_code.replace(f"class myregressor_{i+1}:", f"class {param_new_class_name}:")
            optimized_code = optimized_code.replace(f"class myregressor_{i+1}_param_{p_iter}:",f"class {param_new_class_name}:")

            param_err = code_exec(optimized_code)
            if param_err is not None:
                print(f"代码执行错误: {param_err}")
                param_messages.extend([
                    {"role": "assistant", "content": optimized_code},
                    {"role": "user", "content": f"Code execution failed, error: {param_err}. Please fix and regenerate. Note: Absolutely no cross-validation or parameter search logic is allowed. Also ensure hyperparameters (especially n_estimators) are within reasonable ranges."}
                ])
                continue

            print('---------------优化后的代码\n' + optimized_code)

            if param_new_class_name not in globals():
                raise NameError(f"Class {param_new_class_name} not defined in optimized code")

            model_class = globals()[param_new_class_name]
            optimized_model = model_class()

            # 交叉验证评估优化模型
            cv_scores_mae = []
            cv_scores_rmse = []
            cv_scores_rmlse = []
            for train_idx, val_idx in cv.split(train_x):
                cv_train_x, cv_val_x = train_x.iloc[train_idx], train_x.iloc[val_idx]
                cv_train_y, cv_val_y = train_y.iloc[train_idx], train_y.iloc[val_idx]

                model_copy = copy.deepcopy(optimized_model)
                model_copy.fit(cv_train_x, cv_train_y)

                # 回归模型预测
                preds = model_copy.predict(cv_val_x)
                
                # 计算回归指标
                mae = mean_absolute_error(cv_val_y, preds)
                rmse = np.sqrt(mean_squared_error(cv_val_y, preds))
                
                # 处理RMLSE计算
                if np.any(preds < 0) or np.any(cv_val_y < 0):
                    rmlse = np.nan
                else:
                    rmlse = np.sqrt(mean_squared_log_error(cv_val_y, preds))
                
                cv_scores_mae.append(mae)
                cv_scores_rmse.append(rmse)
                cv_scores_rmlse.append(rmlse)

            # 计算平均指标（过滤NaN值）
            cv_mae = np.mean([x for x in cv_scores_mae if not np.isnan(x)])
            cv_rmse = np.mean([x for x in cv_scores_rmse if not np.isnan(x)])
            cv_rmlse = np.mean([x for x in cv_scores_rmlse if not np.isnan(x)])
            
            # print(f"优化模型5折交叉验证平均MAE: {cv_mae:.4f}")
            print(f"优化模型5折交叉验证平均RMSE: {cv_rmse:.4f}")
            # print(f"优化模型5折交叉验证平均RMLSE: {cv_rmlse:.4f}")

            # 回归任务中，RMSE越小越好
            if cv_rmse < best_cv_rmse:
                print(f"参数优化效果提升：{best_cv_rmse:.4f} --> {cv_rmse:.4f}")
                best_cv_mae = cv_mae
                best_cv_rmse = cv_rmse
                best_cv_rmlse = cv_rmlse
                best_code = optimized_code
                best_model_instance = copy.deepcopy(optimized_model)

                # 全量训练更新
                best_model_fitted = copy.deepcopy(optimized_model)
                best_model_fitted.fit(train_x, train_y)
                
                train_preds = best_model_fitted.predict(train_x)
                best_train_mae = mean_absolute_error(train_y, train_preds)
                best_train_rmse = np.sqrt(mean_squared_error(train_y, train_preds))
                
                # 处理训练集RMLSE
                if np.any(train_preds < 0) or np.any(train_y < 0):
                    best_train_rmlse = np.nan
                else:
                    best_train_rmlse = np.sqrt(mean_squared_log_error(train_y, train_preds))

            param_messages.extend([
                {"role": "assistant", "content": optimized_code},
                {"role": "user",
                "content": f"Current CV results - MAE: {cv_mae:.4f}, RMSE: {cv_rmse:.4f}, RMLSE: {cv_rmlse:.4f}\n"
                            f"Best CV results - MAE: {best_cv_mae:.4f}, RMSE: {best_cv_rmse:.4f}, RMLSE: {best_cv_rmlse:.4f}\n"
                            f"Please continue optimizing hyperparameters to reduce RMSE. Focus on multi-dimensional adjustments instead of increasing n_estimators excessively. Keep n_estimators ≤ 300 for base models and ≤ 100 for ensemble wrappers."
                }
            ])

        except Exception as e:
            print(f"Parameter tuning failed: {str(e)}")
            current_code = optimized_code if 'optimized_code' in locals() and optimized_code else "Failed to generate code"
            error_details = f"Error type: {type(e).__name__}, Details: {str(e)}"
            param_messages.extend([
                {"role": "assistant", "content": current_code},
                {"role": "user",
                "content": f"Parameter tuning failed, {error_details}\n"
                            "Please analyze the error cause and fix the code. Focus on checking:\n"
                            "1. Whether cross-validation or parameter search logic is included (must be completely removed)\n"
                            "2. Whether hyperparameters (especially n_estimators) are within reasonable ranges (≤300 for base models, ≤100 for ensembles)\n"
                            "3. Whether the class definition is correct and includes required methods"
                }
            ])
            continue

    print(f"参数优化完成,最佳5折交叉验证RMSE: {best_cv_rmse:.4f}")

    return {
        'optimized_model_code': best_code,
        'optimized_model_MAE': best_cv_mae,
        'optimized_model_RMSE': best_cv_rmse,
        'optimized_model_RMLSE': best_cv_rmlse,
        'optimized_model_instance': best_model_instance,
        'optimized_model_fitted': best_model_fitted,
        'param_messages': param_messages,
        'train_MAE': best_train_mae,
        'train_RMSE': best_train_rmse,
        'train_RMLSE': best_train_rmlse
    }



# 编译检查模型类代码
def code_exec(code):
    try:
        # 尝试编译检查（compile 成 AST 再执行）
        compiled_code = compile(code, "<string>", "exec")
        exec(compiled_code, globals())
        return None
    except Exception as e:
        print("Code could not be executed:", e)
        return str(e)
# 把字符串写入文件
def write_code_to_file(code:str, loc_path:str):
    os.makedirs(os.path.dirname(loc_path), exist_ok=True)
    with open(loc_path, 'w', encoding='utf-8') as f:
        f.write(code)

def to_pd(df_train, target_name):
    y = df_train[target_name].astype(int)
    x = df_train.drop(target_name, axis=1)
    return x, y

# concrete boston winequality california insurance
def load_origin_data(dataset_name, seed):
    # 需要走 .pkl 的旧数据集关键词（子串匹配）
    old_keys = ('concrete','boston','winequality','california','insurance')
    name_l = dataset_name.lower()
    is_old = any(k in name_l for k in old_keys)
    if is_old:
        # 读取数据集
        loc = f"/home/usr01/cuicui/CAAFE++/tests/data/{dataset_name}.pkl"
        with open(loc, 'rb') as f:
            ds = pickle.load(f)
        target_column_name = ds[4][-1]
        df = ds[1]
        dataset_description = ds[-1]
        df_train, df_test = train_test_split(df, test_size=0.20, random_state=seed)

        return df_train, df_test, target_column_name, dataset_description
    # 否则读取新的 CSV 数据集
    base_loc = "/home/usr01/cuicui/autoML-ensemble/new_dataSet/regression/"
    return load_new_dataset(dataset_name, base_loc=base_loc, seed=seed, test_size=0.2)

if __name__ == '__main__':
    # 解析命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--gpus', default="0", type=str, help='GPU设置')
    parser.add_argument('-s', '--default_seed', default=42, type=int, help='随机种子')
    parser.add_argument('-l', '--llm', default='gpt-3.5-turbo', type=str, help='大模型')
    # parser.add_argument('-l', '--llm', default='gpt-4o', type=str, help='大模型')
    parser.add_argument('-e', '--exam_iterations', default=2, type=int, help='实验次数')
    parser.add_argument('-m', '--model_iterations', default=5, type=int, help='模型迭代次数')
    parser.add_argument('-p', '--param_iterations', default=1, type=int,help='参数调优次数')
    parser.add_argument('-ds', '--dataset_name', default='forest-fires', type=str, help='数据集名称')
    args = parser.parse_args()
    # bike crab forest-fires wine wind puma32H california


    # 用于存储每次实验集成学习的指标的结果
    mae_list_stacking = []
    rmse_list_stacking = []
    rmsle_list_stacking = []


    # concrete boston winequality california insurance
    # 
    ds_name= args.dataset_name
    print(f"=========== Dataset {ds_name} ===========")
    for j in range(args.exam_iterations):
        print(f"=========== Experiment {j + 1}/{args.exam_iterations} ===========")
        # loc = "/home/usr01/cuicui/CAAFE++/tests/data/" + ds_name + ".pkl"
        model_code = '' # 存储 LLM 生成的模型代码,用于LLM进行权重生成
        seed = args.default_seed
        i = 0
        # 设置随机种子
        random.seed(seed)
        np.random.seed(seed)
        # 加载数据集
        """数据集划分成三个部分,df_train、df_val、df_test,分别是训练集、独立的验证集、测试集,比例是3:1:1"""
        df_train, df_test, target_column_name, dataset_description = load_origin_data(ds_name,seed)  # 加载数据集

        stacking_x, stacking_y = to_pd(df_train, target_column_name)  # 获取特征矩阵和标签向量
        df_train, df_val = train_test_split(df_train, test_size=0.25, random_state=seed)
        train_x, train_y = to_pd(df_train, target_column_name)
        val_x, val_y = to_pd(df_val, target_column_name)
        test_x, test_y = to_pd(df_test, target_column_name)

        model_prompt = get_regression_model_prompt(ds_description=dataset_description)

        model_messages = [
            {
                "role": "system",
                "content": (
                    "You are a top-level regression algorithm expert\n."
                    " Your task is to help me iteratively search for the most suitable regression model that performs best on a regression task based on the RMSE (Root Mean Squared Error) metric\n."
                    " Your answer should only generate code."
                ),
            },
            {
                "role": "user",
                "content": model_prompt,
            },
        ]

        model_iter = args.model_iterations
        lowest_RMSE = 0
        best_code = None
        

        # 每一轮的最优模型代码列表，这个代码列表用于集成学习
        best_model_code_list = []
        # 模型实例列表，没有 fit 的模型
        best_model_instance_list = []
        # 模型实例列表，已经 fit 的模型
        best_fitted_model_instance_list = []
        # 模型指标列表
        best_model_RMSE_list = []
        best_model_MAE_list = []
        best_model_RMSLE_list = []

        # 模型生成迭代
        while i < model_iter:
            try:
                # 生成下游模型代码
                code = generate_model(args.llm, model_messages)
                # todo 加 code_clean 代码
                code = clean_llm_code(code)

                # 动态修改类名
                new_class_name = f"myregressor_{i + 1}"
                code = re.sub(r'class\s+myregressor\w*\s*:', f'class {new_class_name}:', code)
                print(f"----------------------------原始代码-----------------------")
                print(code)
            except Exception as e:
                print("Error in LLM API." + str(e))
                continue

            e = code_exec(code)
            # 检查编译错误
            if e is not None:  # 生成的代码执行出错 将错误信息反馈给LLM以生成修复后的代码
                model_messages += [
                    {"role": "assistant", "content": code},
                    {
                        "role": "user",
                        "content": f"""
                            The myregressor code execution failed with error: {type(e)} {e}.\n Code: ```python{code}```
                            Remember, your answer should only generate code.
                            Do not include explanations or comments outside the code block.
                            Generate next code block(fixing error?):
                            """,
                    },
                ]
                continue

            # model_parameter_optimization_with_trajectory
            # model_optimization_with_param_constraints_regression  有效抑制过拟合
            # model_parameter_optimization_with_cv  参数规模不受限，测试集指标低。但是stacking_ensemble可以改善
            # 回归任务把参数约束的逻辑已经去掉了
            optimization_results = model_optimization_with_param_constraints_regression(
                new_class_name=new_class_name,
                base_model_code=code,
                Iterations=args.param_iterations,
                train_x=train_x,
                train_y=train_y,
                dataset_description=dataset_description,
                ds_name=ds_name,
                args=args,
                i=i
            )
            # 获取优化后的模型代码和指标 模型在验证集上的指标
            optimized_model_code = optimization_results['optimized_model_code']
            optimized_model_MAE = optimization_results['optimized_model_MAE']
            optimized_model_RMSE = optimization_results['optimized_model_RMSE']
            optimized_model_RMLSE = optimization_results['optimized_model_RMLSE']
            optimized_model_instance = optimization_results['optimized_model_instance']
            optimized_model_fitted = optimization_results['optimized_model_fitted']
            if optimized_model_fitted is not None:
                best_model_code_list.append(optimized_model_code)
                best_model_instance_list.append(optimized_model_instance)
                best_fitted_model_instance_list.append(optimized_model_fitted)
                best_model_RMSE_list.append(optimized_model_RMSE)
                best_model_MAE_list.append(optimized_model_MAE)
                best_model_RMSLE_list.append(optimized_model_RMLSE)           

            # 打印当前实验详细结果
            print(f"当前实验结果第 {i+1}/{args.model_iterations}")
            print(f"Test RMSE: {optimized_model_RMSE:.4f}")
            # while 循环继续
            i = i + 1

            if optimized_model_RMSE < lowest_RMSE:
                lowest_RMSE = optimized_model_RMSE
                best_code = optimized_model_code

            # 下一轮模型生成提示词拼接
            if len(code) > 10:
                model_messages += [
                    {"role": "assistant", "content": optimized_model_code},
                    {
                        "role": "user",
                        "content": f"""
                        ✅ The classifier code executed successfully.

                        📈 Current model RMSE: {optimized_model_RMSE:.4f}
                        🏆 Lowest historical RMSE so far: {lowest_RMSE:.4f}
                        The code corresponding to the Lowest RMSE so far: {best_code}

                        Please now propose a new regressor that is **more likely to reduce the RMSE value** on the given test data.
                        The model must differ from all previous ones **by model type or internal structure**.

                        ⚠️ Remember:
                        - You must only output valid Python code for a complete regressor named `myregressor`.
                        - Do not repeat models you've already used.

                        🎯 Next code block:
                        """,
                    },
                ]
        print(f"=========== 模型生成迭代完成 ===========")

        # 构造集成学习提示词，让 LLM 输出集成学习基础模型的权重向量
        get_model_weight_prompt = ""
        base_prompt = (
            "You are a machine learning expert. Your task is to ensemble multiple already-trained "
            "regression models to improve generalization on the validation set. "
            "Based on the information provided, decide which models should be included in the ensemble "
            "and assign non-negative weights that sum to 1. "
            "Use multiple metrics (e.g., RMSE [primary], MAE, R^2, Spearman/Pearson) to evaluate each model's "
            "stability, complementarity, and overall performance. When available, also consider diversity signals "
            "such as the error-correlation matrix, the high-error-overlap (Jaccard-Fault) matrix, and the "
            "Rescue–Confidence matrix to prefer models that fix each other's high-error cases.\n\n"
            "Principles:\n"
            "(1) Optimize RMSE on the validation set as the primary objective; break ties by MAE and then R^2.\n"
            "(2) Prefer complementary models (low error correlation, low high-error overlap, high rescue ability), "
            "and downweight clearly unstable or dominated models.\n"
            "(3) Encourage sparsity when differences are small; avoid assigning weight to obviously redundant models.\n"
            "(4) Do not invent any numbers—only use the metrics/predictions provided.\n\n"
            "Return: "
            "(1) a weight vector in the form [w1, w2, ...] aligned with the model order given; "
            "(2) a concise rationale for the weights. Always respond in English.\n"
            "Output format:\n"
            "weights: [0.1234, 0.0000, 0.4567, ...]  # non-negative, 4 decimals, sum ≈ 1\n"
            "reason: <3-6 sentences explaining the choice, referencing metrics and diversity signals>"
        )

        # 数据集描述
        # dataset_description_prompt = get_dataset_description(ds_name,dataset_description)
        # 先用之前的数据集描述
        dataset_description_prompt = dataset_description
        # 模型代码以及模型指标信息
        model_code_prompt = get_model_code_prompt(model_code_list=best_model_code_list,
                                            val_MAE_list=best_model_MAE_list,
                                            val_RMSE_list=best_model_RMSE_list,
                                            val_RMSLE_list=best_model_RMSLE_list,
                                            use_code=False
                                            )
        print(model_code_prompt)
        # 模型在独立验证集表现差异矩阵
        model_difference_matrix = get_model_pre_differences(best_fitted_model_instance_list,val_x,val_y)
        # 结合模型在验证集表现差异矩阵获取模型表现差异提示词
        model_performance_differences_prompt = get_model_performance_differences_prompt(model_difference_matrix)

        # 双错误差异矩阵 Jaccard-Fault 矩阵
        model_intersection_union_matrix = get_model_jaccard_fault(best_fitted_model_instance_list,val_x,val_y)
        # 双错误 交集/并集差异矩阵对应提示词
        model_Jaccard_Fault_prompt = get_model_intersection_union_prompt(model_intersection_union_matrix)

        # 救援-置信矩阵
        model_confidence_matrix = get_rescue_confidence_matrix(best_fitted_model_instance_list,val_x,val_y)
        # 救援-置信矩阵对应提示词
        model_rescue_confidence_prompt = get_rescue_confidence_prompt(model_confidence_matrix)
        
        # 读取文件中的 single-shot 
        loc_path_D = f"{project_root}/regression_ensemble/single-shot/single_shot_prompt_D.txt"
        loc_path_J = f"{project_root}/regression_ensemble/single-shot/single_shot_prompt_J.txt"
        loc_path_R = f"{project_root}/new_ensemble/regression_ensemble/single-shot/single_shot_prompt_R.txt"
        single_shot_prompt_D = read_txt_file(loc_path_D)
        single_shot_prompt_J = read_txt_file(loc_path_J)
        single_shot_prompt_R = read_txt_file(loc_path_R)

        # 组装提示词
        get_model_weight_prompt_D = generate_llm_weight_prompt(
            base_prompt=base_prompt,
            model_code_prompt=model_code_prompt,
            model_performance_differences_prompt=model_performance_differences_prompt,
            single_shot_prompt_D=single_shot_prompt_D,
        )
        write_code_to_file(get_model_weight_prompt_D, f"{project_root}/output/regression/{ds_name}/prompt_D.txt")

        get_model_weight_prompt_J = generate_llm_weight_prompt(
            base_prompt=base_prompt,
            model_code_prompt=model_code_prompt,
            model_Jaccard_Fault_prompt=model_Jaccard_Fault_prompt,
            single_shot_prompt_J=single_shot_prompt_J,
        )
        write_code_to_file(get_model_weight_prompt_J, f"{project_root}/output/regression/{ds_name}/prompt_J.txt")

        get_model_weight_prompt_R = generate_llm_weight_prompt(
            base_prompt=base_prompt,
            model_code_prompt=model_code_prompt,
            model_rescue_confidence_prompt=model_rescue_confidence_prompt,
            single_shot_prompt_R=single_shot_prompt_R
        )
        write_code_to_file(get_model_weight_prompt_R, f"{project_root}/output/regression/{ds_name}/prompt_R.txt")

        weight_vector_list = []  # 三种策略的权重列表矩阵
        for prompt_version, get_model_weight_prompt in zip(
            ['D', 'J', 'R'],
            [get_model_weight_prompt_D, get_model_weight_prompt_J, get_model_weight_prompt_R]
        ):
            print(f"\n================= Prompt Version {prompt_version} =================")
            # print(get_model_weight_prompt)
            message = [
                {"role": "system", "content": base_prompt},
                {"role": "user", "content": get_model_weight_prompt}
            ]
            llm_output = call_llm_chat_completion('gpt-4o', message)
            # 从LLM的输出中解析出来权重 n维向量列表
            llm_text = llm_output.choices[0].message.content
            weight_vector_list.append(extract_weight_list(llm_text))

        """---------------------------------------------------- 集成部分 -----------------------------------------------------"""

        # 权重整合，三种方法进行权重整合
        weight_vector_integrated_list = []
        w_final, info = integrate_three_strategies_reg(
            weight_vector_list, best_fitted_model_instance_list, val_x, val_y,
            method="hedge-geo", tau=10.0, shrink=0.1
        )
        print(f"hedge-geo 最终得到的权重向量: {w_final}")

        result = stacking_ensemble_regression(best_model_instance_list,
                            X_train=stacking_x,
                            y_train=stacking_y,
                            X_test=test_x,
                            y_test=test_y,
                            weight_list=w_final
                        )
        stacking_metrics = result['stacking_metrics']
        # 保存到对应列表
        mae_list_stacking.append(stacking_metrics['mae'])
        rmse_list_stacking.append(stacking_metrics['rmse'])
        rmsle_list_stacking.append(stacking_metrics['rmsle'])

    # 保存最终结果到文件
    # results_path = f"{project_root}/regression_ensemble/res/{ds_name}_final_results.txt"
    # with open(results_path, 'w') as f:
    #     f.write(f"================= {ds_name} 集成结果统计指标 =================\n")
    #     f.write(f"=========== Stacking Ensemble ===========\n")
    #     f.write(f"MAE   : {np.mean(mae_list_stacking):.2f} ± {np.std(mae_list_stacking):.2f}\n")
    #     f.write(f"RMSE  : {np.mean(rmse_list_stacking):.2f} ± {np.std(rmse_list_stacking):.2f}\n")
    #     f.write(f"RMSLE : {np.mean(rmsle_list_stacking):.4f} ± {np.std(rmsle_list_stacking):.4f}\n")
    