import os
import sys
# 添加项目根目录到
project_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__),   # 当前文件所在目录
                 "..", "..")                  # 向上跳两级
)
sys.path.append(project_root)

import re
import time
import copy
# from classification_parameter_optimization_tool import ParameterOptimizationTool

from ensemble_utils import (
    weighted_ensemble_with_llm_weights,
    weighted_voting_with_llm_weights,
    tiered_ensemble_with_llm_weights,
    stacking_ensemble
)
from ensemble_utils2 import stacking_ensemble_v2
from tools import(
    load_origin_data,
    clean_llm_code,
    read_txt_file,
    get_model_code_prompt,
    get_model_pre_differences,
    get_model_performance_differences_prompt,
    generate_llm_weight_prompt,
    extract_weight_list,
    call_llm_chat_completion,
    generate_model,
    get_model_jaccard_fault,
    get_model_intersection_union_prompt,
    get_rescue_confidence_matrix,
    get_rescue_confidence_prompt,
    get_model_prompt,
    train_test_split,
    to_pd,
    get_classification_param_prompt_NOconstraint
)
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
)
from weight_integration_tools import integrate_three_strategies_cls

import numpy as np
import random
import argparse
import warnings
from sklearn.model_selection import StratifiedKFold
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=ResourceWarning)


# 要求最不严格的参数优化，对LLM生成模型的参数无限制
def model_parameter_optimization_with_cv(
    new_class_name, base_model_code, Iterations, train_x, train_y, 
    dataset_description, ds_name, args, i
):
    model_messages = []  # 初始化模型消息列表
    best_cv_auc = 0
    best_cv_f1 = 0
    best_cv_precision = 0
    best_cv_recall = 0
    best_cv_acc = 0
    best_code = base_model_code
    best_model_instance = None
    best_model_fitted = None
    best_train_auc = 0
    best_train_f1 = 0
    best_train_precision = 0
    best_train_recall = 0
    best_train_acc = 0

    try:
        # 初始化5折交叉验证
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.default_seed)
        cv_scores_auc = []
        cv_scores_f1 = []
        cv_scores_precision = []
        cv_scores_recall = []
        cv_scores_acc = []

        # 基础模型交叉验证评估
        model_class = globals()[new_class_name]
        base_model = model_class()

        for train_idx, val_idx in cv.split(train_x, train_y):
            cv_train_x, cv_val_x = train_x.iloc[train_idx], train_x.iloc[val_idx]
            cv_train_y, cv_val_y = train_y.iloc[train_idx], train_y.iloc[val_idx]

            model_copy = copy.deepcopy(base_model)
            model_copy.fit(cv_train_x, cv_train_y)

            if hasattr(model_copy, 'predict_proba'):
                proba = model_copy.predict_proba(cv_val_x)
                
                # 安全处理概率数组维度
                if proba.ndim == 2 and proba.shape[1] > 1:
                    positive_proba = proba[:, 1]
                elif proba.ndim == 2 and proba.shape[1] == 1:
                    positive_proba = proba.flatten()
                elif proba.ndim == 1:
                    positive_proba = proba
                else:
                    raise ValueError(f"Unsupported probability array dimension: {proba.ndim}, shape: {proba.shape}")
                
                preds = (positive_proba > 0.5).astype(int)
                cv_scores_auc.append(roc_auc_score(cv_val_y, positive_proba))
                cv_scores_f1.append(f1_score(cv_val_y, preds))
                cv_scores_precision.append(precision_score(cv_val_y, preds))
                cv_scores_recall.append(recall_score(cv_val_y, preds))
                cv_scores_acc.append(accuracy_score(cv_val_y, preds))

        # 计算基础模型交叉验证指标
        base_cv_auc = np.mean(cv_scores_auc) * 100
        base_cv_f1 = np.mean(cv_scores_f1) * 100
        base_cv_precision = np.mean(cv_scores_precision) * 100
        base_cv_recall = np.mean(cv_scores_recall) * 100
        base_cv_acc = np.mean(cv_scores_acc) * 100
        print(f"基础模型5折交叉验证平均AUC: {base_cv_auc:.4f}")
        print(f"基础模型5折交叉验证平均ACC: {base_cv_acc:.4f}")

        # 初始化最佳值
        best_cv_auc = base_cv_auc
        best_cv_f1 = base_cv_f1
        best_cv_precision = base_cv_precision
        best_cv_recall = base_cv_recall
        best_cv_acc = base_cv_acc
        best_model_instance = copy.deepcopy(base_model)

        # 全量训练最佳模型并计算训练集指标
        best_model_fitted = copy.deepcopy(base_model)
        best_model_fitted.fit(train_x, train_y)
        
        if hasattr(best_model_fitted, 'predict_proba'):
            train_proba = best_model_fitted.predict_proba(train_x)
            if train_proba.ndim == 2 and train_proba.shape[1] > 1:
                train_positive_proba = train_proba[:, 1]
            elif train_proba.ndim == 2 and train_proba.shape[1] == 1:
                train_positive_proba = train_proba.flatten()
            elif train_proba.ndim == 1:
                train_positive_proba = train_proba
            else:
                raise ValueError(f"Training set probability array dimension error: {train_proba.ndim}")
        else:
            train_positive_proba = best_model_fitted.predict(train_x)
        
        train_preds = (train_positive_proba > 0.5).astype(int)
        best_train_auc = roc_auc_score(train_y, train_positive_proba) * 100
        best_train_f1 = f1_score(train_y, train_preds)
        best_train_precision = precision_score(train_y, train_preds)
        best_train_recall = recall_score(train_y, train_preds)
        best_train_acc = accuracy_score(train_y, train_preds)

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
            'optimized_model_auc': 0,
            'optimized_model_f1': 0,
            'optimized_model_precision': 0,
            'optimized_model_recall': 0,
            'optimized_model_acc': 0,
            'optimized_model_instance': None,
            'optimized_model_fitted': None,
            'error_message': str(e),
            'model_messages': model_messages,
            'train_auc': 0,
            'train_f1': 0,
            'train_precision': 0,
            'train_recall': 0,
            'train_acc': 0
        }

    # 参数优化准备：修改提示词，禁止内部交叉验证
    print(f"----------------参数优化开始，优化代码 {new_class_name}------------------")
    param_prompt = get_classification_param_prompt_NOconstraint(
        best_code=best_code,
        best_auc=best_cv_auc,
        dataset_description=dataset_description,
        X_test=train_x,
        feature_columns=train_x.columns.tolist(),
        dataset_name=ds_name,
        max_rows=10
    )
    # 补充提示词：明确禁止内部参数搜索和交叉验证（英文）
    param_prompt += """
    \n\nImportant Constraints (must be strictly followed):
    1. The generated model class must NOT contain any form of cross-validation, parameter search, or optimization logic (such as GridSearchCV, RandomizedSearchCV, cross-validation loops, etc.).
    2. The model must initialize the base model (e.g., RandomForestClassifier) in the __init__ method with specific hyperparameter values (e.g., n_estimators=200). Do not leave hyperparameters empty or use default values.
    3. The fit method must only contain model fitting logic (e.g., self.model.fit(X, y)) and must not add any parameter adjustment or search steps.
    4. Must implement predict and predict_proba methods, where predict_proba returns probabilities for the positive class (1-dimensional array).
    5. The code must be an executable Python class with no explanatory text or comments.
    """

    param_messages = [
        {
            "role": "system",
            "content": "You are a classification optimization assistant.\n"
                    "Your task is to help me improve the 5-fold cross-validation AUC of the given classifier\n"
                    "by tuning hyperparameters only. Your answer must contain only executable Python code with a single class.\n"
                    "Absolutely do NOT include any cross-validation, parameter search (like GridSearchCV) or optimization logic in the class.\n"
                    f"The class name of the model you generate must not be changed; the class name must be: myclassifier_{i+1}"
                    "The class must have __init__, fit, predict, and predict_proba methods with clear hyperparameters in __init__."
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

            param_new_class_name = f"myclassifier_{i+1}_param_{p_iter + 1}"
            optimized_code = optimized_code.replace(f"class myclassifier_{i+1}:", f"class {param_new_class_name}:")
            optimized_code = optimized_code.replace(f"class myclassifier_{i+1}_param_{p_iter}:",f"class {param_new_class_name}:")

            param_err = code_exec(optimized_code)
            if param_err is not None:
                print(f"代码执行错误: {param_err}")
                param_messages.extend([
                    {"role": "assistant", "content": optimized_code},
                    {"role": "user", "content": f"Code execution failed, error: {param_err}. Please fix and regenerate. Note: Absolutely no cross-validation or parameter search logic is allowed. Only keep initialization, fit, predict, and predict_proba methods."}
                ])
                continue

            print('---------------优化后的代码\n' + optimized_code)

            if param_new_class_name not in globals():
                raise NameError(f"Class {param_new_class_name} not defined in optimized code")

            model_class = globals()[param_new_class_name]
            optimized_model = model_class()

            # 交叉验证评估优化模型
            cv_scores_auc = []
            cv_scores_f1 = []
            cv_scores_precision = []
            cv_scores_recall = []
            cv_scores_acc = []
            for train_idx, val_idx in cv.split(train_x, train_y):
                cv_train_x, cv_val_x = train_x.iloc[train_idx], train_x.iloc[val_idx]
                cv_train_y, cv_val_y = train_y.iloc[train_idx], train_y.iloc[val_idx]

                model_copy = copy.deepcopy(optimized_model)
                model_copy.fit(cv_train_x, cv_train_y)

                if hasattr(model_copy, 'predict_proba'):
                    proba = model_copy.predict_proba(cv_val_x)
                    
                    # 安全处理概率数组维度
                    if proba.ndim == 2 and proba.shape[1] > 1:
                        positive_proba = proba[:, 1]
                    elif proba.ndim == 2 and proba.shape[1] == 1:
                        positive_proba = proba.flatten()
                    elif proba.ndim == 1:
                        positive_proba = proba
                    else:
                        raise ValueError(f"Unsupported probability array dimension: {proba.ndim}, shape: {proba.shape}")
                    
                    preds = (positive_proba > 0.5).astype(int)
                    cv_scores_auc.append(roc_auc_score(cv_val_y, positive_proba))
                    cv_scores_f1.append(f1_score(cv_val_y, preds))
                    cv_scores_precision.append(precision_score(cv_val_y, preds))
                    cv_scores_recall.append(recall_score(cv_val_y, preds))
                    cv_scores_acc.append(accuracy_score(cv_val_y, preds))

            cv_auc = np.mean(cv_scores_auc) * 100
            cv_f1 = np.mean(cv_scores_f1) * 100
            cv_precision = np.mean(cv_scores_precision) * 100
            cv_recall = np.mean(cv_scores_recall) * 100
            cv_acc = np.mean(cv_scores_acc) * 100
            print(f"优化模型5折交叉验证平均AUC: {cv_auc:.4f}")
            print(f"优化模型5折交叉验证平均ACC: {cv_acc:.4f}")

            if cv_auc > best_cv_auc:
                print(f"参数优化效果提升：{best_cv_auc:.4f} --> {cv_auc:.4f}")
                best_cv_auc = cv_auc
                best_cv_f1 = cv_f1
                best_cv_precision = cv_precision
                best_cv_recall = cv_recall
                best_cv_acc = cv_acc
                best_code = optimized_code
                best_model_instance = copy.deepcopy(optimized_model)

                # 全量训练更新
                best_model_fitted = copy.deepcopy(optimized_model)
                best_model_fitted.fit(train_x, train_y)
                
                if hasattr(best_model_fitted, 'predict_proba'):
                    train_proba = best_model_fitted.predict_proba(train_x)
                    if train_proba.ndim == 2 and train_proba.shape[1] > 1:
                        train_positive_proba = train_proba[:, 1]
                    elif train_proba.ndim == 2 and train_proba.shape[1] == 1:
                        train_positive_proba = train_proba.flatten()
                    elif train_proba.ndim == 1:
                        train_positive_proba = train_proba
                    else:
                        raise ValueError(f"Training set probability array dimension error: {train_proba.ndim}")
                else:
                    train_positive_proba = best_model_fitted.predict(train_x)
                
                train_preds = (train_positive_proba > 0.5).astype(int)
                best_train_auc = roc_auc_score(train_y, train_positive_proba) * 100
                best_train_f1 = f1_score(train_y, train_preds)
                best_train_precision = precision_score(train_y, train_preds)
                best_train_recall = recall_score(train_y, train_preds)
                best_train_acc = accuracy_score(train_y, train_preds)

            param_messages.extend([
                {"role": "assistant", "content": optimized_code},
                {"role": "user",
                "content": f"Current CV results - AUC: {cv_auc:.4f}, F1: {cv_f1:.4f}, Precision: {cv_precision:.4f}, Recall: {cv_recall:.4f}, ACC: {cv_acc:.4f}\n"
                            f"Best CV results - AUC: {best_cv_auc:.4f}, F1: {best_cv_f1:.4f}, Precision: {best_cv_precision:.4f}, Recall: {best_cv_recall:.4f}, ACC: {best_cv_acc:.4f}\n"
                            f"Please continue optimizing hyperparameters to improve AUC. Note: The generated model class must completely remove all cross-validation and parameter search logic, retaining only initialization, fitting, and prediction methods."
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
                            "2. Whether the class definition is correct and includes __init__, fit, predict, and predict_proba methods\n"
                            "3. Whether hyperparameters are within valid ranges"
                }
            ])
            continue

    print(f"参数优化完成,最佳5折交叉验证AUC: {best_cv_auc:.4f}")

    return {
        'optimized_model_code': best_code,
        'optimized_model_auc': best_cv_auc,
        'optimized_model_f1': best_cv_f1,
        'optimized_model_precision': best_cv_precision,
        'optimized_model_recall': best_cv_recall,
        'optimized_model_acc': best_cv_acc,
        'optimized_model_instance': best_model_instance,
        'optimized_model_fitted': best_model_fitted,
        'param_messages': param_messages,
        'train_auc': best_train_auc,
        'train_f1': best_train_f1,
        'train_precision': best_train_precision,
        'train_recall': best_train_recall,
        'train_acc': best_train_acc
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


if __name__ == '__main__':
    # 解析命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--gpus', default="0", type=str, help='GPU设置')
    parser.add_argument('-s', '--default_seed', default=42, type=int, help='随机种子')
    parser.add_argument('-l', '--llm', default='gpt-3.5-turbo', type=str, help='大模型')
    # parser.add_argument('-l', '--llm', default='gpt-4o', type=str, help='大模型')
    parser.add_argument('-e', '--exam_iterations', default=3, type=int, help='实验次数')
    # parser.add_argument('-f', '--feat_iterations', default=5, type=int, help='特征迭代次数')
    parser.add_argument('-m', '--model_iterations', default=10, type=int, help='模型迭代次数')
    parser.add_argument('-p', '--param_iterations', default=5, type=int,help='参数调优次数')
    parser.add_argument('-ds', '--dataset_name', default='ds_credit', type=str, help='数据集名称')
    args = parser.parse_args()

    # cd1 cc1 ld1  cc2 cd2 cf1 balance-scale ds_credit
    # adult bank blood heart pc1 tic-tac-toe pc3 kc1

    # 用于存储每次实验集成学习的指标的结果

    balanced_acc_list_stacking = []
    # 
    acc_list_stacking_v2 = []
    f1_list_stacking_v2 = []
    auc_list_stacking_v2 = []
    pre_list_stacking_v2 = []
    rec_list_stacking_v2 = []


    # 实验
    for ex in range(args.exam_iterations):
        print(f"================ Experiment {ex + 1} / {args.exam_iterations} ================")
        ds_name= args.dataset_name
        print(f"=========== Dataset {ds_name} ===========")
        # loc = "/home/usr01/cuicui/CAAFE++/tests/data/" + ds_name + ".pkl"

        model_code = '' # 存储 LLM 生成的模型代码,用于LLM进行权重生成

        # 实验次数
        seed = args.default_seed
        # 设置随机种子
        random.seed(seed)
        np.random.seed(seed)
        # 加载数据集
        """数据集划分成三个部分，df_train、df_val、df_test，分别是训练集、独立的验证集、测试集，比例是3:1:1"""
        df_train, df_test, target_column_name, dataset_description = load_origin_data(ds_name,seed=ex)  # 加载数据 一般二分类模型通用
        # 将训练集分成训练集和独立验证集，独立验证集占比 25 %
        stacking_x, stacking_y = to_pd(df_train, target_column_name)  # 获取特征矩阵和标签向量
        df_train, df_val = train_test_split(df_train, test_size=0.25, random_state=seed)
        train_x, train_y = to_pd(df_train, target_column_name)
        val_x, val_y = to_pd(df_val, target_column_name)
        test_x, test_y = to_pd(df_test, target_column_name)

        # 构造提示词需要的数据格式
        # s = build_prompt_samples(df_train)

        # LLM 生成 分类模型代码的提示词 prompt
        model_prompt = get_model_prompt(ds_description=dataset_description)

        # LLM 生成 分类器模型 提示词
        model_messages = [
            {
                "role": "system",
                "content": (
                    "You are a top-level machine learning classification expert.\n"
                    "Your task is to help me iteratively search for the most suitable classifier model.\n"
                    "Your primary goal is to maximize the AUC (Area Under the ROC Curve) on the test set.\n"
                    "You must focus on improving AUC more than any other metric.\n"
                    "Your answer should only generate valid Python code."
                ),
            },
            {
                "role": "user",
                "content": model_prompt,  # 保持生成模型的具体提示结构不变
            },
        ]


        model_iter = args.model_iterations
        best_auc = 0
        best_code = None
        i = 0

        # 每一轮的最优模型代码列表，这个代码列表用于集成学习
        best_model_code_list = []
        # 模型实例列表，没有 fit 的模型
        best_model_instance_list = []
        # 模型实例列表，已经 fit 的模型
        best_fitted_model_instance_list = []
        # 模型 AUC 列表
        best_model_auc_list = []
        # 模型 Accuracy 列表
        best_model_acc_list = []
        best_model_pre_list = []
        best_model_rec_list = []
        best_model_f1_list = []

        # 模型生成迭代
        while i < model_iter:
            try:
                # 生成下游模型代码
                code = generate_model(args.llm, model_messages)
                # todo 加 code_clean 代码
                code = clean_llm_code(code)

                # 动态修改类名
                new_class_name = f"myclassifier_{i + 1}"
                code = re.sub(r'class\s+myclassifier\w*\s*:', f'class {new_class_name}:', code)
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
                            The classifier code execution failed with error: {type(e)} {e}.\n Code: ```python{code}```
                            Remember, your answer should only generate code.
                            Do not include explanations or comments outside the code block.
                            Generate next code block(fixing error?):
                            """,
                    },
                ]
                continue

            # 调用 model_parameter_optimization 函数进行模型参数优化 
            # model_parameter_optimization_with_trajectory
            # model_optimization_with_param_constraints  参数约束，抑制过拟合
            # model_parameter_optimization_with_cv  参数规模不受限，测试集指标低。但是stacking_ensemble可以改善
            optimization_results = model_parameter_optimization_with_cv(
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
            optimized_model_auc = optimization_results['optimized_model_auc']
            optimized_model_acc = optimization_results['optimized_model_acc']
            optimized_model_f1 = optimization_results['optimized_model_f1']
            optimized_model_pre = optimization_results['optimized_model_precision']
            optimized_model_rec = optimization_results['optimized_model_recall']
            optimized_model_instance = optimization_results['optimized_model_instance']
            optimized_model_fitted = optimization_results['optimized_model_fitted']

            # 打印当前实验详细结果
            print(f"当前实验结果第 {i+1}/{args.model_iterations}")
            print(f"Test  AUC: {optimized_model_auc:.4f}")
            # while 循环继续
            i = i + 1

            if optimized_model_auc > best_auc:
                best_auc = optimized_model_auc
                best_code = optimized_model_code

            # 下一轮模型生成提示词拼接
            if len(code) > 10:
                model_messages += [
                    {"role": "assistant", "content": optimized_model_code},
                    {
                        "role": "user",
                        "content": f"""
                        ✅ The classifier code executed successfully.

                        📈 Current model AUC: {optimized_model_auc:.4f}
                        🏆 Best historical AUC so far: {best_auc:.4f}
                        The code corresponding to the optimal AUC so far: {best_code}

                        Please now propose a new classifier that is **more likely to improve the AUC** on the given test data.
                        The model must differ from all previous ones **by model type or internal structure**.

                        ⚠️ Remember:
                        - You must only output valid Python code for a complete classifier named `myclassifier`.
                        - The class must include all imports and implement: `fit`, `predict`, and `predict_proba`.
                        - Do not repeat models you've already used.
                        - Prioritize models that provide reliable probabilistic outputs to help improve AUC.

                        🎯 Next code block:
                        """,
                    },
                ]

            if optimized_model_fitted is not None and hasattr(optimized_model_fitted, 'predict'):
                best_model_code_list.append(optimized_model_code)
                best_model_instance_list.append(optimized_model_instance)
                best_fitted_model_instance_list.append(optimized_model_fitted)
                best_model_auc_list.append(optimized_model_auc)
                best_model_acc_list.append(optimized_model_acc)
                best_model_pre_list.append(optimized_model_pre)
                best_model_rec_list.append(optimized_model_rec)
                best_model_f1_list.append(optimized_model_f1)


        # 构造集成学习提示词，让 LLM 输出集成学习基础模型的权重向量
        get_model_weight_prompt = ""
        base_prompt = (
            "You are a machine learning expert. Your task is to ensemble multiple already-trained "
            "classification models to improve generalization on the validation set. "
            "Based on the information provided, decide which models should be included in the ensemble "
            "and assign non-negative weights that sum to 1. "
            "Use multiple metrics (e.g., Accuracy, AUC) to evaluate each model's stability, "
            "complementarity, and overall performance.\n\n"
            "Return: (1) a weight vector in the form [w1, w2, ...] aligned with the model order given; "
            "(2) a concise rationale for the weights. Always respond in English."
        )
        # 数据集描述
        # dataset_description_prompt = get_dataset_description(ds_name,dataset_description)
        # 先用之前的数据集描述
        dataset_description_prompt = dataset_description
        # 模型代码以及模型指标信息
        model_code_prompt = get_model_code_prompt(model_code_list=best_model_code_list,
                                                val_acc_list=best_model_acc_list,
                                                val_auc_list=best_model_auc_list,
                                                #   best_model_f1_list,
                                                #   best_model_pre_list,
                                                #   best_model_rec_list
                                                use_code=False
                                                )

        # 模型在独立验证集表现差异矩阵
        model_difference_matrix = get_model_pre_differences(best_fitted_model_instance_list,val_x,val_y)
        # 结合模型在验证集表现差异矩阵获取模型表现差异提示词
        model_performance_differences_prompt = get_model_performance_differences_prompt(model_difference_matrix)

        # 双错误差异矩阵 Jaccard-Fault 矩阵
        model_intersection_union_matrix = get_model_jaccard_fault(best_fitted_model_instance_list,val_x,val_y)
        # 双错误 交集/并集差异矩阵对应提示词
        model_Jaccard_Fault_prompt = get_model_intersection_union_prompt(model_intersection_union_matrix)

        # 救援-置信矩阵
        model_confidence_matrix = get_rescue_confidence_matrix(best_fitted_model_instance_list,val_x,val_y,'classification')
        # 救援-置信矩阵对应提示词
        model_rescue_confidence_prompt = get_rescue_confidence_prompt(model_confidence_matrix)
        
        # 读取文件中的 single-shot 
        loc_path_D = f"{project_root}/classification_ensemble/single-shot/single_shot_prompt_D.txt"
        loc_path_J = f"{project_root}/classification_ensemble/single-shot/single_shot_prompt_J.txt"
        loc_path_R = f"{project_root}/classification_ensemble/single-shot/single_shot_prompt_R.txt"
        single_shot_prompt_D = read_txt_file(loc_path_D)
        single_shot_prompt_J= read_txt_file(loc_path_J)
        single_shot_prompt_R= read_txt_file(loc_path_R)

        # 组装提示词
        get_model_weight_prompt_D = generate_llm_weight_prompt(
            base_prompt=base_prompt,
            model_code_prompt=model_code_prompt,
            model_performance_differences_prompt=model_performance_differences_prompt,
            single_shot_prompt_D=single_shot_prompt_D,
        )
        write_code_to_file(get_model_weight_prompt_D, f"{project_root}/output/classification/{ds_name}/prompt_D.txt")

        get_model_weight_prompt_J = generate_llm_weight_prompt(
            base_prompt=base_prompt,
            model_code_prompt=model_code_prompt,
            model_Jaccard_Fault_prompt=model_Jaccard_Fault_prompt,
            single_shot_prompt_J=single_shot_prompt_J,
        )
        write_code_to_file(get_model_weight_prompt_J, f"{project_root}/output/classification/{ds_name}/prompt_J.txt")

        get_model_weight_prompt_R = generate_llm_weight_prompt(
            base_prompt=base_prompt,
            model_code_prompt=model_code_prompt,
            model_rescue_confidence_prompt=model_rescue_confidence_prompt,
            single_shot_prompt_R=single_shot_prompt_R
        )
        write_code_to_file(get_model_weight_prompt_R, f"{project_root}/output/classification/{ds_name}/prompt_R.txt")


        weight_vector_list = []  # 三种策略的权重列表矩阵，3xN大小
        # 调用LLM，得到LLM的输出
        for prompt_version, get_model_weight_prompt in zip(
            ['D', 'J', 'R'],
            [get_model_weight_prompt_D, get_model_weight_prompt_J, get_model_weight_prompt_R]
        ):
            print(f"\n================= Prompt Version {prompt_version} =================")
            message = [
                {"role": "system", "content": base_prompt},
                {"role": "user", "content": get_model_weight_prompt}
            ]
            llm_output = call_llm_chat_completion('gpt-4o', message)
            # 从LLM的输出中解析出来权重 n维向量列表
            llm_text = llm_output.choices[0].message.content
            weight_vector_list.append(extract_weight_list(llm_text))

        """-------集成部分-------"""
        w_final, info = integrate_three_strategies_cls(
            weight_vector_list, best_fitted_model_instance_list, val_x, val_y,
            method="hedge-geo", tau=10.0, shrink=0.1
        )
        print(f"\n=========== 三种方式得到的权重 ===========")
        for prompt_version, weight_vector in zip(['D', 'J', 'R'], weight_vector_list):
            print(f"{prompt_version}--权重向量: {weight_vector}\n")
        print(f"整合后的权重：{w_final}")

        # 保存指标，最后一起打印
        integrated_metrics = ''
        # 计算整合后的权重在测试集上的表现
        result_v2 = stacking_ensemble_v2(best_model_instance_list,
                            X_train=stacking_x,
                            y_train=stacking_y,
                            X_test=test_x,
                            y_test=test_y,
                            weight_list=w_final,
                        )
        stacking_metrics = result_v2['stacking_metrics']
        integrated_metrics += f"=========== Stacking 集成结果 ===========\n"
        integrated_metrics += f"Accuracy : {stacking_metrics['accuracy']*100:.3f}\n"
        integrated_metrics += f"Balance_Accuracy : {stacking_metrics['balanced_accuracy']*100:.3f}\n"
        integrated_metrics += f"F1 Score : {stacking_metrics['f1']*100:.3f}\n"
        integrated_metrics += f"Precision: {stacking_metrics['precision']*100:.3f}\n"
        integrated_metrics += f"Recall   : {stacking_metrics['recall']*100:.3f}\n"
        integrated_metrics += f"AUC      : {stacking_metrics['roc_auc']*100:.3f}\n\n"
        # 保存结果到对应列表中
        acc_list_stacking_v2.append(stacking_metrics['accuracy']*100)
        f1_list_stacking_v2.append(stacking_metrics['f1']*100)
        pre_list_stacking_v2.append(stacking_metrics['precision']*100)
        rec_list_stacking_v2.append(stacking_metrics['recall']*100)
        auc_list_stacking_v2.append(stacking_metrics['roc_auc']*100)
        print(f"第{ex + 1}次实验的 stacking 结果")
        print(integrated_metrics)

    # 计算集成结果的统计指标
    # 保存结果到文件中
    final_results = f"================= {ds_name} 集成结果统计指标 =================\n"
    final_results += "================= Stacking_v2 集成结果 =================\n"
    final_results += f"Accuracy : {np.mean(acc_list_stacking_v2):.2f} ± {np.std(acc_list_stacking_v2):.2f}\n"
    final_results += f"F1 Score : {np.mean(f1_list_stacking_v2):.2f} ± {np.std(f1_list_stacking_v2):.2f}\n"
    final_results += f"Precision: {np.mean(pre_list_stacking_v2):.2f} ± {np.std(pre_list_stacking_v2):.2f}\n"
    final_results += f"Recall   : {np.mean(rec_list_stacking_v2):.2f} ± {np.std(rec_list_stacking_v2):.2f}\n"
    final_results += f"AUC      : {np.mean(auc_list_stacking_v2):.2f} ± {np.std(auc_list_stacking_v2):.2f}\n"
    loc_path_results = f"{project_root}/classification_ensemble/res/{ds_name}.txt"
    print(final_results)
    write_code_to_file(final_results, loc_path_results)