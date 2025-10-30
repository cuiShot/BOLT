import os
import sys
project_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__),   
                 "..")                  
)
sys.path.append(project_root)

import re
import time
import copy
import argparse
import os
import random
import numpy as np
import pickle
from multiclass_ensemble_utils import (
    stacking_ensemble
)

from classification_ensemble.tools import(
    clean_llm_code,
    train_test_split,
    read_txt_file,
    get_model_code_prompt,
    get_model_performance_differences_prompt,
    generate_llm_weight_prompt,
    extract_weight_list,
    call_llm_chat_completion,
    generate_model,
    get_model_intersection_union_prompt,
    get_rescue_confidence_prompt,
    load_new_dataset,
    to_pd,
    get_classification_param_prompt_NOconstraint
)
from multi_classification_tools import (
    get_model_pre_differences_multiclass,
    get_model_jaccard_fault_multiclass,
    get_model_prompt_multiclass,
    get_model_prompt_multiclass_new,
    get_rescue_confidence_matrix_multiclass
)
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
)
from multiclass_weight_integration_tools import integrate_three_strategies_cls
import warnings
from sklearn.model_selection import StratifiedKFold
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=ResourceWarning)

# ================= ③ 交叉验证优化（多分类版；保持原思路） ================= #
def model_parameter_optimization_with_cv(
    new_class_name, base_model_code, Iterations, train_x, train_y, 
    dataset_description, ds_name, args, i
):
    model_messages = []
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

    avg_strategy = getattr(args, "multiclass_average", "macro")

    def _get_classes_from_model(m, y_fit):
        if hasattr(m, "model") and hasattr(m.model, "classes_"):
            return np.array(m.model.classes_)
        elif hasattr(m, "classes_"):
            return np.array(m.classes_)
        else:
            return np.unique(y_fit)

    def _metrics_multiclass(y_true, proba, classes, avg="macro"):
        classes = np.array(classes)
        if not (proba.ndim == 2 and proba.shape[1] >= 2):
            raise ValueError(f"predict_proba must return 2D (n_samples, n_classes), got {proba.shape}")
        pred_idx = np.argmax(proba, axis=1)
        preds = classes[pred_idx]
        try:
            auc = roc_auc_score(y_true, proba, multi_class="ovr", average=avg)
        except Exception:
            auc = np.nan
        f1 = f1_score(y_true, preds, average=avg, zero_division=0)
        prec = precision_score(y_true, preds, average=avg, zero_division=0)
        rec = recall_score(y_true, preds, average=avg, zero_division=0)
        acc = accuracy_score(y_true, preds)
        return auc, f1, prec, rec, acc

    try:
        # 5折交叉验证
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.default_seed)
        cv_scores_auc, cv_scores_f1, cv_scores_precision, cv_scores_recall, cv_scores_acc = ([] for _ in range(5))

        # 基础模型
        model_class = globals()[new_class_name]
        base_model = model_class()

        for train_idx, val_idx in cv.split(train_x, train_y):
            cv_train_x, cv_val_x = train_x.iloc[train_idx], train_x.iloc[val_idx]
            cv_train_y, cv_val_y = train_y.iloc[train_idx], train_y.iloc[val_idx]

            model_copy = copy.deepcopy(base_model)
            model_copy.fit(cv_train_x, cv_train_y)

            proba = model_copy.predict_proba(cv_val_x)
            classes_fold = _get_classes_from_model(model_copy, cv_train_y)
            auc, f1, prec, rec, acc = _metrics_multiclass(cv_val_y, proba, classes_fold, avg_strategy)

            cv_scores_auc.append(auc)
            cv_scores_f1.append(f1)
            cv_scores_precision.append(prec)
            cv_scores_recall.append(rec)
            cv_scores_acc.append(acc)

        base_cv_auc = float(np.nanmean(cv_scores_auc)) * 100
        base_cv_f1 = float(np.mean(cv_scores_f1)) * 100
        base_cv_precision = float(np.mean(cv_scores_precision)) * 100
        base_cv_recall = float(np.mean(cv_scores_recall)) * 100
        base_cv_acc = float(np.mean(cv_scores_acc)) * 100
        print(f"基础模型5折交叉验证平均AUC(macro-OVR): {base_cv_auc:.4f}")
        print(f"基础模型5折交叉验证平均ACC: {base_cv_acc:.4f}")

        best_cv_auc = base_cv_auc
        best_cv_f1 = base_cv_f1
        best_cv_precision = base_cv_precision
        best_cv_recall = base_cv_recall
        best_cv_acc = base_cv_acc
        best_model_instance = copy.deepcopy(base_model)

        # 全量训练
        best_model_fitted = copy.deepcopy(base_model)
        best_model_fitted.fit(train_x, train_y)
        train_proba = best_model_fitted.predict_proba(train_x)
        train_classes = _get_classes_from_model(best_model_fitted, train_y)
        pred_idx = np.argmax(train_proba, axis=1)
        train_preds = np.array(train_classes)[pred_idx]
        try:
            best_train_auc = roc_auc_score(train_y, train_proba, multi_class="ovr", average=avg_strategy) * 100
        except Exception:
            best_train_auc = 0.0
        best_train_f1 = f1_score(train_y, train_preds, average=avg_strategy, zero_division=0)
        best_train_precision = precision_score(train_y, train_preds, average=avg_strategy, zero_division=0)
        best_train_recall = recall_score(train_y, train_preds, average=avg_strategy, zero_division=0)
        best_train_acc = accuracy_score(train_y, train_preds)

    except Exception as e:
        print(f"模型代码执行失败: {str(e)}")
        model_messages.append({"role": "assistant", "content": base_model_code})
        model_messages.append({
            "role": "user",
            "content": (
                f"Code execution failed, error: {type(e)} {e}.\n"
                f"Code: ```python{base_model_code}```\n"
                "Please generate the fixed code block. Note: The model class must NOT contain any cross-validation, "
                "parameter search, or grid search logic (such as GridSearchCV, RandomizedSearchCV). Only initialize "
                "the model with specific hyperparameters in __init__ and call fit directly in the fit method. "
                "For multiclass, predict_proba must return a 2D array of shape (n_samples, n_classes) aligned with self.model.classes_."
            )
        })
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

    # 参数优化提示词（仅多分类）
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
    param_prompt += """
    
    Important Constraints (must be strictly followed):
    1. Do NOT include any cross-validation, parameter search, or optimization logic (e.g., GridSearchCV, RandomizedSearchCV, CV loops).
    2. Initialize the base estimator in __init__ with explicit hyperparameters (e.g., n_estimators=300, max_depth=12). Do not rely on all defaults.
    3. fit must only call self.model.fit(X, y). No tuning inside the class.
    4. Implement predict and predict_proba. For multiclass, predict_proba must return a 2D array (n_samples, n_classes) whose columns follow self.model.classes_ exactly.
    5. The output must be a single, executable Python class with NO extra text.
    """

    param_messages = [
        {
            "role": "system",
            "content": (
                "You are a classification optimization assistant.\n"
                "Your task is to improve 5-fold cross-validation ROC AUC for a multiclass classifier by tuning hyperparameters only.\n"
                "Use macro-averaged One-vs-Rest (OVR) ROC AUC for evaluation.\n"
                "Your answer must contain only executable Python code with a single class.\n"
                "Do NOT include any cross-validation or parameter search logic in the class.\n"
                f"The class name you generate must be: myclassifier_{i+1}\n"
                "The class must have __init__, fit, predict, and predict_proba; predict_proba returns a 2D array."
            )
        },
        {"role": "user", "content": param_prompt},
    ]

    # 参数优化迭代
    for p_iter in range(Iterations):
        print(f"++++++ 第 {p_iter + 1} 次优化 +++++++")
        try:
            optimized_code = generate_model(args.llm, param_messages)
            optimized_code = clean_llm_code(optimized_code)

            param_new_class_name = f"myclassifier_{i+1}_param_{p_iter + 1}"
            optimized_code = optimized_code.replace(f"class myclassifier_{i+1}:", f"class {param_new_class_name}:")
            optimized_code = optimized_code.replace(f"class myclassifier_{i+1}_param_{p_iter}:", f"class {param_new_class_name}:")

            param_err = code_exec(optimized_code)
            if param_err is not None:
                print(f"代码执行错误: {param_err}")
                param_messages.extend([
                    {"role": "assistant", "content": optimized_code},
                    {"role": "user", "content": "Code execution failed, error: " + str(param_err) + 
                     ". Please fix and regenerate. Only keep __init__/fit/predict/predict_proba. "
                     "predict_proba must return 2D (n_samples, n_classes)."}
                ])
                continue

            print('---------------优化后的代码\n' + optimized_code)

            if param_new_class_name not in globals():
                raise NameError(f"Class {param_new_class_name} not defined in optimized code")

            model_class = globals()[param_new_class_name]
            optimized_model = model_class()

            # 交叉验证评估优化模型（多分类）
            cv_scores_auc, cv_scores_f1, cv_scores_precision, cv_scores_recall, cv_scores_acc = ([] for _ in range(5))
            for train_idx, val_idx in cv.split(train_x, train_y):
                cv_train_x, cv_val_x = train_x.iloc[train_idx], train_x.iloc[val_idx]
                cv_train_y, cv_val_y = train_y.iloc[train_idx], train_y.iloc[val_idx]

                model_copy = copy.deepcopy(optimized_model)
                model_copy.fit(cv_train_x, cv_train_y)

                proba = model_copy.predict_proba(cv_val_x)
                classes_fold = _get_classes_from_model(model_copy, cv_train_y)
                auc, f1, prec, rec, acc = _metrics_multiclass(cv_val_y, proba, classes_fold, avg_strategy)

                cv_scores_auc.append(auc)
                cv_scores_f1.append(f1)
                cv_scores_precision.append(prec)
                cv_scores_recall.append(rec)
                cv_scores_acc.append(acc)

            cv_auc = float(np.nanmean(cv_scores_auc)) * 100
            cv_f1 = float(np.mean(cv_scores_f1)) * 100
            cv_precision = float(np.mean(cv_scores_precision)) * 100
            cv_recall = float(np.mean(cv_scores_recall)) * 100
            cv_acc = float(np.mean(cv_scores_acc)) * 100
            print(f"优化模型5折交叉验证平均AUC(macro-OVR): {cv_auc:.4f}")
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

                # 全量训练更新（多分类）
                best_model_fitted = copy.deepcopy(optimized_model)
                best_model_fitted.fit(train_x, train_y)
                train_proba = best_model_fitted.predict_proba(train_x)
                train_classes = _get_classes_from_model(best_model_fitted, train_y)
                pred_idx = np.argmax(train_proba, axis=1)
                train_preds = np.array(train_classes)[pred_idx]
                try:
                    best_train_auc = roc_auc_score(train_y, train_proba, multi_class="ovr", average=avg_strategy) * 100
                except Exception:
                    best_train_auc = 0.0
                best_train_f1 = f1_score(train_y, train_preds, average=avg_strategy, zero_division=0)
                best_train_precision = precision_score(train_y, train_preds, average=avg_strategy, zero_division=0)
                best_train_recall = recall_score(train_y, train_preds, average=avg_strategy, zero_division=0)
                best_train_acc = accuracy_score(train_y, train_preds)

            param_messages.extend([
                {"role": "assistant", "content": optimized_code},
                {"role": "user",
                 "content": (
                     f"Current CV results - AUC: {cv_auc:.4f}, F1: {cv_f1:.4f}, Precision: {cv_precision:.4f}, "
                     f"Recall: {cv_recall:.4f}, ACC: {cv_acc:.4f}\n"
                     f"Best CV results - AUC: {best_cv_auc:.4f}, F1: {best_cv_f1:.4f}, Precision: {best_cv_precision:.4f}, "
                     f"Recall: {best_cv_recall:.4f}, ACC: {best_cv_acc:.4f}\n"
                     "Please continue optimizing hyperparameters to improve macro-OVR AUC. "
                     "No CV/search logic inside the class. predict_proba must return 2D (n_samples, n_classes)."
                 )}
            ])

        except Exception as e:
            print(f"Parameter tuning failed: {str(e)}")
            current_code = optimized_code if 'optimized_code' in locals() and optimized_code else "Failed to generate code"
            error_details = f"Error type: {type(e).__name__}, Details: {str(e)}"
            param_messages.extend([
                {"role": "assistant", "content": current_code},
                {"role": "user",
                 "content": (
                     f"Parameter tuning failed, {error_details}\n"
                     "Please analyze the error cause and fix the code. Focus on checking:\n"
                     "1. No cross-validation or parameter search logic inside the class\n"
                     "2. Class includes __init__, fit, predict, predict_proba\n"
                     "3. Hyperparameters are in valid ranges\n"
                     "4. For multiclass: predict_proba returns 2D aligned to self.model.classes_."
                 )}
            ])
            continue

    print(f"参数优化完成,最佳5折交叉验证AUC(macro-OVR): {best_cv_auc:.4f}")

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

# 加载多分类数据集 老数据集# cmc eucalyptus jungle_chess balance-scale
def load_origin_data_multi(dataset_name, seed):
    # 需要走 .pkl 的旧数据集关键词（子串匹配）
    old_keys = ('cmc','eucalyptus','jungle_chess','balance-scale')
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
        df_train, df_test = train_test_split(df, test_size=0.25, random_state=seed)
        # 'eucalyptus'数据集额外的处理
        if 'eucalyptus' in loc:
            # 如果存在，进行特定处理
            df_train = df_train.dropna()
            df_test = df_test.dropna()
            df_train.replace([float('inf'), float('-inf')], 0, inplace=True)
            df_test.replace([float('inf'), float('-inf')], 0, inplace=True)
        return df_train, df_test, target_column_name, dataset_description
     # 否则读取新的 CSV 数据集
    base_loc = "/home/usr01/cuicui/autoML-ensemble/new_dataSet/multiclass/"
    return load_new_dataset(dataset_name, base_loc=base_loc, seed=seed, test_size=0.2)

# 把字符串写入文件
def write_code_to_file(code:str, loc_path:str):
    os.makedirs(os.path.dirname(loc_path), exist_ok=True)
    with open(loc_path, 'w', encoding='utf-8') as f:
        f.write(code)


if __name__ == "__main__":

    # ==== CLI args (same shape as binary version) ====
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--gpus', default="0", type=str, help='GPU设置')
    parser.add_argument('-s', '--default_seed', default=42, type=int, help='随机种子')
    parser.add_argument('-l', '--llm', default='gpt-3.5-turbo', type=str, help='大模型')
    # parser.add_argument('-l', '--llm', default='gpt-4o', type=str, help='大模型')
    parser.add_argument('-e', '--exam_iterations', default=2, type=int, help='实验次数')
    parser.add_argument('-m', '--model_iterations', default=5, type=int, help='模型迭代次数')
    parser.add_argument('-p', '--param_iterations', default=1, type=int, help='参数调优次数')
    parser.add_argument('-ds', '--dataset_name', default='vehicle', type=str, help='数据集名称(多分类)')
    args = parser.parse_args()

    # cmc eucalyptus jungle_chess balance-scale
    # arrhythmia car myocardial vehicle

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    acc_list_stacking, f1_list_stacking, auc_list_stacking, pre_list_stacking, rec_list_stacking = [], [], [], [], []

    # === Experiments ===
    for ex in range(args.exam_iterations):
        print(f"================ Experiment {ex + 1} / {args.exam_iterations} ================")
        ds_name = args.dataset_name
        print(f"=========== Dataset {ds_name} (Multi-class) ===========")
        # loc = f"/home/usr01/cuicui/CAAFE++/tests/data/{ds_name}.pkl"

        # Seeds
        seed = args.default_seed
        random.seed(seed)
        np.random.seed(seed)

        # Load & split
        df_train, df_test, target_column_name, dataset_description = load_origin_data_multi(ds_name, seed)
        # 打印数据集描述
        # print("Dataset description:")
        # print(dataset_description)
        stacking_x, stacking_y = to_pd(df_train, target_column_name)
        train_x, val_x = train_test_split(stacking_x, test_size=0.25, random_state=seed)
        train_y, val_y = train_test_split(stacking_y, test_size=0.25, random_state=seed)
        test_x, test_y = to_pd(df_test, target_column_name)

        # Prompt building for model generation
        # samples_for_prompt = build_prompt_samples(df_train)
        model_prompt = get_model_prompt_multiclass_new(ds_description=dataset_description)

        model_messages = [
            {
                "role": "system",
                "content": (
                    "You are a top machine learning classification expert.\n"
                    "We are solving a MULTI-CLASS classification problem.\n"
                    "Primary objective: maximize macro-averaged ROC-AUC (one-vs-rest) on held-out data.\n"
                    "Also track macro F1, Precision, Recall, and Accuracy.\n"
                    "Only output valid Python code for a complete classifier class."
                ),
            },
            {"role": "user", "content": model_prompt},
        ]

        # Iterative model search
        model_iter = args.model_iterations
        best_auc_macro = -1.0
        best_code = None
        i = 0

        best_model_code_list = []
        best_model_instance_list = []  # unfitted
        best_fitted_model_instance_list = []  # fitted
        best_model_auc_list = []
        best_model_acc_list = []
        best_model_pre_list = []
        best_model_rec_list = []
        best_model_f1_list = []

        # 模型生成迭代
        while i < model_iter:
            try:
                # 1) Let LLM propose a classifier; rename class to keep unique
                code = generate_model(args.llm, model_messages)
                code = re.sub(r"class\s+myclassifier\w*\s*:", f"class myclassifier_{i + 1}:", code)
                print("----------------------------原始代码-----------------------")
                print(code)
            except Exception as e:
                print("Error in LLM API." + str(e))
                continue

            # 2) Compile
            err = code_exec(code)
            if err is not None:
                model_messages += [
                    {"role": "assistant", "content": code},
                    {
                        "role": "user",
                        "content": (
                            f"The classifier code execution failed with error: {type(err)} {err}.\n"
                            f"Code: ```python{code}```\n"
                            "Remember, ONLY output code. Generate the next block (fix it):"
                        ),
                    },
                ]
                continue

            # 3) Multi-class parameter optimization (to be implemented in this file)
            new_class_name = f"myclassifier_{i + 1}"
            # 参数优化
            # model_parameter_optimization_with_cv
            # 
            # model_parameter_optimization_with_trajectory model_optimization_with_param_constraints_3
            optimization_results = model_parameter_optimization_with_cv(
                new_class_name=new_class_name,
                base_model_code=code,
                Iterations=args.param_iterations,
                train_x=train_x,
                train_y=train_y,
                dataset_description=dataset_description,
                ds_name=ds_name,
                args=args,
                i=i,
            )

            optimized_model_code = optimization_results['optimized_model_code']
            optimized_model_auc = optimization_results['optimized_model_auc']  # macro AUC (ovr) expected
            optimized_model_acc = optimization_results['optimized_model_acc']
            optimized_model_f1 = optimization_results['optimized_model_f1']
            optimized_model_pre = optimization_results['optimized_model_precision']
            optimized_model_rec = optimization_results['optimized_model_recall']
            optimized_model_instance = optimization_results['optimized_model_instance']
            optimized_model_fitted = optimization_results['optimized_model_fitted']

            print(f"当前第 {i+1}/{args.model_iterations} 个模型的实验结果：")
            print(f"模型代码：\n{optimized_model_code}")
            print(f"Validation Accuracy (macro): {optimized_model_acc:.4f}")
            print(f"Validation F1 (macro): {optimized_model_f1:.4f}")
            print(f"Validation Precision (macro): {optimized_model_pre:.4f}")
            print(f"Validation Recall (macro): {optimized_model_rec:.4f}")
            print(f"Validation Macro AUC (ovr): {optimized_model_auc:.4f}")
            i += 1

            if optimized_model_auc > best_auc_macro:
                best_auc_macro = optimized_model_auc
                best_code = optimized_model_code

            # 4) Next-round guidance for LLM
            if len(optimized_model_code) > 10:
                model_messages += [
                    {"role": "assistant", "content": optimized_model_code},
                    {
                        "role": "user",
                        "content": (
                            "✅ The classifier executed successfully.\n\n"
                            f"📈 Current model Macro AUC (ovr): {optimized_model_auc:.4f}\n"
                            f"🏆 Best historical Macro AUC so far: {best_auc_macro:.4f}\n"
                            f"The code of the current best model: {best_code}\n\n"
                            "Propose a NEW classifier more likely to improve Macro AUC (ovr).\n"
                            "It must differ from prior models by model family or internal structure.\n"
                            "Output ONLY valid Python code for class `myclassifier`."
                        ),
                    },
                ]

            # 5) Accumulate for ensembling
            if optimized_model_fitted is not None and optimized_model_auc != 0 and hasattr(optimized_model_fitted, 'predict'):
                best_model_code_list.append(optimized_model_code)
                best_model_instance_list.append(optimized_model_instance)
                best_fitted_model_instance_list.append(optimized_model_fitted)
                best_model_auc_list.append(optimized_model_auc)
                best_model_acc_list.append(optimized_model_acc)
                best_model_pre_list.append(optimized_model_pre)
                best_model_rec_list.append(optimized_model_rec)
                best_model_f1_list.append(optimized_model_f1)

        print("+++++++++++++++ 模型生成结束 +++++++++++++++")
        print(f"本轮最佳模型 Macro AUC (ovr): {best_auc_macro:.4f}")
        
        # 构造集成学习提示词，让 LLM 输出集成学习基础模型的权重向量
        base_prompt = (
            "You are a machine learning expert. Your task is to ensemble multiple already-trained "
            "classification models to improve generalization on the validation set. "
            "Decide which models to include and assign non-negative weights that sum to 1. "
            "Use multi-class, macro-averaged metrics (Accuracy, ROC-AUC ovr, F1, Precision, Recall).\n\n"
            "Return: (1) a weight vector [w1, w2, ...] aligned with the number of models; "
            "(2) a concise rationale."
        )

        # Encode models + metrics into prompt (reuse helpers)
        model_code_prompt = get_model_code_prompt(
            model_code_list=best_model_code_list,
            val_acc_list=best_model_acc_list,
            val_auc_list=best_model_auc_list,
            use_code=False,
        )
        # Differences & diversity signals
        diff_matrix = get_model_pre_differences_multiclass(best_fitted_model_instance_list, val_x, val_y)
        diff_prompt = get_model_performance_differences_prompt(diff_matrix)
        jaccard_matrix = get_model_jaccard_fault_multiclass(best_fitted_model_instance_list, val_x, val_y)
        jaccard_prompt = get_model_intersection_union_prompt(jaccard_matrix)
        rescue_matrix = get_rescue_confidence_matrix_multiclass(best_fitted_model_instance_list, val_x, val_y)
        rescue_prompt = get_rescue_confidence_prompt(rescue_matrix)

        # Single-shot templates
        loc_path_D = f"{project_root}/multi_classification_ensemble/single-shot/single_shot_prompt_D.txt"
        loc_path_J = f"{project_root}/multi_classification_ensemble/single-shot/single_shot_prompt_J.txt"
        loc_path_R = f"{project_root}/multi_classification_ensemble/single-shot/single_shot_prompt_R.txt"
        single_shot_prompt_D = read_txt_file(loc_path_D)
        single_shot_prompt_J = read_txt_file(loc_path_J)
        single_shot_prompt_R = read_txt_file(loc_path_R)

        # Assemble weight prompts (D/J/R)
        get_model_weight_prompt_D = generate_llm_weight_prompt(
            base_prompt=base_prompt,
            model_code_prompt=model_code_prompt,
            model_performance_differences_prompt=diff_prompt,
            single_shot_prompt_D=single_shot_prompt_D,
        )
        get_model_weight_prompt_J = generate_llm_weight_prompt(
            base_prompt=base_prompt,
            model_code_prompt=model_code_prompt,
            model_Jaccard_Fault_prompt=jaccard_prompt,
            single_shot_prompt_J=single_shot_prompt_J,
        )
        get_model_weight_prompt_R = generate_llm_weight_prompt(
            base_prompt=base_prompt,
            model_code_prompt=model_code_prompt,
            model_rescue_confidence_prompt=rescue_prompt,
            single_shot_prompt_R=single_shot_prompt_R,
        )
        # 保存提示词
        write_code_to_file(get_model_weight_prompt_D, f"{project_root}/output/multi_classification/{ds_name}/prompt_D.txt")
        write_code_to_file(get_model_weight_prompt_J, f"{project_root}/output/multi_classification/{ds_name}/prompt_J.txt")
        write_code_to_file(get_model_weight_prompt_R, f"{project_root}/output/multi_classification/{ds_name}/prompt_R.txt")

        weight_vector_list = []
        for tag, prompt in zip(['D', 'J', 'R'], [get_model_weight_prompt_D, get_model_weight_prompt_J, get_model_weight_prompt_R]):
            print(f"\n================= Prompt Version {tag} =================")
            message = [
                {"role": "system", "content": base_prompt},
                {"role": "user", "content": prompt},
            ]
            llm_output = call_llm_chat_completion(args.llm, message)
            llm_text = llm_output.choices[0].message.content

            print("LLM Output:\n", llm_text)

            weight_vector_list.append(extract_weight_list(llm_text))

        # === Evaluate integrated ensemble strategies on TEST set ===
        # 权重整合
        w_final, info = integrate_three_strategies_cls(
            weight_vector_list, best_fitted_model_instance_list, val_x, val_y,
            method='hedge-geo', tau=10.0, shrink=0.1
        )
        print("Weights:", w_final)

        # 集成
        # 保存指标，最后一起打印
        integrated_metrics = ''
        result = stacking_ensemble(best_model_instance_list,
                    X_train=stacking_x,
                    y_train=stacking_y,
                    X_test=test_x,
                    y_test=test_y,
                    weight_list=w_final,
                )
        stacking_metrics = result['stacking_metrics']
        integrated_metrics += f"=========== Stacking 集成结果 ===========\n"
        integrated_metrics += f"Accuracy : {stacking_metrics['accuracy']*100:.3f}\n"
        integrated_metrics += f"F1 Score : {stacking_metrics['f1']*100:.3f}\n"
        integrated_metrics += f"Precision: {stacking_metrics['precision']*100:.3f}\n"
        integrated_metrics += f"Recall   : {stacking_metrics['recall']*100:.3f}\n"
        integrated_metrics += f"AUC      : {stacking_metrics['auc']*100:.3f}\n\n"
        # 保存结果到对应列表中
        acc_list_stacking.append(stacking_metrics['accuracy']*100)
        f1_list_stacking.append(stacking_metrics['f1']*100)
        pre_list_stacking.append(stacking_metrics['precision']*100)
        rec_list_stacking.append(stacking_metrics['recall']*100)
        auc_list_stacking.append(stacking_metrics['auc']*100)
        print(integrated_metrics)

    # 保存实验结果  先打印，再保存
    final_results = f"================= {ds_name} 集成结果统计指标 =================\n"
    final_results += "================= Stacking 集成结果 =================\n"
    final_results += f"Accuracy : {np.mean(acc_list_stacking):.2f} ± {np.std(acc_list_stacking):.2f}\n"
    final_results += f"F1 Score : {np.mean(f1_list_stacking):.2f} ± {np.std(f1_list_stacking):.2f}\n"
    final_results += f"Precision: {np.mean(pre_list_stacking):.2f} ± {np.std(pre_list_stacking):.2f}\n"
    final_results += f"Recall   : {np.mean(rec_list_stacking):.2f} ± {np.std(rec_list_stacking):.2f}\n"
    final_results += f"AUC      : {np.mean(auc_list_stacking):.2f} ± {np.std(auc_list_stacking):.2f}\n\n"
    print(final_results)
    # # 保存到文件
    # results_path = f"{project_root}/multi_classification_ensemble/res/{ds_name}.txt"
    # write_code_to_file(final_results,results_path)