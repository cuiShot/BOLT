## 脚本命令 chmod +x run_regression.sh   ./run_regression.sh

# concrete boston winequality california insurance
# concrete
# python llm_auto_ensemble_regression.py --llm 'gpt-3.5-turbo' --exam_iterations 3 --dataset_name concrete

# # boston
# python llm_auto_ensemble_regression.py --llm 'gpt-3.5-turbo' --exam_iterations 3 --dataset_name boston

# # winequality
# python llm_auto_ensemble_regression.py --llm 'gpt-3.5-turbo' --exam_iterations 3 --dataset_name winequality

# # california
# python llm_auto_ensemble_regression.py --llm 'gpt-3.5-turbo' --exam_iterations 3 --model_iterations 5 --param_iterations 5 --dataset_name california

# # insurance
# python llm_auto_ensemble_regression.py --llm 'gpt-3.5-turbo' --exam_iterations 2 --model_iterations 5 --param_iterations 5 --dataset_name insurance

# 使用环境
# conda activate GCR-hu

# 消融实验
python regression_ablation.py --dataset_name concrete
# python regression_ablation.py --dataset_name boston
python regression_ablation.py --dataset_name winequality
python regression_ablation.py --dataset_name crab