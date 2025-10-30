## 脚本命令 chmod +x run_multi.sh   ./run_multi.sh

# cmc eucalyptus jungle_chess balance-scale
# cmc
python llm_auto_ensemble_multi_classification.py --llm 'gpt-4o' --exam_iterations 2 --model_iterations 5 --param_iterations 5 --dataset_name cmc

# eucalyptus
python llm_auto_ensemble_multi_classification.py --llm 'gpt-4o' --exam_iterations 2 --model_iterations 5 --param_iterations 5 --dataset_name eucalyptus

# jungle_chess
python llm_auto_ensemble_multi_classification.py --llm 'gpt-4o' --exam_iterations 2 --model_iterations 5 --param_iterations 5 --dataset_name jungle_chess

# balance-scale
python llm_auto_ensemble_multi_classification.py --llm 'gpt-4o' --exam_iterations 2 --model_iterations 5 --param_iterations 5 --dataset_name balance-scale

