export CUDA_VISIBLE_DEVICES=0

basic_path=<your ckpt path>
task_suite_name=libero130
json_path=/data/libero/$task_suite_name-ac.json
ckpt_name=latest.pth

# Output debugging information
echo "Basic path: $basic_path"
echo "Checkpoint name: $ckpt_name"
echo "Task suite name: $task_suite_name"
echo "Json path: $json_path"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "Python version: $(python --version)"


# Run the evaluation script
python eval/eval_libero.py \
    --basic_path $basic_path \
    --statistic_name statistics.json \
    --ckpt_name $ckpt_name \
    --save_path $basic_path \
    --json_path $json_path\
    --task_suite_name $task_suite_name \
    --num_episodes 10 \
    --eval_horizon 300 \
    --cross_modal True 

