export CUDA_VISIBLE_DEVICES=0,1,2
export num_gpus=3
export CUBLAS_WORKSPACE_CONFIG=":16:8" # https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
export PYTHONHASHSEED=0
export task_name="banking77"
export model_name="gpt2-medium"
export output_dir="~/data/eff_tl"
export split_ratio=0.25
export TORCH_DISTRIBUTED_DEBUG=INFO

learning_rates="1e-5"
num_prefixes="5 10 50" 

for num_prefix in $num_prefixes; do
    for learning_rate in $learning_rates; do
        python -m torch.distributed.launch --nproc_per_node=$num_gpus \
            run_ood.py \
            --model_name_or_path $model_name \
            --task_name $task_name \
            --do_train \
            --do_eval \
            --do_predict \
            --max_seq_length 128 \
            --per_device_train_batch_size 8 \
            --per_device_eval_batch_size 8 \
            --learning_rate $learning_rate \
            --num_train_epochs 20 \
            --split_ratio $split_ratio \
            --output_dir $output_dir/$model_name/$task_name/prefix-tuning/$split_ratio/$num_prefix/$learning_rate \
            --overwrite_output_dir \
            --logging_steps 10 \
            --logging_dir $output_dir/$model_name/$task_name/prefix-tuning/$split_ratio/$num_prefix/$learning_rate \
            --evaluation_strategy epoch \
            --save_strategy epoch \
            --warmup_ratio 0.06 \
            --seed 0 \
            --weight_decay 0.1 \
            --apply_prefix \
            --num_prefix $num_prefix \
            --mid_dim 512 \
            --save_total_limit 1
    done
done