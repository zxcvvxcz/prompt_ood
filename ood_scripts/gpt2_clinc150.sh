export CUDA_VISIBLE_DEVICES=0,1,2,3
export num_gpus=4
export CUBLAS_WORKSPACE_CONFIG=":16:8" # https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
export PYTHONHASHSEED=0
export task_name="clinc150"
export model_name="gpt2-medium"
export output_dir="~/data/eff_tl"
export split_ratio="full"
export TORCH_DISTRIBUTED_DEBUG=INFO

learning_rates="1e-5"

for learning_rate in $learning_rates; do
    # python run_ood.py \
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
        --split False \
        --output_dir $output_dir/$model_name/$task_name/fine-tuning/$split_ratio/$learning_rate \
        --overwrite_output_dir \
        --logging_steps 10 \
        --pad_to_max_length True \
        --logging_dir $output_dir/$model_name/$task_name/fine-tuning/$split_ratio/$learning_rate \
        --evaluation_strategy epoch \
        --save_strategy epoch \
        --warmup_ratio 0.06 \
        --seed 0 \
        --weight_decay 0.1 \
        --save_total_limit 1
done