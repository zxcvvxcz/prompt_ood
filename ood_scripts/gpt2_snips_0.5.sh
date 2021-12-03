export CUDA_VISIBLE_DEVICES=0,1,2
export num_gpus=3
export CUBLAS_WORKSPACE_CONFIG=":16:8" # https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
export PYTHONHASHSEED=0
export output_dir="~/data/eff_tl/gpt2_medium/snips"
export TORCH_DISTRIBUTED_DEBUG=INFO

learning_rates="1e-5"

for learning_rate in $learning_rates; do
    # torchrun --nproc_per_node=$num_gpus \
    #     run_ood.py \
    # python3 run_ood.py \
    python3 -m torch.distributed.launch --nproc_per_node=$num_gpus \
        run_ood.py \
        --model_name_or_path gpt2-medium \
        --task_name sst2 \
        --do_train \
        --do_eval \
        --do_predict \
        --max_seq_length 128 \
        --per_device_train_batch_size 2 \
        --per_device_eval_batch_size 2 \
        --learning_rate $learning_rate \
        --num_train_epochs 20 \
        --split_ratio 0.5 \
        --output_dir $output_dir/fine-tuning/$learning_rate \
        --overwrite_output_dir \
        --logging_steps 50 \
        --pad_to_max_length False \
        --logging_dir $output_dir/fine-tuning/$learning_rate \
        --evaluation_strategy epoch \
        --save_strategy epoch \
        --warmup_ratio 0.06 \
        --seed 0 \
        --weight_decay 0.1 \
        --save_total_limit 1
done