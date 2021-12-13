import subprocess

def main():
    num_gpus = 4
    model_name = 'gpt2-large'
    task_name = 'clinc150'
    output_dir = "~/share/data/eff_tl"
    split_ratio = "full"
    learning_rate = 1e-5
    ddp_script = f'python -m torch.distributed.launch --nproc_per_node={num_gpus} --master_port="8888" '\
        'run_ood.py '\
        f'--model_name_or_path {model_name} '\
        f'--task_name {task_name} '\
        '--do_train '\
        '--do_eval '\
        '--do_predict '\
        '--max_seq_length 128 '\
        '--per_device_train_batch_size 2 '\
        '--per_device_eval_batch_size 2 '\
        '--learning_rate $learning_rate '\
        '--num_train_epochs 20 '\
        '--split False ' \
        f'--output_dir {output_dir}/{model_name}/{task_name}/fine-tuning/{split_ratio}/{learning_rate} '\
        '--overwrite_output_dir '\
        '--logging_steps 10 ' \
        '--pad_to_max_length True' \
        f'--logging_dir {output_dir}/{model_name}/{task_name}/fine-tuning/{split_ratio}/{learning_rate} ' \
        '--evaluation_strategy epoch ' \
        '--save_strategy epoch ' \
        '--warmup_ratio 0.06 '\
        '--seed 0 '\
        '--weight_decay 0.1 '\
        '--save_total_limit 1 '
    
    
    # subprocess.call(ddp_script, shell=True)
    bash_script = 'bash main_ddp.sh'
    subprocess.call(bash_script, shell=True)


if __name__ == '__main__':
    main()