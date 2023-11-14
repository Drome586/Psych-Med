#!/usr/bin/env bash

python run_glue.py --model_name_or_path pretrained/BERT --task_name med --classification_task_name disease \
        --do_train --do_eval --data_dir data --max_seq_length 512 --per_gpu_eval_batch_size 8 \
        --per_gpu_train_batch_size 8 --learning_rate 1e-5 --num_train_epochs 3 \
        --output_dir output --model_type bert --class_type base --gradient_accumulation_steps 2
