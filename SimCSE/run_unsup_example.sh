#!/bin/bash

# In this example, we show how to train SimCSE on unsupervised Wikipedia data.
# If you want to train it with multiple GPU cards, see "run_sup_example.sh"
# about how to use PyTorch's distributed data parallel.

python train.py \
    --model_name_or_path bert-base-uncased \
    --train_file /home/linzhisheng/Digital_Research/filter_model/data/filter_model_data.csv \
    --validate_file /home/linzhisheng/Digital_Research/filter_model/data/filter_model_manual_400.csv \
    --output_dir result/bert-base-uncased-test \
    --num_train_epochs 50 \
    --per_device_train_batch_size 64 \
    --learning_rate 5e-5 \
    --max_seq_length 32 \
    --evaluation_strategy steps \
    --eval_steps 8 \
    --logging_steps 8\
    --pooler_type cls \
    --mlp_only_train \
    --overwrite_output_dir \
    --temp 0.05 \
    --do_train \
    --do_eval \
    --fp16 \
    "$@"