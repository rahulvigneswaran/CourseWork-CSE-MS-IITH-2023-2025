#!/bin/bash

if [[ $1 == 'train' ]]; then
    echo 'Run training...'
    python3 train_synthetic.py \
        --data ../data \
        --dataset reverse \
        --n_layer 4 \
        --d_model 128 \
        --n_head 4 \
        --d_head 64 \
        --d_inner 256 \
        --dropout 0.1 \
        --dropatt 0.0 \
        --optim adam \
        --lr 0.0001 \
        --lr_min 0.000001 \
        --decay_rate 0.5 \
        --log_interval 4000 \
        --eval_interval 12000 \
        --scheduler 'dev_perf' \
        --patience 10 \
        --warmup_step 0 \
        --max_step 700000 \
        --tgt_len 48 \
        --mem_len 0 \
        --eval_tgt_len 48 \
        --batch_size 32 \
        --num_mem_tokens 0 \
        --mem_backprop_depth 0 \
        --mem_at_end \
        --max_eval_steps 50 \
        --read_mem_from_cache \
        --attn_type 0 \
        --answer_size 24\
        --cuda\
        --multi_gpu\
        --device_ids 0 1\
        --work_dir ../evaluation/noname \
        ${@:2}
else
    echo 'unknown argment 1'
fi
