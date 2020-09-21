#!/bin/bash

CUDA_VISIBLE_DEVICES=0,1 python -u train.py --cuda --num_workers=1 --parallel\
    --epochs=5 --model_name=Loire --do_train\
    --task=commonsenseqa --train_size=xl\
    --data_dir=PATH_TO_DATA\
    --model=roberta-base --max_seq_length=128 --seed=1\
    --batch_size=16 --accumulation_steps=1 --log_per_steps=500\
    --warmup=0.1 --lr=1e-5 --adam_betas='(0.9, 0.98)' --adam_eps=1e-06\
    --dropout=0.3 --weight_decay=0.01\
    --feature --norm --proj=f2t prefix\
    --pretrained_bert=PATH_TO_VIBERT\
    --save_ckpt