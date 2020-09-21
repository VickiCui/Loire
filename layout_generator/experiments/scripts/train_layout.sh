#!/bin/bash
set -x
set -e
export PYTHONUNBUFFERED="True"

LOG="experiments/logs/train_layout.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

time  ./tools/my_train_layout.py --cuda --parallel\
    --n_epochs=15 --batch_size=8 --accumulation_steps=4\
    --num_workers=4 --warmup=0.05 --finetune_lr=5e-5\
    --data=coco --data_dir=PATH_TO_COCO\
    --log_dir=./ViBERT --exp_name=ViBERT\
    --max_input_length=64 --max_output_length=20\
    --output_cls_size=83\
    --n_embed=768 --n_src_hidden=768 --n_conv_hidden=256 --n_tgt_hidden=768\
    --attn_emb=False  --test=False --reduction=False --bidirectional=False\
    --mse=True --large_first=True 
