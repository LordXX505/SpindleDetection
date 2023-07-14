#!/bin/bash
#JOB_DIR='./log'
expert='E1'
python submitit_pretrain.py \
    --nodes 1 \
    --ngpus 1 \
    --partition GPU36 \
    --batch_size 32 \
    --model Unet_drop_fs2_fnfp \
    --epochs 200 \
    --lr 1e-4 \
    --min_lr 5e-8 \
    --weight_decay 1e-8 \
    --dist_eval \
    --optim AdamW \
    --accum_iter 1 \
    --lr_policy plateau \
    --device cuda \
    --dropout 0.25 \
    --Criterion fpfn \
    --expert $expert


#    --Using_deep \



