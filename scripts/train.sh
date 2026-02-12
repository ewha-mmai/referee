#!/bin/bash

export CUDA_VISIBLE_DEVICES=3
EXP_DIR=./exp

python -m src.train \
--num_workers 16 \
--exp_dir ${EXP_DIR} \
--config configs/pair_sync.yaml \
--model_path model/pretrained/pretrained.pth \
--train_json data/train_set.json \
--val_json data/val_set.json \
--start_epoch 0 \
--start_step 1 \
--n_epochs 1000 \
--n_print_steps 50 \
--save_model
