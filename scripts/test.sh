#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

python -m src.test \
--num_workers 16 \
--model_path model/pretrained/pretrained.pth \
--test_json data/test_pairs.json \
--config configs/pair_sync.yaml