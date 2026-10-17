#!/bin/bash
set -e

CUDA_VISIBLE_DEVICES=0 python tracking/train.py \
  --script sutrack \
  --config sutrack_b224_rgbt \
  --save_dir . \
  --mode single

# bash test_rgbt.sh
