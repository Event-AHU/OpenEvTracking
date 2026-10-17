#!/usr/bin/env bash
set -e

export CUDA_VISIBLE_DEVICES=0

cd "$(dirname "$0")/VOT22RGBD_workspace"
conda run -n sutrack python ../tracking/vot_compat.py evaluate sutrack_b224
conda run -n sutrack python ../tracking/vot_compat.py analysis sutrack_b224 --nocache
