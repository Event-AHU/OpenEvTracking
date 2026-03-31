#!/bin/bash
# tracking/test.sh
# Usage:
#   单序列可视化:  ./test.sh -c sortrack_visevent -m smoke -g 0 -e 40
#   全量评估:      ./test.sh -c sortrack_fe108_sor -m eval  -g 0 -t 8 -e 7 -d fe108
#   指定序列:      ./test.sh -c sortrack_fe108 -m smoke -g 5 -e 30 -s bike222

PROJECT_ROOT=$(pwd)
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

# 默认参数
TRACKER="sortrack"
CONFIG="sortrack_visevent"
MODE="smoke"
GPU=6
DATASET="FE108"
THREADS=8
SEQUENCE="UAV_long_001"
EPOCH=40                     # 必须通过 -e 指定，与 yaml TEST.EPOCH 解耦

while getopts "c:m:g:t:s:d:e:" opt; do
  case $opt in
    c) CONFIG="$OPTARG"   ;;
    m) MODE="$OPTARG"     ;;
    g) GPU="$OPTARG"      ;;
    t) THREADS="$OPTARG"  ;;
    s) SEQUENCE="$OPTARG" ;;
    d) DATASET="$OPTARG"  ;;
    e) EPOCH="$OPTARG"    ;;
    *) echo "Usage: ./test.sh -c config -m [smoke|eval] -g gpu_id -e epoch [-s seq] [-t threads] [-d dataset]"
       exit 1 ;;
  esac
done

# EPOCH 必须指定
if [ -z "$EPOCH" ]; then
    echo "[ERROR] -e EPOCH is required. Example: ./test.sh -e 35"
    exit 1
fi

export CUDA_VISIBLE_DEVICES=$GPU

# checkpoint 路径
CKPT_PATH="${PROJECT_ROOT}/output/checkpoints/train/sortrack/${CONFIG}/sortrack_ep$(printf '%04d' ${EPOCH}).pth.tar"

echo "=========================================================="
echo "  sortrack Test/Eval Engine"
echo "=========================================================="
echo "  Project Root : ${PROJECT_ROOT}"
echo "  Config       : ${CONFIG}"
echo "  Epoch        : ${EPOCH}"
echo "  Checkpoint   : ${CKPT_PATH}"
echo "  Mode         : ${MODE}"
echo "  Dataset      : ${DATASET}"
echo "  GPU          : ${GPU}"
echo "=========================================================="

# 提前验证 checkpoint 存在
if [ ! -f "${CKPT_PATH}" ]; then
    echo "[ERROR] Checkpoint not found: ${CKPT_PATH}"
    echo "Available checkpoints:"
    ls "${PROJECT_ROOT}/output/checkpoints/train/sortrack/${CONFIG}/" 2>/dev/null || \
        echo "  <directory not found>"
    exit 1
fi
echo "[OK] Checkpoint verified."

if [ "$MODE" == "smoke" ]; then
    echo "[ACTION] Smoke test on sequence: ${SEQUENCE}"
    python tracking/test.py \
        "${TRACKER}" "${CONFIG}" \
        --dataset_name "${DATASET}" \
        --sequence     "${SEQUENCE}" \
        --debug        1 \
        --threads      0 \
        --num_gpus     1 \
        --runid        "${EPOCH}"

elif [ "$MODE" == "eval" ]; then
    RESULT_DIR="${PROJECT_ROOT}/output/test/tracking_results/${TRACKER}/${CONFIG}_$(printf '%03d' ${EPOCH})"
    echo "[ACTION] Full evaluation → ${RESULT_DIR}"

    python tracking/test.py \
        "${TRACKER}" "${CONFIG}" \
        --dataset_name "${DATASET}" \
        --threads      "${THREADS}" \
        --debug        0 \
        --num_gpus     1 \
        --runid        "${EPOCH}"

    echo "=========================================================="
    echo "[DONE] Results saved to: ${RESULT_DIR}"
    echo "[NEXT] Run evaluation toolkit or:"
    echo "       python analysis/analysis_results.py --config ${CONFIG} --epoch ${EPOCH}"
    echo "=========================================================="

else
    echo "[ERROR] Unknown mode '${MODE}'. Use 'smoke' or 'eval'."
    exit 1
fi