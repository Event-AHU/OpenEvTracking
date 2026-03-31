#!/bin/bash
# Usage:
#   Test Mode (Single GPU):
#       ./train.sh -c visevent_base -m test
#   Train Mode (Multi GPU):
#       ./train.sh -c sortrack_fe108_sor -m train -g 1 -p 29508

# 环境与基础配置
PROJECT_ROOT=$(pwd)
export PYTHONPATH=$PROJECT_ROOT:$PYTHONPATH
# 显卡序号
export CUDA_VISIBLE_DEVICES=4
# 离线
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# 默认参数设置
SCRIPT="ceutrack"             # 对应的脚本名
CONFIG="ceutrack_visevent_sor"         # yaml 配置文件名
MODE="train"                   # 默认模式：test | train
GPUS=1                      
MASTER_PORT=$((29500 + $RANDOM % 100)) # 随机端口
RESUME_PATH=""                  # 断点 cp 路径

# 命令行参数
while getopts "c:m:g:p:r:" opt; do
  case $opt in
    c) CONFIG=$OPTARG ;;
    m) MODE=$OPTARG ;;
    g) GPUS=$OPTARG ;;
    p) MASTER_PORT=$OPTARG ;;
    r) RESUME_PATH=$OPTARG ;;
    *) echo "Usage: ./train.sh -c config -m mode -g gpus -r resume_path" && exit 1 ;;
  esac
done

# 输出路径与日志管理
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="$PROJECT_ROOT/output/${CONFIG}_${TIMESTAMP}"
LOG_FILE="$OUTPUT_DIR/${MODE}-${CONFIG}.log"

echo "============================================"
echo "STORTrack"
echo "Project Root: $PROJECT_ROOT"
echo "Config:       $CONFIG"
echo "Mode:         $MODE"
echo "GPUs:         $GPUS"
echo "Output:       $OUTPUT_DIR"
echo "Master Port:  $MASTER_PORT"
echo "============================================"

mkdir -p "$OUTPUT_DIR"

# 执行逻辑 

if [ "$MODE" == "test" ]; then
    # 测试：单卡前台运行
    echo "Running SMOKE TEST (Single GPU)..."
    python lib/train/run_training.py \
        --script $SCRIPT \
        --config "$CONFIG" \
        --save_dir "$OUTPUT_DIR" \
        --use_lmdb 0 \
        --local_rank -1 
        # 2>&1 | tee "$LOG_FILE"

elif [ "$MODE" == "train" ]; then
    # 多卡分布后台运行
    echo "Training started in BACKGROUND mode..."
    echo "[ACTION] 使用以下命令实时监控进度:"
    echo "    tail -f $LOG_FILE"
    
    nohup torchrun \
        --nproc_per_node=$GPUS \
        --master_port=$MASTER_PORT \
        lib/train/run_training.py \
        --script $SCRIPT \
        --config "$CONFIG" \
        --save_dir "$OUTPUT_DIR" \
        --use_lmdb 0  \
        --resume "$RESUME_PATH" > "$LOG_FILE" 2>&1 &    
    
    # 记录主进程 PID
    MAIN_PID=$!
    echo $MAIN_PID > "$OUTPUT_DIR/train.pid"

    # 生成停止脚本
    STOP_SCRIPT="$OUTPUT_DIR/stop_train.sh"
    echo "#!/bin/bash" > $STOP_SCRIPT
    echo "echo 'Stopping STORTrack Training (PID: $MAIN_PID)...'" >> $STOP_SCRIPT
    echo "pkill -P $MAIN_PID" >> $STOP_SCRIPT 
    echo "kill -9 $MAIN_PID" >> $STOP_SCRIPT  
    chmod +x $STOP_SCRIPT

    # 创建指向最新输出的软链接
    ln -snf "$OUTPUT_DIR" "$PROJECT_ROOT/output/latest"

    echo "============================================"
    echo "PID:           $MAIN_PID"
    echo "Log:           $LOG_FILE"
    echo "Stop Script:   bash $STOP_SCRIPT"
    echo "Latest Link:   output/latest"
    echo "============================================"
else
    echo "错误: $MODE. 请使用 'test' 或 'train'."
    exit 1
fi