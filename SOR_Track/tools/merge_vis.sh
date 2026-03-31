#!/bin/bash
# tools/merge_vis.sh
# 调用 tools/merge_vis.py，对两个模型配置的 debug_vis 输出进行对比图合并
#
# Usage:
#   基础用法（单序列）:
#     ./merge_vis.sh -a ceutrack_fe108 -b ceutrack_fe108_sor -s bike222
#
#   指定 layout:
#     ./merge_vis.sh -a ceutrack_fe108 -b ceutrack_fe108_sor -s bike222 -l overlay
#
#   生成视频:
#     ./merge_vis.sh -a ceutrack_fe108 -b ceutrack_fe108_sor -s bike222 -v
#
#   自定义 label:
#     ./merge_vis.sh -a ceutrack_fe108 -b ceutrack_fe108_sor -s bike222 \
#                   --la "Base-ep020" --lb "SOR-ep024"
#
#   批量所有序列（-s all）:
#     ./merge_vis.sh -a ceutrack_fe108 -b ceutrack_fe108_sor -s all

PROJECT_ROOT=$(pwd)
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

#  默认参数 
TRACKER="ceutrack"
CONFIG_A=""
CONFIG_B=""
SEQUENCE=""
LAYOUT="side_by_side"          # side_by_side | overlay | quad
HEIGHT=360
VIDEO=0
FPS=10
LABEL_A=""                     # 不指定时自动从 CONFIG_A 生成
LABEL_B=""
DEBUG_ROOT="${PROJECT_ROOT}/debug_vis"
OUT_ROOT="${PROJECT_ROOT}/debug_vis/compare"

#  参数解析 
# 支持短选项和两个长选项 --la / --lb
while [[ $# -gt 0 ]]; do
  case $1 in
    -a)  CONFIG_A="$2";  shift 2 ;;
    -b)  CONFIG_B="$2";  shift 2 ;;
    -s)  SEQUENCE="$2";  shift 2 ;;
    -l)  LAYOUT="$2";    shift 2 ;;
    -H)  HEIGHT="$2";    shift 2 ;;
    -f)  FPS="$2";       shift 2 ;;
    -v)  VIDEO=1;        shift   ;;
    --la) LABEL_A="$2";  shift 2 ;;
    --lb) LABEL_B="$2";  shift 2 ;;
    -o)  OUT_ROOT="$2";  shift 2 ;;
    -h|--help)
      echo "Usage: ./merge_vis.sh -a CONFIG_A -b CONFIG_B -s SEQ [options]"
      echo ""
      echo "  必选:"
      echo "    -a CONFIG_A     Model A 的 config 名（即 debug_vis 中的子目录名）"
      echo "    -b CONFIG_B     Model B 的 config 名"
      echo "    -s SEQ          序列名，或 'all' 处理两个目录的公共序列"
      echo ""
      echo "  可选:"
      echo "    -l LAYOUT       布局: side_by_side(默认) | overlay | quad"
      echo "    -H HEIGHT       每个面板高度，单位像素 [默认: 360]"
      echo "    -v              同时导出 .mp4 视频"
      echo "    -f FPS          视频帧率 [默认: 10]"
      echo "    --la LABEL_A    Model A 的显示标签 [默认: CONFIG_A]"
      echo "    --lb LABEL_B    Model B 的显示标签 [默认: CONFIG_B]"
      echo "    -o OUT_ROOT     输出根目录 [默认: debug_vis/compare]"
      exit 0 ;;
    *)
      echo "[ERROR] Unknown option: $1"
      echo "Run './merge_vis.sh -h' for usage."
      exit 1 ;;
  esac
done

#  参数校验 
if [ -z "$CONFIG_A" ] || [ -z "$CONFIG_B" ]; then
    echo "[ERROR] -a CONFIG_A and -b CONFIG_B are both required."
    echo "  Example: ./merge_vis.sh -a ceutrack_fe108 -b ceutrack_fe108_sor -s bike222"
    exit 1
fi

if [ -z "$SEQUENCE" ]; then
    echo "[ERROR] -s SEQUENCE is required. Use '-s all' for batch processing."
    exit 1
fi

# label 默认值
[ -z "$LABEL_A" ] && LABEL_A="$CONFIG_A"
[ -z "$LABEL_B" ] && LABEL_B="$CONFIG_B"

DIR_A_ROOT="${DEBUG_ROOT}/${TRACKER}/${CONFIG_A}"
DIR_B_ROOT="${DEBUG_ROOT}/${TRACKER}/${CONFIG_B}"

#  头部信息 
echo "=========================================================="
echo "  CEUTrack Visualization Merge"
echo "=========================================================="
echo "  Model A      : ${LABEL_A}"
echo "    dir        : ${DIR_A_ROOT}"
echo "  Model B      : ${LABEL_B}"
echo "    dir        : ${DIR_B_ROOT}"
echo "  Sequence     : ${SEQUENCE}"
echo "  Layout       : ${LAYOUT}"
echo "  Height       : ${HEIGHT}px"
echo "  Video export : $([ $VIDEO -eq 1 ] && echo 'YES' || echo 'no')"
echo "  Output root  : ${OUT_ROOT}"
echo "=========================================================="

#  核心：构建 merge 命令 
build_video_flag() {
    [ $VIDEO -eq 1 ] && echo "--video" || echo ""
}

run_merge() {
    local seq="$1"
    local dir_a="${DIR_A_ROOT}/${seq}"
    local dir_b="${DIR_B_ROOT}/${seq}"
    local out_dir="${OUT_ROOT}/${CONFIG_A}_vs_${CONFIG_B}/${seq}"

    # 检查源目录
    if [ ! -d "$dir_a" ]; then
        echo "[SKIP] Dir not found for Model A: ${dir_a}"
        return 1
    fi
    if [ ! -d "$dir_b" ]; then
        echo "[SKIP] Dir not found for Model B: ${dir_b}"
        return 1
    fi

    echo "[RUN] Merging sequence: ${seq}"
    echo "      A: ${dir_a}"
    echo "      B: ${dir_b}"
    echo "      → ${out_dir}"

    python tools/merge_vis.py \
        --dir_a   "${dir_a}"    \
        --dir_b   "${dir_b}"    \
        --out_dir "${out_dir}"  \
        --label_a "${LABEL_A}"  \
        --label_b "${LABEL_B}"  \
        --layout  "${LAYOUT}"   \
        --height  "${HEIGHT}"   \
        --fps     "${FPS}"      \
        $(build_video_flag)

    local exit_code=$?
    if [ $exit_code -eq 0 ]; then
        echo "[OK] Done: ${seq}"
        # 打印输出帧数
        local n_frames=$(ls "${out_dir}"/*.jpg 2>/dev/null | wc -l)
        echo "     Generated ${n_frames} comparison frames."
        [ $VIDEO -eq 1 ] && echo "     Video: ${out_dir}/_compare.mp4"
    else
        echo "[ERROR] merge_vis.py returned exit code ${exit_code} for seq: ${seq}"
    fi
    echo ""
    return $exit_code
}

#  单序列 or 批量 
if [ "$SEQUENCE" == "all" ]; then
    # 取两个目录下序列名的交集
    seqs_a=$(ls -d "${DIR_A_ROOT}"/*/ 2>/dev/null | xargs -I{} basename {})
    seqs_b=$(ls -d "${DIR_B_ROOT}"/*/ 2>/dev/null | xargs -I{} basename {})

    if [ -z "$seqs_a" ]; then
        echo "[ERROR] No sequences found in: ${DIR_A_ROOT}"
        exit 1
    fi
    if [ -z "$seqs_b" ]; then
        echo "[ERROR] No sequences found in: ${DIR_B_ROOT}"
        exit 1
    fi

    # bash 交集：只处理两边都有的序列
    common_seqs=()
    for seq in $seqs_a; do
        if echo "$seqs_b" | grep -qx "$seq"; then
            common_seqs+=("$seq")
        fi
    done

    if [ ${#common_seqs[@]} -eq 0 ]; then
        echo "[ERROR] No common sequences between the two config directories."
        echo "  A has: $(echo $seqs_a | tr ' ' '\n' | head -5) ..."
        echo "  B has: $(echo $seqs_b | tr ' ' '\n' | head -5) ..."
        exit 1
    fi

    echo "[BATCH] Found ${#common_seqs[@]} common sequences:"
    printf '  %s\n' "${common_seqs[@]}"
    echo ""

    success=0
    fail=0
    for seq in "${common_seqs[@]}"; do
        run_merge "$seq"
        [ $? -eq 0 ] && ((success++)) || ((fail++))
    done

    echo "=========================================================="
    echo "[BATCH DONE] Success: ${success}  Failed: ${fail}"
    echo "  Output: ${OUT_ROOT}/${CONFIG_A}_vs_${CONFIG_B}/"
    echo "=========================================================="

else
    # 单序列
    run_merge "$SEQUENCE"
    exit_code=$?
    if [ $exit_code -eq 0 ]; then
        echo "=========================================================="
        echo "[DONE]"
        echo "  Frames : ${OUT_ROOT}/${CONFIG_A}_vs_${CONFIG_B}/${SEQUENCE}/"
        [ $VIDEO -eq 1 ] && \
        echo "  Video  : ${OUT_ROOT}/${CONFIG_A}_vs_${CONFIG_B}/${SEQUENCE}/_compare.mp4"
        echo "=========================================================="
    fi
    exit $exit_code
fi