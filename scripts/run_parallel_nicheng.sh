#!/bin/bash

# 基于你的nicheng数据集配置的并行处理脚本
# 使用tmux创建16个会话，每个处理不同的文件范围

# 设置工作目录
cd /data/home/wudezhi/project/school/x-meta

# 你的原始配置参数
MODEL_PATH="/data/home/wudezhi/project/school/x-meta/checkpoints/gutclip_exp_best_20250724-090522_top10.5034_recall@50.8152_mrr0.6400.pt"
CONFIG_PATH="/data/home/wudezhi/project/school/x-meta/gutclip/configs/default.yaml"
DATA_DIR="/data/home/wudezhi/project/school/x-meta/datasets/nicheng/nicheng_count_matrix_output"
OUTPUT_DIR="/data/home/wudezhi/project/school/x-meta/datasets/nicheng/embedding_output"
TREE_PATH="/data/home/wudezhi/project/school/x-meta/datasets/raw/tree/newick.txt"
OTU_LIST_PATH="/data/home/wudezhi/project/school/x-meta/datasets/raw/tree/otu.csv"

# 并行处理参数
BATCH_SIZE=32
NUM_WORKERS=32
USE_MULTIPROCESS=false
DEVICE="auto"
NUM_INSTANCES=16

# 计算总文件数和每个实例处理的文件数
TOTAL_FILES=$(find "$DATA_DIR" -name "*.tsv" | wc -l)
FILES_PER_INSTANCE=$((TOTAL_FILES / NUM_INSTANCES))

echo "=========================================="
echo "Nicheng Parallel Processing with tmux"
echo "=========================================="
echo "Model: $MODEL_PATH"
echo "Config: $CONFIG_PATH"
echo "Data Directory: $DATA_DIR"
echo "Output Directory: $OUTPUT_DIR"
echo "Tree File: $TREE_PATH"
echo "OTU List: $OTU_LIST_PATH"
echo "Batch Size: $BATCH_SIZE"
echo "Workers: $NUM_WORKERS"
echo "Multiprocess: $USE_MULTIPROCESS"
echo "Device: $DEVICE"
echo "Total TSV files: $TOTAL_FILES"
echo "Number of instances: $NUM_INSTANCES"
echo "Files per instance: $FILES_PER_INSTANCE"
echo "=========================================="

# 检查必需文件
if [ ! -f "$MODEL_PATH" ]; then
    echo "Error: Model file not found: $MODEL_PATH"
    exit 1
fi

if [ ! -f "$CONFIG_PATH" ]; then
    echo "Error: Config file not found: $CONFIG_PATH"
    exit 1
fi

if [ ! -d "$DATA_DIR" ]; then
    echo "Error: Data directory not found: $DATA_DIR"
    exit 1
fi

if [ ! -f "$TREE_PATH" ]; then
    echo "Error: Tree file not found: $TREE_PATH"
    exit 1
fi

if [ ! -f "$OTU_LIST_PATH" ]; then
    echo "Error: OTU list file not found: $OTU_LIST_PATH"
    exit 1
fi

# 检查tmux是否安装
if ! command -v tmux &> /dev/null; then
    echo "Error: tmux is not installed. Please install tmux first."
    exit 1
fi

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# 创建日志目录
LOG_DIR="logs_nicheng_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

echo "Log directory: $LOG_DIR"
echo "Starting $NUM_INSTANCES tmux sessions..."

# 启动16个tmux会话
for i in $(seq 0 $((NUM_INSTANCES-1))); do
    START_IDX=$((i * FILES_PER_INSTANCE))
    END_IDX=$((START_IDX + FILES_PER_INSTANCE))
    
    # 最后一个实例处理剩余的所有文件
    if [ $i -eq $((NUM_INSTANCES-1)) ]; then
        END_IDX=$TOTAL_FILES
    fi
    
    SESSION_NAME="nicheng_$i"
    LOG_FILE="$LOG_DIR/instance_${i}.log"
    
    echo "Starting session $SESSION_NAME: files $START_IDX to $((END_IDX-1))"
    
    # 创建tmux会话并运行命令
    if [ "$USE_MULTIPROCESS" = "true" ]; then
        # 使用多进程模式
        tmux new-session -d -s "$SESSION_NAME" \
            "python gutclip/cmdline/extract_taxa_embeddings_v2_streaming.py \
                --model_path '$MODEL_PATH' \
                --cfg '$CONFIG_PATH' \
                --data_dir '$DATA_DIR' \
                --output_dir '$OUTPUT_DIR' \
                --tree_path '$TREE_PATH' \
                --otu_list_path '$OTU_LIST_PATH' \
                --batch_size $BATCH_SIZE \
                --num_workers $NUM_WORKERS \
                --device '$DEVICE' \
                --start_idx $START_IDX \
                --end_idx $END_IDX \
                2>&1 | tee '$LOG_FILE'"
    else
        # 使用单进程模式（你的原始设置）
        tmux new-session -d -s "$SESSION_NAME" \
            "python gutclip/cmdline/extract_taxa_embeddings_v2_streaming.py \
                --model_path '$MODEL_PATH' \
                --cfg '$CONFIG_PATH' \
                --data_dir '$DATA_DIR' \
                --output_dir '$OUTPUT_DIR' \
                --tree_path '$TREE_PATH' \
                --otu_list_path '$OTU_LIST_PATH' \
                --batch_size $BATCH_SIZE \
                --num_workers $NUM_WORKERS \
                --device '$DEVICE' \
                --no_multiprocess \
                --start_idx $START_IDX \
                --end_idx $END_IDX \
                2>&1 | tee '$LOG_FILE'"
    fi
    
    # 等待一下再创建下一个会话
    sleep 1
done

echo "=========================================="
echo "All tmux sessions created successfully!"
echo "=========================================="
echo ""
echo "Useful commands:"
echo "  - List all sessions: tmux list-sessions"
echo "  - Attach to session i: tmux attach-session -t nicheng_i"
echo "  - Kill session i: tmux kill-session -t nicheng_i"
echo "  - Kill all sessions: tmux kill-server"
echo ""
echo "Monitor progress:"
echo "  - Watch all logs: tail -f $LOG_DIR/instance_*.log"
echo "  - Check specific log: tail -f $LOG_DIR/instance_0.log"
echo "  - Count completed files: ls $OUTPUT_DIR/*.pt | wc -l"
echo ""
echo "Session details:"
for i in $(seq 0 $((NUM_INSTANCES-1))); do
    START_IDX=$((i * FILES_PER_INSTANCE))
    END_IDX=$((START_IDX + FILES_PER_INSTANCE))
    if [ $i -eq $((NUM_INSTANCES-1)) ]; then
        END_IDX=$TOTAL_FILES
    fi
    echo "  nicheng_$i: files $START_IDX to $((END_IDX-1))"
done
