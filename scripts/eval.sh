#!/bin/bash

# 设置环境变量
export PYTHONPATH=$PYTHONPATH:$(pwd)
export CUDA_VISIBLE_DEVICES=0  # 使用第一块GPU

# 默认参数
CKPT_PATH="runs/gutclip/checkpoints/last.ckpt"  # 模型权重路径
TEST_JSONL="data/test_pairs.jsonl"             # 测试集路径
OUT_DIR="eval_out"                             # 输出目录
BATCH_SIZE=256                                 # 批处理大小
DEVICE="cuda"                                  # 计算设备
FP16=true                                      # 是否使用半精度

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --ckpt)
            CKPT_PATH="$2"
            shift 2
            ;;
        --test_jsonl)
            TEST_JSONL="$2"
            shift 2
            ;;
        --out_dir)
            OUT_DIR="$2"
            shift 2
            ;;
        --bs)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --no_fp16)
            FP16=false
            shift
            ;;
        --no_vis)
            NO_VIS="--no_vis"
            shift
            ;;
        *)
            echo "Unknown parameter: $1"
            exit 1
            ;;
    esac
done

# 检查必要文件是否存在
if [ ! -f "$CKPT_PATH" ]; then
    echo "Error: Model checkpoint not found at $CKPT_PATH"
    exit 1
fi

if [ ! -f "$TEST_JSONL" ]; then
    echo "Error: Test JSONL file not found at $TEST_JSONL"
    exit 1
fi

# 创建输出目录
mkdir -p "$OUT_DIR"

# 构建命令
CMD="python gutclip/cmdline/eval.py \
    --ckpt $CKPT_PATH \
    --test_jsonl $TEST_JSONL \
    --bs $BATCH_SIZE \
    --device $DEVICE \
    --out_dir $OUT_DIR"

# 添加可选参数
if [ "$FP16" = true ]; then
    CMD="$CMD --fp16"
fi

if [ ! -z "$NO_VIS" ]; then
    CMD="$CMD $NO_VIS"
fi

# 打印运行信息
echo "=== Running GutCLIP Evaluation ==="
echo "Checkpoint: $CKPT_PATH"
echo "Test data: $TEST_JSONL"
echo "Output dir: $OUT_DIR"
echo "Batch size: $BATCH_SIZE"
echo "Device: $DEVICE"
echo "FP16: $FP16"
echo "Visualization: $([ -z "$NO_VIS" ] && echo "Enabled" || echo "Disabled")"
echo "================================="

# 运行评估
echo "Running command: $CMD"
eval $CMD

# 检查运行状态
if [ $? -eq 0 ]; then
    echo "Evaluation completed successfully!"
    echo "Results saved to: $OUT_DIR"
else
    echo "Evaluation failed!"
    exit 1
fi
