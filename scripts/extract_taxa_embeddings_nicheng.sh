#!/bin/bash

# Extract taxa embeddings from nicheng count matrix data using pretrained GutCLIP model
# This script processes all TSV files in the nicheng dataset and extracts tree embeddings

# Set working directory
cd /data/home/wudezhi/project/school/x-meta

# Configuration
MODEL_PATH="/data/home/wudezhi/project/school/x-meta/checkpoints/gutclip_exp_best_20250724-090522_top10.5034_recall@50.8152_mrr0.6400.pt"
CONFIG_PATH="/data/home/wudezhi/project/school/x-meta/gutclip/configs/default.yaml"
DATA_DIR="/data/home/wudezhi/project/school/x-meta/datasets/nicheng/nicheng_count_matrix_output"
OUTPUT_DIR="/data/home/wudezhi/project/school/x-meta/datasets/nicheng/embedding_output"
TREE_PATH="/data/home/wudezhi/project/school/x-meta/datasets/raw/tree/newick.txt"
OTU_LIST_PATH="/data/home/wudezhi/project/school/x-meta/datasets/raw/tree/otu.csv"
START_IDX=1601
END_IDX=2001


# Script parameters
BATCH_SIZE=32
NUM_WORKERS=32  # Workers for sample-level multiprocessing
USE_MULTIPROCESS=false  # Set to false to disable multiprocessing
DEVICE="auto"

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Print configuration
echo "=========================================="
echo "GutCLIP Taxa Embedding Extraction"
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
echo "Start Index: $START_IDX"
echo "End Index: $END_IDX"
echo "=========================================="

# Check if required files exist
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

# Count TSV files
TSV_COUNT=$(find "$DATA_DIR" -name "*.tsv" | wc -l)
echo "Found $TSV_COUNT TSV files to process"
echo ""

# Run the extraction script
echo "Starting embedding extraction..."
if [ "$USE_MULTIPROCESS" = "true" ]; then
    echo "Using multiprocessing mode..."
    python gutclip/cmdline/extract_taxa_embeddings_v2_streaming.py \
        --model_path "$MODEL_PATH" \
        --cfg "$CONFIG_PATH" \
        --data_dir "$DATA_DIR" \
        --output_dir "$OUTPUT_DIR" \
        --tree_path "$TREE_PATH" \
        --otu_list_path "$OTU_LIST_PATH" \
        --batch_size "$BATCH_SIZE" \
        --num_workers "$NUM_WORKERS" \
        --device "$DEVICE"
else
    echo "Using single process mode..."
    python gutclip/cmdline/extract_taxa_embeddings_v2_streaming.py \
        --model_path "$MODEL_PATH" \
        --cfg "$CONFIG_PATH" \
        --data_dir "$DATA_DIR" \
        --output_dir "$OUTPUT_DIR" \
        --tree_path "$TREE_PATH" \
        --otu_list_path "$OTU_LIST_PATH" \
        --batch_size "$BATCH_SIZE" \
        --num_workers "$NUM_WORKERS" \
        --device "$DEVICE" \
        --no_multiprocess \
        --start_idx "$START_IDX" \
        --end_idx "$END_IDX"
fi

# Check if the script completed successfully
if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "Extraction completed successfully!"
    echo "Output files:"
    echo "  - Individual files: $OUTPUT_DIR/*.pt (one per TSV file, .tsv suffix removed)"
    echo "  - Example: sample1.tsv -> $OUTPUT_DIR/sample1.pt"
    echo "=========================================="
else
    echo ""
    echo "=========================================="
    echo "Extraction failed! Please check the error messages above."
    echo "=========================================="
    exit 1
fi
