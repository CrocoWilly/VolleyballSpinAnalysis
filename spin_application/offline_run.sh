#!/bin/bash

# 設定模型與相機組合
MODEL="yolov8n_conti_1280_v1.pt"
CAMSET="camsets/camset_1013_all"
SCRIPT="rallyWithSpin_20250207_1730.py"

# 資料夾遍歷
DATA_DIR="data"
OUTPUT_DIR="test_results"

# 確保輸出資料夾存在
mkdir -p "$OUTPUT_DIR"

# 遍歷所有 1015-01 ~ 1015-30 的資料夾
for folder in $(seq -w 11 30); do
    FOLDER_PATH="$DATA_DIR/1015-$folder"

    # 確保資料夾存在
    if [ -d "$FOLDER_PATH" ]; then
        echo "Processing folder: $FOLDER_PATH"
        
        # 遍歷資料夾內所有以 "HDR80_A" 開頭的影片檔案
        for file in "$FOLDER_PATH"/HDR80_A*.mov; do
            # 檢查是否有匹配的檔案
            if [ -f "$file" ]; then
                filename=$(basename -- "$file")
                filename_noext="${filename%.*}"
                
                # 設定輸出資料夾
                output_subdir="$OUTPUT_DIR/${filename_noext}_detect"
                mkdir -p "$output_subdir"
                
                # 執行 Python 指令
                echo "Processing file: $file"
                python "$SCRIPT" --model "$MODEL" --camset "$CAMSET" --source "$file" --outdir "$output_subdir"
            fi
        done
    fi
done

echo "Processing all video completed."
