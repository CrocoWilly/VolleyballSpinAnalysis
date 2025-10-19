#!/bin/bash

# 定義來源與目標資料夾
SOURCE_DIR="test_results"
DEST_DIR="video_A_results"

# 確保目標資料夾存在
mkdir -p "$DEST_DIR"

# 遍歷 test_results 內的所有子資料夾
for dir in "$SOURCE_DIR"/*/; do
    # 檢查是否為資料夾
    if [ -d "$dir" ]; then
        # 查找並複製所有以 "HDR80_A" 開頭的檔案
        for file in "$dir"/HDR80_A*; do
            if [ -f "$file" ]; then
                cp "$file" "$DEST_DIR"
                echo "Copied: $file -> $DEST_DIR"
            fi
        done
    fi
done
