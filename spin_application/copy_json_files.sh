#!/bin/bash

# 定義來源與目標資料夾
SOURCE_DIR="test_results"
DEST_DIR="json_results"

# 確保目標資料夾存在
mkdir -p "$DEST_DIR"

# 遍歷 test_results 內的所有子資料夾
for dir in "$SOURCE_DIR"/*/; do
    # 檢查是否為資料夾
    if [ -d "$dir" ]; then
        # 取得資料夾名稱
        folder_name=$(basename "$dir")
        # 定義來源檔案路徑
        source_file="$dir/ball_data.json"
        # 定義目標檔案路徑
        dest_file="$DEST_DIR/${folder_name}_ball_data.json"
        
        # 檢查 ball_data.json 是否存在
        if [ -f "$source_file" ]; then
            cp "$source_file" "$dest_file"
            echo "Copied: $source_file -> $dest_file"
        else
            echo "Warning: $source_file not found!"
        fi
    fi
done