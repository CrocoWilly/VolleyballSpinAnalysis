#!/bin/bash
# camera_names=(A D)
name=$1
if [ -z "$name" ]; then
    echo "Usage: $0 <name>"
    exit 1
fi
camset_dir=./camsets/camset_temp
mkdir -p "${camset_dir}"
camera_dir="${camset_dir}/${name}"
python camera_utils/chessboard_annotate_all.py --cam "${camera_dir}"

