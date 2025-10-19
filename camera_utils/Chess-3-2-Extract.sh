#!/bin/bash
# Put the sample video as CAMERA_DIR/CAMERA_NAME.mp4 or 
name=$1
camera_names=(A D)
camset_dir=./camsets/camset_temp
mkdir -p "${camset_dir}"
# for cname in "${camera_names[@]}"; do
#     camera_dir="${camset_dir}/${cname}"
#     python -m camera_utils.annotate --cam "${camera_dir}" --ext
# done
camera_dir="${camset_dir}/${name}"
python -m camera_utils.annotate --cam "${camera_dir}" --ext