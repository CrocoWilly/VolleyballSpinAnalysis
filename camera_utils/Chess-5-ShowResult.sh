#!/bin/bash
function join_by { local IFS="$1"; shift; echo "$*"; }

camera_names=(A D)
camset_dir=./camsets/camset_temp
# mkdir -p "${camset_dir}"
# for cname in "${camera_names[@]}"; do
#     camera_dir="${camset_dir}/${cname}"
#     python -m camera_utils.calibrate_tool --cam "${camera_dir}"
# done
# cams_str=$(join_by , "${camera_names[@]}")
python -m camera_utils.calibrate_tool --show --camset "${camset_dir}"
