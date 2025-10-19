#!/bin/bash
camera_names=(A D)
camset_dir=./camsets/camset_temp
mkdir -p "${camset_dir}"
for cname in "${camera_names[@]}"; do
    camera_dir="${camset_dir}/${cname}"
    # python -m camera_utils.annotate --cam "${camera_dir}" --label --type main --img "${camera_dir}/samples/1.jpg"
    python -m camera_utils.annotate --cam "${camera_dir}" --label --type main --scale 0.75
done

