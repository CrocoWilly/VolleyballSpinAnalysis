#!/bin/bash
ref_file="camera_utils/ref_files/court4pt.json"  # 4 corners of court
camera_names=(A D)
camset_dir=./camsets/camset_temp
mkdir -p "${camset_dir}"
for cname in "${camera_names[@]}"; do
    camera_dir="${camset_dir}/${cname}"
    mkdir -p "${camera_dir}"
    mkdir -p "${camera_dir}/refs"
    cp "${ref_file}" "${camera_dir}/refs/main.json"
done