#!/bin/bash
camset_dir=./camsets/camset_20240419
python -m camera_utils.stream_align --camset "${camset_dir}" \
    --cameras A D --sources "${camset_dir}/A.mov" "${camset_dir}/D.mov" \
    --display_size 1600