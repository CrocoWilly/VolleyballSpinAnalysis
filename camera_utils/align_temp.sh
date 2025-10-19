#!/bin/bash
camset_dir=./camsets/camset_temp
python -m camera_utils.stream_align --camset "${camset_dir}" \
    --cameras A D --sources 0 3 --is_device \
    --display_size 1600