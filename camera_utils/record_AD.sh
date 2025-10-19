#!/bin/bash
camset_dir=./camsets/camset_0211
python -m camera_utils.stream_record --camset "${camset_dir}" \
    --cameras A D --sources 0 3 --is_device \
    --display_size 1600