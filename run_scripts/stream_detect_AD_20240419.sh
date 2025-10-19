#!/bin/bash
camset_dir=./camsets/camset_20240419
python stream_detect.py --camset "${camset_dir}" \
    --cameras A D --sources 0 3 --is_device \
    --display_size 1600