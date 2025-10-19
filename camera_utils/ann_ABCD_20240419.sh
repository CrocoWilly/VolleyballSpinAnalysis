#!/bin/bash
camset_dir=./camsets/camset_20240419
# for cname in A B C D; do
#     ( python camera_utils/annotate.py --cam "${camset_dir}/${cname}" --ext & )
# done
# for cname in A B C D; do
python -m camera_utils.annotate --cam "${camset_dir}/D" --label --type main --img "${camset_dir}/D/samples/331.jpg"
