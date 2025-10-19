#!/bin/bash
# Usage example: to record /dev/video0 with name A, 
# run: ./Chess-Record-Single.sh 0 A
source=$1
name=$2
camera_names=(A D)
camset_dir=./camsets/camset_temp
python -m camera_utils.stream_pure_record --sources $source --names $name --is_device \
    --display_size 1600 --outdir ${camset_dir} --outfmt "chess_{}.mp4"