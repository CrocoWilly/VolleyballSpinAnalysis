#!/bin/bash
result_dir="./result/20240125"
mkdir -p $result_dir
video_dir="./data/20240125raw"
for video in $video_dir/HDR80_A_*; do
    video_name=$(basename $video)
    echo "Processing $video_name"
    log_path="${result_dir}/$(basename $video)_log.txt"
    echo Log to $log_path
    # continue
    python rally.py --camset ./camsets/camset_20240125 --cameras A,D \
        --source $video --outdir $result_dir > $log_path 2>&1
done

./run_scripts/comp_mp4_folder.sh $result_dir