result_dir="file_result"
video_dir="data/02all"
camset_dir="./camsets/camset_0211"
rm -r "${result_dir}"
mkdir -p $result_dir
for video in `ls $video_dir/HDR80_A*.mov`; do
    video_name=$(basename "$video")
    video_name=${video_name%.*}
    echo "Processing $video_name"
    video_result_dir="$result_dir/$video_name"
    # if [ -d "$video_result_dir" ]; then
    #     echo "Directory $video_result_dir already exists. Skipping."
    #     continue
    # fi
    mkdir -p $video_result_dir
    python rally_rw.py \
        --camset $camset_dir \
        --source $video \
        --loglevel SUCCESS \
        --outdir $video_result_dir > ${video_result_dir}/log.txt 2>&1
    if [ $? -ne 0 ]; then
        echo "Error processing $video"
        # exit 1
    fi
done
echo Result directory: $result_dir
# ./run_scripts/comp_mp4_folder.sh ./stream_resultff