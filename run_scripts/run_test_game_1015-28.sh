videos=($(ls ./data/1015-28/HDR80_A_*.mov))
game_dir="./game_result/1015-28"
camset_dir="./camsets/camset_1013_all"
declare -a cameras=("A" "D")
caemras_str="A,D"
model="yolov8n_conti_1280_v1.pt"
mkdir -p $game_dir
for video in "${videos[@]}"; do
    video_name=$(basename $video)
    video_stem="${video_name%.*}"
    rally_dir="${game_dir}/$video_stem"
    # if [ -d "$rally_dir" ]; then
    #     echo "Skip $rally_dir"
    #     continue
    # fi
    mkdir -p $rally_dir
    log_path="${rally_dir}/log.txt"
    echo "Processing $video", dir= ${rally_dir}/, log to $log_path
    # swap the HDR80_A_ to HDR80_?_ to get every videos
    videos_str=""
    for camera in "${cameras[@]}"; do
        videos_str+="$(echo $video | sed -e "s/_A_/_${camera}_/g") "
    done
    python rally.py \
        --model $model \
        --camset $camset_dir \
        --source $video \
        --loglevel SUCCESS \
        --outdir $rally_dir > $log_path 2>&1
done
cur_date=$(date +"%Y-%m-%d %H:%M:%S.%3N")
echo Completed $cur_date
