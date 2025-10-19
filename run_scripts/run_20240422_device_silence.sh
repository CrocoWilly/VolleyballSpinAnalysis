mkdir -p ./device_result
# rm ./device_result/*.mp4
python rally.py --camset ./camsets/camset_20240419 \
    --source "0,3" --cameras A,D \
    --stream --is_device --display \
    --outdir ./device_result > /dev/null 2>&1
# ./run_scripts/comp_mp4_folder.sh ./stream_result