mkdir -p ./device_result
# rm ./device_result/*.mp4
python rallyWithSpin_20250408_1130.py --camset ./camsets/camset_temp \
    --model yolov8n_mikasa_1280_v1.pt \
    --source "0,3" --cameras A,D \
    --stream --is_device --display \
    --outdir ./device_result > log.txt 2>&1
# ./run_scripts/comp_mp4_folder.sh ./stream_result