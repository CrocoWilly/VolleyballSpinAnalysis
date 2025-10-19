mkdir -p ./device_result
# rm ./device_result/*.mp4
sudo conda run -n vt2 python rally.py --camset ./camsets/camset_temp \
    --model yolov8n_conti_1280_v1.pt \
    --source "0,3" --cameras A,D \
    --stream --is_device --display \
    --outdir ./device_result > /dev/null 2>&1
# ./run_scripts/comp_mp4_folder.sh ./stream_result