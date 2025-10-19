rm ./stream_result/*.mp4
python rally.py --camset ./camsets/camset_1013_all --source "0,3" --stream --is_device --cameras A,D --outdir ./stream_result > log.txt 2>&1
./run_scripts/comp_mp4_folder.sh ./stream_result