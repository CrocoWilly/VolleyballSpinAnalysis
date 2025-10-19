rm ./stream_result/*.mp4
python rally.py --camset ./camsets/camset_0211 \
    --source ./data/0212_10min/HDR80_A_Live_20230212_160854_000.mov \
    --stream --loglevel DEBUG --write-full \
    --outdir ./stream_result > log.txt 2>&1
# ./run_scripts/comp_mp4_folder.sh ./stream_result