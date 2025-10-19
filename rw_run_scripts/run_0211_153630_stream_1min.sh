rm ./stream_result/*.mp4
python rally_rw.py --camset ./camsets/camset_0211 \
    --source ./data/0212_1min/HDR80_A_Live_20230212_160854_000.mov_1min.mov \
    --stream --loglevel INFO --write-full \
    --outdir ./stream_result > log.txt 2>&1
# ./run_scripts/comp_mp4_folder.sh ./stream_resultff