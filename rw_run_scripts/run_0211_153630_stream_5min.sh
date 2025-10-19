rm -r ./stream_result_5min/
python rally_rw.py --camset ./camsets/camset_0211 \
    --source ./data/0212_5min/HDR80_A_Live_20230212_160854_000.mov_5min.mov \
    --stream --loglevel INFO --write-full \
    --outdir ./stream_result_5min > log.txt 2>&1
./run_scripts/comp_mp4_folder.sh ./stream_result_5min