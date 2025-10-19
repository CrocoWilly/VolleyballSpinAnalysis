rm ./stream_result_1min/*.mp4
commit_hash=`git rev-parse --short HEAD`
log="perf_test/log_${commit_hash}.txt"
echo "Logging to $log"
viztracer -o trace/result_1min.json --tracer_entries 5000000 -- rally_rw.py --camset ./camsets/camset_0211 \
    --source ./data/0212_1min/HDR80_A_Live_20230212_160854_000.mov_1min.mov \
    --stream --loglevel SUCCESS --write-full \
    --outdir ./stream_result_1min > $log 2>&1
# ./run_scripts/comp_mp4_folder.sh ./stream_result