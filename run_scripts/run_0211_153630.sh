python rally.py \
    --loglevel DEBUG \
    > log2.txt 2>&1
./run_scripts/mp4compress.sh ./result/HDR80_A_Live_20230211_153630_000.mov_detect.mp4 ./result/HDR80_A_Live_20230211_153630_000.mov_detect.mp4_comp.mp4 
./run_scripts/mp4compress.sh ./result/HDR80_D_Live_20230211_153630_000.mov_detect.mp4 ./result/HDR80_D_Live_20230211_153630_000.mov_detect.mp4_comp.mp4 