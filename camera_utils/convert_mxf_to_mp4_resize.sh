#!/bin/bash
ffmpeg -i ./camsets/chesstest_20240803/T/320_0006.MXF -s 1920x1080 -c:v libx264  ./camsets/chesstest_20240803/T/chess_sample2.mp4