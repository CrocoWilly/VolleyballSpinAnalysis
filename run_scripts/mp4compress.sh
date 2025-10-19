#!/bin/bash
ffmpeg -i $1 -vcodec h264 -crf 18 -acodec mp2 $2 -y
