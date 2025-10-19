#!/bin/bash
ffmpeg -f v4l2 -list_formats all -i /dev/video0