#!/bin/bash

python analyze_spikes.py --gamedir device_result
python make_analyze_video.py device_result --noteam