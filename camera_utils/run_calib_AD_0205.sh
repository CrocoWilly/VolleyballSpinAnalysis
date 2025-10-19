#!/bin/bash
camset_dir=./camsets/camset_0205
# both use A as base because it's good somehow
python -m camera_utils.calibrate_tool --cam "${camset_dir}/A" --base ./camsets/base/A --inguess
python -m camera_utils.calibrate_tool --cam "${camset_dir}/D" --base ./camsets/base/A --inguess
python -m camera_utils.calibrate_tool --camset "${camset_dir}" --cams A,D 
python -m camera_utils.epipolar_test --camset "${camset_dir}"
ffmpeg -i "${camset_dir}/A_D_epiline.mp4" -c:v libx264 -crf 18 -preset slow -c:a copy "${camset_dir}/A_D_epiline_re.mp4"