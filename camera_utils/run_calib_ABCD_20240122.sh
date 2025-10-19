#!/bin/bash
camset_dir=./camsets/camset_20240122
# python camera_utils/aruco_annotate.py --cam "${camset_dir}/A" --video "${camset_dir}/A.mov" --ref --onlymid
# python camera_utils/aruco_annotate.py --cam "${camset_dir}/B" --video "${camset_dir}/B.mov" --ref --onlymid
# python camera_utils/aruco_annotate.py --cam "${camset_dir}/C" --video "${camset_dir}/C.mov" --ref --onlymid
# python camera_utils/aruco_annotate.py --cam "${camset_dir}/D" --video "${camset_dir}/D.mov" --ref --onlymid
python -m camera_utils.calibrate_tool --cam "${camset_dir}/A"
python -m camera_utils.calibrate_tool --cam "${camset_dir}/B"
python -m camera_utils.calibrate_tool --cam "${camset_dir}/C"
python -m camera_utils.calibrate_tool --cam "${camset_dir}/D"
python -m camera_utils.calibrate_tool --camset "${camset_dir}" --cams A,B,C,D
python -m camera_utils.epipolar_test --camset "${camset_dir}"
ffmpeg -i "${camset_dir}/A_D_epiline.mp4" -c:v libx264 -crf 18 -preset slow -c:a copy "${camset_dir}/A_D_epiline_re.mp4"
