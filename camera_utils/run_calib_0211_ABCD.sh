#!/bin/bash
camset_dir=./camsets/camset_0211_ABCD_retry

# python camera_utils/annotate.py --cam "${camset_dir}/B" --ext
# python camera_utils/annotate.py --cam "${camset_dir}/C" --ext
# exit

# python camera_utils/annotate.py --cam "${camset_dir}/C" --label
# python camera_utils/annotate.py --cam "${camset_dir}/B" --label
# python camera_utils/annotate.py --cam "${camset_dir}/A" --label
# python camera_utils/annotate.py --cam "${camset_dir}/D" --label
# exit

python -m camera_utils.calibrate_tool --cam "${camset_dir}/A" --base ./camsets/camset_tvl20/A --infix
python -m camera_utils.calibrate_tool --cam "${camset_dir}/B" --base ./camsets/camset_tvl20/A --inguess
python -m camera_utils.calibrate_tool --cam "${camset_dir}/D" --base ./camsets/camset_tvl20/A --inguess
# python -m camera_utils.calibrate_tool --cam "${camset_dir}/C" --base ./camsets/base/A --infix
python -m camera_utils.calibrate_tool --camset "${camset_dir}" --cams A,B,D
#python -m camera_utils.epipolar_test --camset "${camset_dir}"
#ffmpeg -i "${camset_dir}/A_D_epiline.mp4" -c:v libx264 -crf 18 -preset slow -c:a copy "${camset_dir}/A_D_epiline_re.mp4"#
python -m camera_utils.camset_epipole_test ${camset_dir}