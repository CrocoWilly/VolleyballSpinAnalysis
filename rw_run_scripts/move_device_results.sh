source_dir=./device_result
# dest_root=/mnt/DATA3/
dest_root=/home/share/pre_device_result
timestamp=`date +%Y%m%d_%H%M%S`
dest_dir=${dest_root}/device_result_${timestamp}
mkdir -p ${dest_dir}
# Move files
mv ${source_dir}/* ${dest_dir}