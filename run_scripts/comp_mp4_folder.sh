#!/bin/bash
folder=$1
regex='^Rally.*mp4'
for file in $folder/*; do
    if [[ $(basename $file) =~ $regex ]]; then
        echo $file
        # swap file name, prepend with comp_
        file_comp="$(dirname $file)/Comp_$(basename $file)"
        echo Compressed to $file_comp
        ./run_scripts/mp4compress.sh $file $file_comp
    fi
done

regex='^Full.*mp4'
for file in $folder/*; do
    if [[ $(basename $file) =~ $regex ]]; then
        echo $file
        # swap file name, prepend with comp_
        file_comp="$(dirname $file)/Comp_$(basename $file)"
        echo Compressed to $file_comp
        ./run_scripts/mp4compress.sh $file $file_comp
    fi
done