#!/bin/bash

# set folder to the path of the directory containing the images
folder="videos/video_1000_1000"

# iterate over each jpeg file in the directory and its subdirectories
find "$folder" -type f -iname "*.jpeg" | while read -r file
do
    # extract the directory and filename from the file's path
    dir=$(dirname "$file")
    base=$(basename "$file" .jpeg)

    # separate the name from the number at the last underscore
    name=$(echo "$base" | rev | cut -d'_' -f2- | rev)
    num=$(echo "$base" | rev | cut -d'_' -f1 | rev)

    # add zero-padding to the number
    num_padded=$(printf "%04d" "$num")

    # construct the new filename
    newfile="${dir}/${name}_${num_padded}.jpeg"

    # rename the file
    mv -- "$file" "$newfile"
done