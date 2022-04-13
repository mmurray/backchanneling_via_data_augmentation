#!/usr/bin/env bash
# Pass a path to a folder full of mp4 videos
data_path=$1
for file in "$data_path"/*.mp4; do
  wav_file="$data_path/$(basename ${file%.*}).wav"
  if [[ ! -f "$wav_file" ]]; then
    ffmpeg -n -i "$file" -ac 1 -f wav "$wav_file"
  fi
done
