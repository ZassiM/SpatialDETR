video_folder="videos/video_0_1000/layer_0"
video_name=$(basename $(dirname "$video_folder"))
ffmpeg -y -framerate 24 -pattern_type glob -i "${video_folder}"/*_*.jpeg -c:v libx264 "${video_name}.mp4"
