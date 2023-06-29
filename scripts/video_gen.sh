video_folder="video_3000_1000/layer_5"
video_name=$(basename $(dirname "$video_folder"))
ffmpeg -framerate 24  -i "${video_folder}/Raw_Attention_3%03d.jpeg" "${video_name}.mp4"
