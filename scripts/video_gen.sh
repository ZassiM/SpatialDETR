video_folder="videos/video_0_1000/layer_0"
video_name=$(basename $(dirname "$video_folder"))
ffmpeg -framerate 24 -i "videos/video_0_6019/layer_0/Raw_Attention_%04d.jpeg" "${video_name}.mp4"
