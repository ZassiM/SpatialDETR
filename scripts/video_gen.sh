video_folder="videos/video_X/layer_0"
video_name=$(basename $(dirname "$video_folder"))

start_number=$(ls "${video_folder}/Raw_Attention_"*.jpeg | sort -V | head -n 1)
start_number=$(basename "$start_number" .jpeg) # strip the directory and extension
start_number=${start_number#Raw_Attention_} # strip the prefix

ffmpeg -framerate 24 -start_number ${start_number} -i "${video_folder}/Raw_Attention_%04d.jpeg" -c:v libx264 -preset slow -crf 18 "${video_name}.mp4"