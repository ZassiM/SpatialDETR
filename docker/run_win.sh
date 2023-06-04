DOCKER_NAME=spatial_detr_ma_zahr

# path to directory where nusenes data is stored
nusc_data_dir="/mnt/c/Users/wasso/Desktop/dataset/nuscenes"
# path to directory where models / logs shall be stored in
work_dirs="/mnt/c/Users/wasso/Desktop/Project/SpatialDETR/work_dirs"
# path to this repository root
repo_dir=$PWD

xhost +

docker run \
--rm \
--name $DOCKER_NAME \
--env DISPLAY=${DISPLAY} \
--shm-size=16gb \
--net=host \
--volume /tmp/.X11-unix:/tmp/.X11-unix:rw \
--volume $HOME/.Xauthority:/root/.Xauthority:rw \
--gpus 'all,"capabilities=compute,utility,graphics"' \
--mount source=$repo_dir,target=/workspace,type=bind,consistency=cached \
--mount source=$work_dirs,target=/workspace/work_dirs,type=bind,consistency=cached \
--mount source=$nusc_data_dir,target=/workspace/data/nuscenes,type=bind,consistency=cached \
-it \
$DOCKER_NAME

