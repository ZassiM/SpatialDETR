#!/bin/bash

IMAGE_NAME=xai_ma_zahr
CONTAINER_NAME=xai

# Path to directory where nuScenes Trainval dataset is stored
nusc_train_val_data_dir="/work/beemelmanns/nuscenes"
# Path to directory where models / logs shall be stored in
work_dirs="/work/beemelmanns/work_dirs"
# Path to this repository root
repo_dir="/work/beemelmanns/xai"

# --user $(id -u):$(id -g) \

docker run \
--name $CONTAINER_NAME \
--rm \
--gpus 'all,"capabilities=compute,utility,graphics"' \
--shm-size=16g \
--net=host \
--env DISPLAY=${DISPLAY} \
--volume /tmp/.X11-unix:/tmp/.X11-unix:rw \
--volume $HOME/.Xauthority:/root/.Xauthority:rw \
--mount source=$repo_dir,target=/workspace,type=bind,consistency=cached \
--mount source=$nusc_train_val_data_dir,target=/workspace/data/nuscenes,type=bind,consistency=cached \
--mount source=$work_dirs,target=/workspace/work_dirs,type=bind,consistency=cached \
-it \
$IMAGE_NAME
